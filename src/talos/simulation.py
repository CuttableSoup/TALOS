"""End-to-end engagement loop: screen -> greedy pick -> burst -> dv -> removal check.

Resource gating is the module-8 battery + heat-buffer state machine: a
platform becomes ready when a full ``burst_s`` burst fits, then selects a target,
settles for ``atp.settle_s`` and fires for up to the resource-allowed duration.
Duty cycle is an output.

Simplifications (documented, to revisit):
  * Time is processed in chunks; each object is engaged at most once per chunk
    (across all platforms), and windows are recomputed each chunk.
  * Burst dv is applied impulsively at the burst midpoint.
  * Platforms are processed sequentially within a chunk.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .constants import R_EARTH, YEAR
from .debris import DebrisPopulation
from .engagement import AblationParams, integrate_burst, pulse_times
from .lifetime import apply_dv, is_removed, orbital_lifetime
from .propagator import propagate
from .platform import Platform
from .resources import ResourceState
from .scheduler import burst_states, evaluate_candidates, pick
from .screening import screen_windows

if TYPE_CHECKING:
    from .dynamics import PopulationDynamics
    from .planner import CommitPlanner
    from .sun import SunEnvironment


@dataclass(frozen=True)
class EngagementEvent:
    platform: str
    debris_index: int
    t_start: float
    t_stop: float
    n_pulses: int
    n_fired: int
    dv_retrograde: float
    dv_norm: float
    optical_energy_j: float
    perigee_alt_before_m: float
    perigee_alt_after_m: float
    removed: bool
    committed: bool = False  # target was in the planner's commit set


@dataclass
class SimResult:
    events: list[EngagementEvent]
    duration_s: float
    platforms: list[str]
    fire_time_s: dict[str, float]
    removed: np.ndarray  # bool per population sample
    max_duty: float  # steady-state duty ceiling from power/thermal, mean over platforms
    population: DebrisPopulation = field(repr=False)
    # Natural lifetime at sim start and lifetime at sim end, for every engaged object
    lifetime_initial_s: dict[int, float] = field(default_factory=dict)
    lifetime_final_s: dict[int, float] = field(default_factory=dict)
    planner_stats: dict = field(default_factory=dict)
    env_stats: dict = field(default_factory=dict)  # eclipse / heater / brownout, mean per platform
    pop_stats: dict = field(default_factory=dict)  # sources / sinks (PopulationDynamics)
    # risk transfer: time [s] each engaged object spends in each protected shell,
    # natural path (from its initial state) vs after the laser (from its final state)
    shell_names: tuple[str, ...] = ()
    shell_res_initial_s: np.ndarray | None = field(default=None, repr=False)  # (n_hit, k)
    shell_res_final_s: np.ndarray | None = field(default=None, repr=False)

    def records(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def summary(self, credit_min_lifetime_s: float = YEAR, horizon_s: float = 200 * YEAR) -> dict:
        """Engagement and removal counts are physical events and are NOT scaled by
        sample weight: a platform fires at one target per burst. They represent the
        real system only when the run is target-saturated (``saturation`` ~ 1); a
        sparse sample under-counts. Increase N until these metrics converge.

        ``removed_credited`` only counts objects whose natural lifetime exceeded
        ``credit_min_lifetime_s`` (objects about to re-enter anyway earn no credit).
        ``debris_years_removed`` sums lifetime reduction, capped at ``horizon_s``.
        """
        li, lf = self.lifetime_initial_s, self.lifetime_final_s
        credited = [j for j in np.flatnonzero(self.removed) if li.get(int(j), 0.0) > credit_min_lifetime_s]
        dyears = sum(min(li[j], horizon_s) - min(lf[j], horizon_s) for j in li) / YEAR
        useful = [e for e in self.events if e.n_fired > 0 and e.dv_retrograde > 0]
        hours = self.duration_s / 3600.0
        risk = {}
        for k, name in enumerate(self.shell_names):
            nat = float(self.shell_res_initial_s[:, k].sum()) / YEAR
            aft = float(self.shell_res_final_s[:, k].sum()) / YEAR
            risk |= {f"risk_{name}_obj_yr_natural": nat, f"risk_{name}_obj_yr_after": aft,
                     f"risk_{name}_delta_obj_yr_per_yr": (aft - nat) * YEAR / self.duration_s}
        duty = float(np.mean(list(self.fire_time_s.values()))) / self.duration_s
        return {
            "sim_hours": hours,
            "n_platforms": len(self.platforms),
            "n_samples": len(self.removed),
            "useful_engagements": len(useful),
            "useful_engagements_per_hr_per_sat": len(useful) / hours / len(self.platforms),
            "mean_dv_retro_per_engagement": float(np.mean([e.dv_retrograde for e in useful])) if useful else 0.0,
            "duty_cycle_mean": duty,
            "saturation": duty / self.max_duty,
            "removed": int(self.removed.sum()),
            "removed_per_yr": float(self.removed.sum()) * YEAR / self.duration_s,
            "removed_credited": len(credited),
            "removed_credited_per_yr": len(credited) * YEAR / self.duration_s,
            "debris_years_removed": dyears,
            "debris_years_removed_per_yr": dyears * YEAR / self.duration_s,
            **self.env_stats,
            **self.pop_stats,
            **risk,
        }


def _write_elements(pop: DebrisPopulation, j: int, el) -> None:
    for name in ("a", "e", "i", "raan", "argp", "M"):
        getattr(pop.elements, name)[j] = getattr(el, name)


def run_engagements(
    platforms: list[Platform],
    population: DebrisPopulation,
    duration_s: float,
    dt: float = 1.0,
    chunk_s: float = 600.0,
    params: AblationParams = AblationParams(),
    rng: np.random.Generator | None = None,
    batch: int = 1000,
    score: str = "removal_progress",
    planner: "CommitPlanner | None" = None,
    env: "SunEnvironment | None" = None,
    dynamics: "PopulationDynamics | None" = None,
    on_chunk=None,
    risk_shells: "dict[str, tuple[float, float]] | None" = None,
) -> SimResult:
    """Run the engagement loop. With ``planner`` (a CommitPlanner), committed
    objects take priority and fillers that would block a reservation are skipped.
    With ``env`` (a SunEnvironment), platforms follow their illumination timeline
    (array output, eclipse, albedo) and screening applies the sun gates; without
    it platforms are always sunlit and the sun gates are off.
    ``risk_shells`` ({name: (lo_m, hi_m)} altitude bands) adds shell residence of every
    engaged object, natural vs after the laser, to the result (risk transfer).
    With ``dynamics`` (a PopulationDynamics), the whole population decays naturally
    and gains new objects every ``dynamics.cfg.update_s``; naturally decayed
    objects leave the screening set but are not credited as removals.
    ``on_chunk(t_s, events, removed, decayed)`` is called after every chunk (progress
    logging / event checkpoints); it must not modify its arguments."""
    pop = copy.deepcopy(population)
    pop0 = copy.deepcopy(population)  # initial state of every object (incl. ones added later)
    removed = np.zeros(len(pop), dtype=bool)
    decayed = np.zeros(len(pop), dtype=bool)
    touched = np.zeros(len(pop), dtype=bool)  # engaged at least once
    n_initial = len(pop)
    ready_at = {p.name: 0.0 for p in platforms}
    if env is None:
        state = {p.name: ResourceState(p.resources, p.laser) for p in platforms}
    else:
        state = {p.name: ResourceState(p.resources, p.laser,
                                       env=env.timeline(p, duration_s + chunk_s, p.resources.radiator_earth_view),
                                       fire_in_eclipse=env.cfg.fire_in_eclipse)
                 for p in platforms}
    if planner is not None:
        planner.env = env
    fire_time = {p.name: 0.0 for p in platforms}
    events: list[EngagementEvent] = []

    for c0 in np.arange(0.0, duration_s, chunk_s):
        t = c0 + np.arange(0.0, min(chunk_s, duration_s - c0), dt)
        engaged: set[int] = set()
        sun_hat = env.sun_hat(t) if env is not None else None
        if dynamics is not None and dynamics.due(float(c0)):
            n_old = len(pop)
            pop, dmask = dynamics.step(pop, removed | decayed, float(c0), initial=pop0, engaged=touched)
            grow = np.zeros(len(pop) - n_old, dtype=bool)
            removed, decayed = np.concatenate([removed, grow]), np.concatenate([decayed, grow]) | dmask
            touched = np.concatenate([touched, grow])
            if len(pop) > n_old:
                pop0 = pop0.append(pop.subset(np.arange(n_old, len(pop))))
        if planner is not None:
            planner.refresh(float(t[0]), pop, removed | decayed)
        for p_idx, plat in enumerate(platforms):
            wins = screen_windows(plat, pop, t, candidates=np.flatnonzero(~(removed | decayed)), batch=batch,
                                  sun_hat=sun_hat)
            if not wins:
                continue
            k = int(np.searchsorted(t, ready_at[plat.name]))
            while k < len(t):
                live = [w for w in wins if not removed[w.debris_index] and w.debris_index not in engaged]
                open_w = [w for w in live if w.start <= k < w.stop]
                if not open_w:
                    later = [w.start for w in live if w.start > k]
                    if not later:
                        break
                    k = min(later)
                    continue
                rs = state[plat.name]
                rs.advance(max(rs.t, t[k] + plat.atp.settle_s))
                cands = evaluate_candidates(plat, pop, open_w, t, k, params, max_fire_s=rs.max_fire_s())
                is_commit = False
                if planner is not None and cands:
                    mine = [c for c in cands if planner.is_committed(c.debris_index)]
                    rest = [c for c in cands if not planner.is_committed(c.debris_index)]
                    bc = pick(mine, score) if mine else None
                    bf = pick(rest, score) if rest else None
                    if bc is not None and planner.accept(p_idx, bc, bf, plat.atp.settle_s):
                        best, is_commit = bc, True
                    elif bf is None:
                        k += 1
                        continue
                    else:
                        best = bf
                        blocked, resume = planner.filler_blocks(p_idx, rs, best, plat)
                        if blocked:
                            k = max(k + 1, int(np.searchsorted(t, resume)))
                            continue
                else:
                    best = pick(cands, score)
                if best is None:
                    k += 1
                    continue

                j = best.debris_index
                tp = pulse_times(best.t_start, best.t_stop - best.t_start, plat.laser.prf_hz)
                res = integrate_burst(
                    *burst_states(plat, pop, j, tp), plat.laser, pop.areal_density[j], params,
                    jitter_rad=plat.atp.jitter_rad if rng is not None else 0.0, rng=rng,
                )
                tm = 0.5 * (best.t_start + best.t_stop)
                ep = float(pop.epoch_s[j])
                el_tm = propagate(pop.elements[j], tm - ep, pop.ballistic_coeff[j], ep)
                el_new = apply_dv(el_tm, res.dv_vec)
                gone = bool(is_removed(el_new, pop.ballistic_coeff[j], tm))
                _write_elements(pop, j, el_new)
                pop.epoch_s[j] = tm
                removed[j] |= gone
                engaged.add(j)
                touched[j] = True
                if planner is not None:
                    planner.record(j, res.dv_retrograde, best.t_stop, pop, removed)

                fire = best.t_stop - best.t_start
                fire_time[plat.name] += fire
                rs.fire(fire)
                ready_at[plat.name] = rs.t + rs.wait_for(plat.resources.burst_s)
                events.append(EngagementEvent(
                    platform=plat.name, debris_index=j, t_start=best.t_start, t_stop=best.t_stop,
                    n_pulses=res.n_pulses, n_fired=res.n_fired, dv_retrograde=res.dv_retrograde,
                    dv_norm=float(np.linalg.norm(res.dv_vec)), optical_energy_j=res.optical_energy_j,
                    perigee_alt_before_m=float(el_tm.rp - R_EARTH),
                    perigee_alt_after_m=float(el_new.rp - R_EARTH), removed=gone, committed=is_commit,
                ))
                k = int(np.searchsorted(t, ready_at[plat.name]))
        if on_chunk is not None:
            on_chunk(float(min(c0 + chunk_s, duration_s)), events, removed, decayed)

    max_duty = float(np.mean([s.steady_state_max_duty() for s in state.values()]))
    hit = np.array(sorted({e.debris_index for e in events}), dtype=int)
    li, lf = {}, {}
    names = tuple(risk_shells) if risk_shells else ()
    shells = np.array([risk_shells[n] for n in names], float) if names else None
    res_i = res_f = np.zeros((len(hit), len(names)))
    if len(hit):
        # both lifetimes measured from each object's initial epoch
        e0, e1 = pop0.elements[hit], pop.elements[hit]
        B = pop.ballistic_coeff[hit]
        t0, t1 = pop0.epoch_s[hit], pop.epoch_s[hit]
        L0 = orbital_lifetime(e0.a, e0.e, B, t0_s=t0, shells_m=shells)
        L1 = orbital_lifetime(e1.a, e1.e, B, t0_s=t1, shells_m=shells)
        if shells is not None:
            (L0, res_i), (L1, res_f) = L0, L1
        li = dict(zip(hit.tolist(), np.atleast_1d(L0).tolist()))
        lf = dict(zip(hit.tolist(), (np.atleast_1d(L1) + t1 - t0).tolist()))
    sts = list(state.values())
    for s in sts:
        s.advance(max(s.t, duration_s))
    yr = YEAR / duration_s
    shots = {p: 0 for p in fire_time}
    for e in events:
        shots[e.platform] += e.n_pulses
    cap = np.mean([s.cfg.battery_capacity_j for s in sts])
    env_stats = {  # hardware life (roadmap 4), mean per platform; cycle limits are not modelled
        "bursts_per_sat_per_yr": float(np.mean([s.bursts for s in sts])) * yr,
        "laser_shots_per_sat_per_yr": float(np.mean(list(shots.values()))) * yr,
        "battery_efc_per_sat_per_yr": float(np.mean([s.discharge_j for s in sts])) / cap * yr,
        "battery_dod_mean": float(sum(s.burst_dod_sum for s in sts) / max(1, sum(s.bursts for s in sts))),
        "battery_dod_max": float(max(s.burst_dod_max for s in sts)),
    }
    if env is not None:
        env_stats |= {
            "eclipse_fraction": float(np.mean([s.env.eclipse_fraction for s in sts])),
            "heater_kwh_per_day": float(np.mean([s.heater_j for s in sts])) / 3.6e6 * 86400.0 / duration_s,
            "brownout_s": float(np.mean([s.brownout_s for s in sts])),
        }
    pop_stats = {}
    if dynamics is not None:
        live = int((~(removed | decayed)).sum())
        pop_stats = {
            **dynamics.stats,
            "pop_initial": n_initial,
            "pop_final_live": live,
            "net_change_per_yr": (live - n_initial) * YEAR / duration_s,
            "natural_decays_per_yr": dynamics.stats["natural_decays"] * YEAR / duration_s,
        }
    return SimResult(events, duration_s, [p.name for p in platforms], fire_time, removed, max_duty, pop, li, lf,
                     dict(planner.stats, committed_now=len(planner.committed)) if planner is not None else {},
                     env_stats, pop_stats, names, res_i, res_f)
