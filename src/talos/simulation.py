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

import numpy as np

from .constants import R_EARTH, YEAR
from .debris import DebrisPopulation
from .engagement import AblationParams, integrate_burst, pulse_times
from .lifetime import apply_dv, is_removed, move_epoch, orbital_lifetime
from .platform import Platform
from .resources import ResourceState
from .scheduler import burst_states, evaluate_candidates, pick
from .screening import screen_windows


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
) -> SimResult:
    pop = copy.deepcopy(population)
    removed = np.zeros(len(pop), dtype=bool)
    ready_at = {p.name: 0.0 for p in platforms}
    state = {p.name: ResourceState(p.resources, p.laser) for p in platforms}
    fire_time = {p.name: 0.0 for p in platforms}
    events: list[EngagementEvent] = []

    for c0 in np.arange(0.0, duration_s, chunk_s):
        t = c0 + np.arange(0.0, min(chunk_s, duration_s - c0), dt)
        engaged: set[int] = set()
        for plat in platforms:
            wins = screen_windows(plat, pop, t, candidates=np.flatnonzero(~removed), batch=batch)
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
                best = pick(evaluate_candidates(plat, pop, open_w, t, k, params, max_fire_s=rs.max_fire_s()), score)
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
                el_tm = move_epoch(pop.elements[j], tm)
                el_new = apply_dv(el_tm, res.dv_vec)
                gone = bool(is_removed(el_new, pop.ballistic_coeff[j]))
                _write_elements(pop, j, move_epoch(el_new, -tm))
                removed[j] |= gone
                engaged.add(j)

                fire = best.t_stop - best.t_start
                fire_time[plat.name] += fire
                rs.fire(fire)
                ready_at[plat.name] = rs.t + rs.wait_for(plat.resources.burst_s)
                events.append(EngagementEvent(
                    platform=plat.name, debris_index=j, t_start=best.t_start, t_stop=best.t_stop,
                    n_pulses=res.n_pulses, n_fired=res.n_fired, dv_retrograde=res.dv_retrograde,
                    dv_norm=float(np.linalg.norm(res.dv_vec)), optical_energy_j=res.optical_energy_j,
                    perigee_alt_before_m=float(el_tm.rp - R_EARTH),
                    perigee_alt_after_m=float(el_new.rp - R_EARTH), removed=gone,
                ))
                k = int(np.searchsorted(t, ready_at[plat.name]))

    max_duty = float(np.mean([s.steady_state_max_duty() for s in state.values()]))
    hit = np.array(sorted({e.debris_index for e in events}), dtype=int)
    li, lf = {}, {}
    if len(hit):
        e0, e1 = population.elements[hit], pop.elements[hit]
        B = pop.ballistic_coeff[hit]
        li = dict(zip(hit.tolist(), np.atleast_1d(orbital_lifetime(e0.a, e0.e, B)).tolist()))
        lf = dict(zip(hit.tolist(), np.atleast_1d(orbital_lifetime(e1.a, e1.e, B)).tolist()))
    return SimResult(events, duration_s, [p.name for p in platforms], fire_time, removed, max_duty, pop, li, lf)
