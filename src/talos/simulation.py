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

from . import _kernels
from .constants import R_EARTH, YEAR
from .debris import DebrisPopulation
from .engagement import AblationParams, integrate_burst, pulse_times
from .lifetime import apply_dv, is_removed, orbital_lifetime, shell_fraction
from .propagator import propagate
from .platform import Platform
from .resources import ResourceState
from .campaign import focus_need, focus_summary, reference_impulse, select_focus
from .scheduler import burst_states, evaluate_candidates, pick
from .screening import chunk_cache, screen_windows

if TYPE_CHECKING:
    from .campaign import CampaignConfig
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
    # Mean elements right after the burst, at epoch t_elements_s (mid-burst): what the
    # object flies from then on (playback / orbit reconstruction). NaN in old checkpoints.
    t_elements_s: float = float("nan")
    a_after_m: float = float("nan")
    e_after: float = float("nan")
    i_after_rad: float = float("nan")
    raan_after_rad: float = float("nan")
    argp_after_rad: float = float("nan")
    M_after_rad: float = float("nan")
    focus: bool = False  # target was in the campaign's focus set (talos.campaign)


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
    # natural path (from its initial state) vs the lasered path: in-run residence (accrued daily,
    # elements held between accruals) + residence after its final state
    shell_names: tuple[str, ...] = ()
    shell_res_initial_s: np.ndarray | None = field(default=None, repr=False)  # (n_hit, k)
    shell_res_final_s: np.ndarray | None = field(default=None, repr=False)
    # summed natural-path shell residence [s] of every fragment shed by ablation (new risk)
    shell_res_fragments_s: np.ndarray | None = field(default=None, repr=False)
    # fresh-platform restart (resume with fresh_platforms): the run covers [t_start_s, duration_s]
    # and counts only removals after it; objects removed before are flagged here
    t_start_s: float = 0.0
    removed_at_start: np.ndarray | None = field(default=None, repr=False)
    # focus campaign (talos.campaign): focus-set indices, dv still needed at start and end
    focus_idx: np.ndarray | None = field(default=None, repr=False)
    focus_need0: np.ndarray | None = field(default=None, repr=False)
    focus_need1: np.ndarray | None = field(default=None, repr=False)

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
        rem = self.removed.copy()
        if self.removed_at_start is not None:
            rem[:len(self.removed_at_start)] &= ~self.removed_at_start
        span = self.duration_s - self.t_start_s
        credited = [j for j in np.flatnonzero(rem) if li.get(int(j), 0.0) > credit_min_lifetime_s]
        dyears = sum(min(li[j], horizon_s) - min(lf[j], horizon_s) for j in li) / YEAR
        useful = [e for e in self.events if e.n_fired > 0 and e.dv_retrograde > 0]
        hours = span / 3600.0
        risk = {}
        for k, name in enumerate(self.shell_names):
            nat = float(self.shell_res_initial_s[:, k].sum()) / YEAR
            aft = float(self.shell_res_final_s[:, k].sum()) / YEAR
            risk |= {f"risk_{name}_obj_yr_natural": nat, f"risk_{name}_obj_yr_after": aft,
                     f"risk_{name}_delta_obj_yr_per_yr": (aft - nat) * YEAR / span}
            if self.shell_res_fragments_s is not None:
                risk[f"risk_{name}_obj_yr_fragments"] = float(self.shell_res_fragments_s[k]) / YEAR
        duty = float(np.mean(list(self.fire_time_s.values()))) / span
        return {
            "sim_hours": hours,
            "n_platforms": len(self.platforms),
            "n_samples": len(self.removed),
            "useful_engagements": len(useful),
            "useful_engagements_per_hr_per_sat": len(useful) / hours / len(self.platforms),
            "mean_dv_retro_per_engagement": float(np.mean([e.dv_retrograde for e in useful])) if useful else 0.0,
            "duty_cycle_mean": duty,
            "saturation": duty / self.max_duty,
            "removed": int(rem.sum()),
            "removed_per_yr": float(rem.sum()) * YEAR / span,
            "removed_credited": len(credited),
            "removed_credited_per_yr": len(credited) * YEAR / span,
            "debris_years_removed": dyears,
            "debris_years_removed_per_yr": dyears * YEAR / span,
            **self.env_stats,
            **self.pop_stats,
            **risk,
            **(focus_summary(self.focus_idx, self.focus_need0, self.focus_need1, rem, self.population,
                             self.events) if self.focus_idx is not None else {}),
        }


def _ablation_fragments(pop: DebrisPopulation, j: int, el, tm: float, params: AblationParams,
                        rng: np.random.Generator) -> DebrisPopulation | None:
    """Fragments >= 1 cm shed by object ``j`` at sim time ``tm`` (state ``el`` after the burst):
    Poisson count, sizes a fraction of the parent's, isotropic dv spread, NASA SBM A/m,
    parent's material, albedo and sample weight. None if none survive."""
    from .elements import elements_to_rv, rv_to_elements
    from .population import sbm_area_to_mass

    n = int(rng.poisson(params.fragment_count_mean))
    if n == 0:
        return None
    lo, hi = params.fragment_size_frac
    d = pop.diameter_m[j] * rng.uniform(lo, hi, n)
    r, v = elements_to_rv(el)
    fel = rv_to_elements(np.broadcast_to(r, (n, 3)).copy(),
                         v + rng.normal(0.0, params.fragment_dv_sigma_m_s, (n, 3)))
    keep = np.flatnonzero((d >= 0.01) & (fel.rp - R_EARTH > 150e3))
    if not len(keep):
        return None
    d = d[keep]
    return DebrisPopulation(elements=fel[keep], diameter_m=d, areal_density=1.0 / sbm_area_to_mass(d, rng),
                            weight=pop.weight[j], albedo=pop.albedo[j], epoch_s=tm, material=pop.material[j])


def _write_elements(pop: DebrisPopulation, j: int, el) -> None:
    for name in ("a", "e", "i", "raan", "argp", "M"):
        getattr(pop.elements, name)[j] = getattr(el, name)
    pop.invalidate_drag(j)


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
    checkpoint_path=None,
    checkpoint_every_s: float | None = None,
    checkpoint_meta: dict | None = None,
    resume: dict | None = None,
    predict_pulse_stride: int = 1,
    knowledge=None,
    campaign: "CampaignConfig | None" = None,
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
    logging / event checkpoints); it must not modify its arguments.

    Checkpoints: with ``checkpoint_path`` and ``checkpoint_every_s`` the full loop state is
    pickled at chunk starts every ``checkpoint_every_s`` (plus ``checkpoint_meta``, e.g. the
    scenario). ``resume`` (a dict from ``load_checkpoint``) continues from that state with
    the same inputs (platforms, population, rng, dynamics, planner as built for the original
    run); ``duration_s`` may be longer than the original. A resumed run reproduces the
    uninterrupted one exactly.
    ``campaign`` (talos.campaign.CampaignConfig, enabled): focus set chosen at the start (or at a
    fresh-platform restart) gets priority; see talos.campaign."""
    pop = copy.deepcopy(population)
    pop.cm_scale = np.asarray(params.cm_scale_for(pop.material), float)  # laser coupling by material
    pop0 = copy.deepcopy(pop)  # initial state of every object (incl. ones added later)
    pending: list[DebrisPopulation] = []  # ablation fragments, added at the next chunk start
    frag_stats = {"ablation_fragmentations": 0, "ablation_fragments": 0}
    frag_idx: list[int] = []  # population indices of ablation fragments
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
    ready_s = {p.name: p.resources.ready_s(p.laser) for p in platforms}  # wait for this burst length
    events: list[EngagementEvent] = []

    names = tuple(risk_shells) if risk_shells else ()
    shells = np.array([risk_shells[n] for n in names], float) if names else None
    res_sim = np.zeros((len(pop), len(names)))  # shell residence during the run [s]
    res_t = 0.0  # residence accumulated up to this sim time
    c_start = 0.0
    fresh = resume is not None and bool(resume.get("fresh_platforms", False))
    ref = None  # fresh restart: population state at the restart, the baseline for lifetimes / risk
    removed_at_start = None
    dyn_stats0: dict = {}
    if resume is not None:
        r = resume
        if r["chunk_s"] != chunk_s or r["dt"] != dt:
            raise ValueError("checkpoint was written with a different chunk_s / dt")
        if not fresh and r["platforms"] != [p.name for p in platforms]:
            raise ValueError("checkpoint was written with a different constellation")
        pop, pop0, removed, decayed, touched = r["pop"], r["pop0"], r["removed"], r["decayed"], r["touched"]
        c_start = r["c0"]
        pending, frag_stats = r.get("pending", []), r.get("frag_stats", frag_stats)
        frag_idx = list(r.get("frag_idx", []))
        if fresh:
            # new hardware over the same sky: platforms, resources and tallies start at the
            # restart; the population (and its dynamics / generators) continue
            pop.cm_scale = np.asarray(params.cm_scale_for(pop.material), float)  # this run's coupling
            removed_at_start = removed.copy()
            n_initial = int((~(removed | decayed)).sum())
            ref = copy.deepcopy(pop)
            res_sim, res_t = np.zeros((len(pop), len(names))), c_start
            for st in state.values():
                st.t = c_start
            frag_stats = {k: 0 for k in frag_stats}
            frag_idx = []
        else:
            n_initial, ready_at, fire_time, events = r["n_initial"], r["ready_at"], r["fire_time"], r["events"]
            res_sim, res_t = r["res_sim"], r["res_t"]
            if knowledge is not None and r.get("knowledge") is not None:
                knowledge.__dict__.update(r["knowledge"])
            for p in platforms:  # resource states; the illumination timeline is rebuilt for this duration
                st = r["state"][p.name]
                st.env = state[p.name].env
                state[p.name] = st
        if rng is not None:
            rng.bit_generator.state = r["rng"]
        if dynamics is not None:
            dynamics.next_update, dynamics.stats = r["dynamics"]["next_update"], dict(r["dynamics"]["stats"])
            dynamics.rng.bit_generator.state = r["dynamics"]["rng"]
            dyn_stats0 = dict(dynamics.stats) if fresh else {}
        if planner is not None and not fresh:
            planner.__dict__.update(r["planner"])
            planner.env = env
    use_focus = campaign is not None and campaign.enabled
    focus_idx = focus_need0 = None
    if use_focus:
        if resume is not None and not fresh and resume.get("focus") is not None:
            focus_idx, focus_need0 = resume["focus"]
        else:
            focus_idx = select_focus(campaign, pop, removed | decayed)
            focus_need0 = focus_need(pop, focus_idx)
    focus_mask = np.zeros(len(pop), dtype=bool)
    if use_focus:
        focus_mask[focus_idx] = True
    next_ckpt = (c_start // checkpoint_every_s + 1) * checkpoint_every_s if checkpoint_every_s else np.inf

    def save_checkpoint(c0: float) -> None:
        import os
        import pickle

        sts = {}
        for name, st in state.items():  # drop the (large, rebuildable) illumination timeline
            c = copy.copy(st)
            c.env = None
            sts[name] = c
        blob = {
            "version": 1, "c0": c0, "chunk_s": chunk_s, "dt": dt, "platforms": [p.name for p in platforms],
            "pop": pop, "pop0": pop0, "removed": removed, "decayed": decayed, "touched": touched,
            "n_initial": n_initial, "ready_at": ready_at, "fire_time": fire_time, "events": events,
            "res_sim": res_sim, "res_t": res_t, "state": sts,
            "rng": rng.bit_generator.state if rng is not None else None,
            "dynamics": None if dynamics is None else {"next_update": dynamics.next_update,
                                                       "stats": dict(dynamics.stats),
                                                       "rng": dynamics.rng.bit_generator.state},
            "planner": None if planner is None else {k: v for k, v in planner.__dict__.items() if k != "env"},
            "pending": pending, "frag_stats": frag_stats, "frag_idx": frag_idx,
            "knowledge": None if knowledge is None else dict(knowledge.__dict__),
            "focus": None if focus_idx is None else (focus_idx, focus_need0),
            "meta": checkpoint_meta or {},
        }
        tmp = f"{checkpoint_path}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, checkpoint_path)

    def flush_pending() -> None:
        """Append queued ablation fragments to the population."""
        nonlocal pop, pop0, pending, removed, decayed, touched
        n_old = len(pop)
        for f in pending:
            pop = pop.append(f)
        pending = []
        new = np.arange(n_old, len(pop))
        pop.cm_scale[new] = params.cm_scale_for(pop.material[new])
        grow = np.zeros(len(new), dtype=bool)
        removed, decayed, touched = (np.concatenate([x, grow]) for x in (removed, decayed, touched))
        pop0 = pop0.append(pop.subset(new))
        frag_idx.extend(new.tolist())

    def accrue(t_now: float) -> None:
        """Shell residence of live objects over [res_t, t_now], elements held at their current values."""
        nonlocal res_t, res_sim
        if shells is None or t_now <= res_t:
            return
        if len(res_sim) < len(pop):
            res_sim = np.vstack([res_sim, np.zeros((len(pop) - len(res_sim), len(names)))])
        el = pop.elements
        near = np.flatnonzero(~(removed | decayed) & (el.rp - R_EARTH < shells[:, 1].max())
                              & (el.a * (1 + el.e) - R_EARTH >= shells[:, 0].min()))
        if len(near):
            res_sim[near] += shell_fraction(el.a[near], el.e[near], shells) * (t_now - res_t)
        res_t = t_now

    for c0 in np.arange(c_start, duration_s, chunk_s):
        if checkpoint_path is not None and c0 >= next_ckpt - 1e-6:
            save_checkpoint(float(c0))
            next_ckpt = (c0 // checkpoint_every_s + 1) * checkpoint_every_s
        t = c0 + np.arange(0.0, min(chunk_s, duration_s - c0), dt)
        if c0 - res_t >= 86400.0 - 1e-6:
            accrue(float(c0))
        engaged: set[int] = set()
        sun_hat = env.sun_hat(t) if env is not None else None
        if dynamics is not None and dynamics.due(float(c0)):
            n_old = len(pop)
            pop, dmask = dynamics.step(pop, removed | decayed, float(c0), initial=pop0, engaged=touched)
            grow = np.zeros(len(pop) - n_old, dtype=bool)
            removed, decayed = np.concatenate([removed, grow]), np.concatenate([decayed, grow]) | dmask
            touched = np.concatenate([touched, grow])
            if len(pop) > n_old:
                new = np.arange(n_old, len(pop))
                pop.cm_scale[new] = params.cm_scale_for(pop.material[new])
                pop0 = pop0.append(pop.subset(new))
        if pending:  # fragments shed by ablation during the previous chunk
            flush_pending()
        if knowledge is not None:
            knowledge.new_chunk(float(c0))
        if planner is not None:
            planner.refresh(float(t[0]), pop, removed | decayed)
        # kernel element arguments for the whole population, shared by all platforms this chunk.
        # Objects engaged later in the chunk keep stale args, but they are excluded via ``engaged``.
        # Also used for candidate prediction: engaged objects are excluded from both.
        cache = chunk_cache(pop) if _kernels.HAVE_NUMBA else None
        for p_idx, plat in enumerate(platforms):
            live_idx = (focus_idx[~(removed | decayed)[focus_idx]] if use_focus and campaign.exclusive
                        else np.flatnonzero(~(removed | decayed)))
            wins = screen_windows(plat, pop, t, candidates=live_idx, batch=batch,
                                  sun_hat=sun_hat, cache=cache)
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
                cands = evaluate_candidates(plat, pop, open_w, t, k, params, max_fire_s=rs.max_fire_s(),
                                            pulse_stride=predict_pulse_stride,
                                            elem_args=None if cache is None else cache.args,
                                            state_error=(knowledge.state_error if knowledge is not None
                                                         and knowledge.has_state_error else None))
                is_commit = False
                is_focus = False
                if use_focus and cands:
                    if len(focus_mask) < len(pop):  # objects added since (sources, fragments) are not focus
                        focus_mask = np.concatenate([focus_mask, np.zeros(len(pop) - len(focus_mask), bool)])
                    i_ref = campaign.min_impulse_frac * reference_impulse(
                        min(plat.resources.burst_s, rs.max_fire_s()), plat.laser, params)
                    fc = [c for c in cands if focus_mask[c.debris_index]
                          and c.dv_retrograde * float(pop.laser_areal_density(c.debris_index)) >= i_ref]
                    bfoc = pick(fc, score) if fc else None
                    if campaign.exclusive:
                        if bfoc is None:
                            k += 1
                            continue
                        is_focus = True
                    elif bfoc is not None:
                        imp = lambda c: c.dv_retrograde * float(pop.laser_areal_density(c.debris_index))  # noqa: E731
                        is_focus = imp(bfoc) >= campaign.quality * max(imp(c) for c in cands)
                if is_focus:
                    best = bfoc
                elif planner is not None and cands:
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
                    *burst_states(plat, pop, j, tp), plat.laser, float(pop.laser_areal_density(j)), params,
                    jitter_rad=plat.atp.jitter_rad if rng is not None else 0.0, rng=rng,
                    pointing_bias=knowledge.pointing_bias() if knowledge is not None and rng is not None else None,
                )
                tm = 0.5 * (best.t_start + best.t_stop)
                ep = float(pop.epoch_s[j])
                el_tm = propagate(pop.elements[j], tm - ep, adot=pop.drag_adot(j))
                el_new = apply_dv(el_tm, res.dv_vec)
                gone = bool(is_removed(el_new, pop.ballistic_coeff[j], tm))
                if params.fragment_prob_per_burst > 0 and rng is not None \
                        and rng.uniform() < params.fragment_prob_per_burst:
                    f = _ablation_fragments(pop, j, el_new, tm, params, rng)
                    frag_stats["ablation_fragmentations"] += 1
                    if f is not None:
                        pending.append(f)
                        frag_stats["ablation_fragments"] += len(f)
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
                ready_at[plat.name] = rs.t + rs.wait_for(ready_s[plat.name])
                events.append(EngagementEvent(
                    platform=plat.name, debris_index=j, t_start=best.t_start, t_stop=best.t_stop,
                    n_pulses=res.n_pulses, n_fired=res.n_fired, dv_retrograde=res.dv_retrograde,
                    dv_norm=float(np.linalg.norm(res.dv_vec)), optical_energy_j=res.optical_energy_j,
                    perigee_alt_before_m=float(el_tm.rp - R_EARTH),
                    perigee_alt_after_m=float(el_new.rp - R_EARTH), removed=gone, committed=is_commit,
                    t_elements_s=float(tm), a_after_m=float(el_new.a), e_after=float(el_new.e),
                    i_after_rad=float(el_new.i), raan_after_rad=float(el_new.raan),
                    argp_after_rad=float(el_new.argp), M_after_rad=float(el_new.M), focus=is_focus,
                ))
                k = int(np.searchsorted(t, ready_at[plat.name]))
        if on_chunk is not None:
            on_chunk(float(min(c0 + chunk_s, duration_s)), events, removed, decayed)

    accrue(float(duration_s))
    if pending:  # fragments from the last chunk
        flush_pending()
    res_frag = None
    if shells is not None and params.fragment_prob_per_burst > 0:
        res_frag = np.zeros(len(names))
        if frag_idx:
            fi = np.array(frag_idx)
            ef = pop0.elements[fi]
            _, rf = orbital_lifetime(ef.a, ef.e, pop0.ballistic_coeff[fi], t0_s=pop0.epoch_s[fi], shells_m=shells)
            res_frag = rf.sum(axis=0)
    max_duty = float(np.mean([s.steady_state_max_duty() for s in state.values()]))
    hit = np.array(sorted({e.debris_index for e in events}), dtype=int)
    li, lf = {}, {}
    res_i = res_f = np.zeros((len(hit), len(names)))
    if len(hit):
        # both lifetimes measured from each object's initial epoch (fresh restart: its state then)
        base = pop0 if ref is None else ref.append(pop0.subset(np.arange(len(ref), len(pop0))))
        e0, e1 = base.elements[hit], pop.elements[hit]
        B = pop.ballistic_coeff[hit]
        t0, t1 = base.epoch_s[hit], pop.epoch_s[hit]
        L0 = orbital_lifetime(e0.a, e0.e, B, t0_s=t0, shells_m=shells)
        L1 = orbital_lifetime(e1.a, e1.e, B, t0_s=t1, shells_m=shells)
        if shells is not None:
            (L0, res_i), (L1, res_f) = L0, L1
            res_f = res_f + res_sim[hit]  # lasered path = in-run residence + after the final state
        li = dict(zip(hit.tolist(), np.atleast_1d(L0).tolist()))
        lf = dict(zip(hit.tolist(), (np.atleast_1d(L1) + t1 - t0).tolist()))
    sts = list(state.values())
    for s in sts:
        s.advance(max(s.t, duration_s))
    span = duration_s - (c_start if fresh else 0.0)
    yr = YEAR / span
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
        # storage end of life (damage = 1), worst platform: cycle damage at this run's cycling plus
        # calendar ageing (Miner's rule), 1 / (cycle damage per yr + 1 / calendar life)
        "storage_life_yr": float(min(_storage_life_yr(s.storage_damage / (span / YEAR),
                                                      s.cfg.storage_calendar_life_yr) for s in sts)),
        "storage_cycle_life_yr": float(min((span / YEAR) / s.storage_damage if s.storage_damage > 0 else np.inf
                                           for s in sts)),
        "ready_burst_s": float(np.mean(list(ready_s.values()))),
    }
    if env is not None:
        env_stats |= {
            "eclipse_fraction": float(np.mean([s.env.eclipse_fraction for s in sts])),
            "heater_kwh_per_day": float(np.mean([s.heater_j for s in sts])) / 3.6e6 * 86400.0 / span,
            "brownout_s": float(np.mean([s.brownout_s for s in sts])),
        }
    pop_stats = dict(frag_stats) if params.fragment_prob_per_burst > 0 else {}
    if dynamics is not None:
        live = int((~(removed | decayed)).sum())
        dstats = {k: v - dyn_stats0.get(k, 0) for k, v in dynamics.stats.items()}
        pop_stats = {
            **pop_stats,
            **dstats,
            "pop_initial": n_initial,
            "pop_final_live": live,
            "net_change_per_yr": (live - n_initial) * YEAR / span,
            "natural_decays_per_yr": dstats["natural_decays"] * YEAR / span,
        }
    return SimResult(events, duration_s, [p.name for p in platforms], fire_time, removed, max_duty, pop, li, lf,
                     dict(planner.stats, committed_now=len(planner.committed)) if planner is not None else {},
                     env_stats, pop_stats, names, res_i, res_f, res_frag,
                     c_start if fresh else 0.0, removed_at_start,
                     focus_idx, focus_need0, focus_need(pop, focus_idx) if use_focus else None)


def _storage_life_yr(cycle_damage_per_yr: float, calendar_life_yr: float) -> float:
    rate = cycle_damage_per_yr + (1.0 / calendar_life_yr if calendar_life_yr > 0 else np.inf)
    return 1.0 / rate if rate > 0 else np.inf


def load_checkpoint(path) -> dict:
    """Read a checkpoint written by ``run_engagements(checkpoint_path=...)``.
    Only load files you wrote yourself: this is a pickle."""
    import pickle

    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if blob.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version {blob.get('version')!r}")
    return blob
