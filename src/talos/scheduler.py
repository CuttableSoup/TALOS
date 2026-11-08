"""Module 6 — engagement scheduler.

When a platform is ready to fire, every open window gets a deterministic
(expected-value) burst prediction, evaluated for all candidates in one batched
call, and the best-scoring target is chosen.

Scores:
  * ``removal_progress`` (default): fraction of the object's remaining
    dv-to-removal this burst delivers, capped at 1. Favours light objects and
    objects already partly lowered, so repeat engagements concentrate dv
    instead of spreading it thin.
  * ``closest_to_removal``: smallest dv still needed after the burst.
  * ``dv_per_energy``: retrograde dv per joule fired (the original greedy rule).

TODO: OR-Tools CP-SAT or Pyomo+HiGHS receding-horizon ILP; cross-platform
handoff of partly lowered objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .debris import DebrisPopulation
from .engagement import AblationParams, predict_bursts, pulse_times
from .lifetime import REMOVAL_PERIGEE_ALT_M
from .platform import Platform
from .propagator import states_at
from .screening import Window


@dataclass(frozen=True)
class Candidate:
    debris_index: int
    t_start: float
    t_stop: float  # end of burst (burst length or window end, whichever first)
    dv_retrograde: float  # predicted
    n_fired: int
    optical_energy_j: float
    dv_required: float  # dv still needed to reach the removal perigee


def dv_to_remove(a, e, target_perigee_alt_m: float = REMOVAL_PERIGEE_ALT_M) -> np.ndarray:
    """Ideal retrograde dv [m/s], applied at apogee, to lower perigee to the removal altitude."""
    a = np.asarray(a, float)
    e = np.asarray(e, float)
    ra = a * (1 + e)
    rp_t = R_EARTH + target_perigee_alt_m
    v_a = np.sqrt(MU_EARTH * (2 / ra - 1 / a))
    a2 = 0.5 * (ra + rp_t)
    v_a2 = np.sqrt(MU_EARTH * (2 / ra - 1 / a2))
    return np.where(a * (1 - e) <= rp_t, 0.0, v_a - v_a2)


def score_dv_per_energy(c: Candidate) -> float:
    return c.dv_retrograde / c.optical_energy_j


def score_removal_progress(c: Candidate) -> float:
    frac = 1.0 if c.dv_required <= 0 else min(c.dv_retrograde / c.dv_required, 1.0)
    return frac + 1e-6 * c.dv_retrograde  # tie-break on dv among completions


def score_closest_to_removal(c: Candidate) -> float:
    """Minimise dv still needed after this burst: strongly concentrates on
    objects already partly lowered (and on low, light ones)."""
    return -(c.dv_required - c.dv_retrograde)


SCORES: dict[str, Callable[[Candidate], float]] = {
    "removal_progress": score_removal_progress,
    "closest_to_removal": score_closest_to_removal,
    "dv_per_energy": score_dv_per_energy,
}


def burst_states(platform: Platform, population: DebrisPopulation, j: int, tp: np.ndarray):
    """Sat position, debris position & velocity at pulse times ``tp`` for object ``j``."""
    r_s, _ = states_at(platform.orbit, tp)
    el = population.elements[j]
    r_d, v_d = states_at(el, tp, population.ballistic_coeff[j], epoch=population.epoch_s[j])
    return r_s, r_d, v_d


def evaluate_candidates(
    platform: Platform,
    population: DebrisPopulation,
    open_windows: list[Window],
    t_grid: np.ndarray,
    k: int,
    params: AblationParams,
    max_fire_s: float | None = None,
    min_fired_frac: float = 0.1,
) -> list[Candidate]:
    """Deterministic burst predictions for windows open at grid index ``k``.

    The burst starts after the ATP settle time and lasts up to
    min(burst_s, ``max_fire_s``), truncated where the window closes. A window
    still open at the end of the grid is allowed to run past it. Bursts shorter
    than ``min_burst_s`` are not considered.
    """
    las, res = platform.laser, platform.resources
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 1.0
    f_max = res.burst_s if max_fire_s is None else min(res.burst_s, max_fire_s)
    if f_max < res.min_burst_s - 1e-9 or not open_windows:
        return []
    min_fired = max(1, int(min_fired_frac * res.min_burst_s * las.prf_hz))
    t0 = t_grid[k] + platform.atp.settle_s

    js = np.array([w.debris_index for w in open_windows])
    t_end = np.array([np.inf if w.stop >= len(t_grid) else t_grid[w.stop - 1] + dt for w in open_windows])
    t1 = np.minimum(t0 + f_max, t_end)
    ok = t1 - t0 >= res.min_burst_s - 1e-9
    if not ok.any():
        return []
    js, t1 = js[ok], t1[ok]

    tp = pulse_times(t0, f_max, las.prf_hz)
    active = tp[:, None] < t1[None, :] - 1e-9
    r_s, _ = states_at(platform.orbit, tp)
    sub = population.elements[js]
    r_d, v_d = states_at(sub, tp, population.ballistic_coeff[js], epoch=population.epoch_s[js])
    pred = predict_bursts(r_s, r_d, v_d, active, las, population.areal_density[js], params)
    need = dv_to_remove(sub.a, sub.e)

    keep = (pred.n_fired >= min_fired) & (pred.dv_retrograde > 0)
    return [
        Candidate(int(js[i]), float(t0), float(t1[i]), float(pred.dv_retrograde[i]),
                  int(pred.n_fired[i]), float(pred.optical_energy_j[i]), float(need[i]))
        for i in np.flatnonzero(keep)
    ]


def pick(
    candidates: list[Candidate],
    score: str | Callable[[Candidate], float] = "removal_progress",
) -> Candidate | None:
    fn = SCORES[score] if isinstance(score, str) else score
    return max(candidates, key=fn) if candidates else None
