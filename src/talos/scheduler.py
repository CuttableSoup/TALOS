"""Module 6 — engagement scheduler (greedy).

At each moment a platform is ready to fire, evaluate every open window with a
deterministic burst prediction and pick the target with the best score
(default: retrograde dv per unit optical energy fired, the README's greedy rule).

TODO: OR-Tools CP-SAT or Pyomo+HiGHS receding-horizon ILP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .debris import DebrisPopulation
from .engagement import AblationParams, BurstResult, integrate_burst, pulse_times
from .platform import Platform
from .propagator import states_at
from .screening import Window


@dataclass(frozen=True)
class Candidate:
    debris_index: int
    t_start: float
    t_stop: float  # end of burst (burst length or window end, whichever first)
    predicted: BurstResult


def score_dv_per_energy(c: Candidate) -> float:
    return c.predicted.dv_retrograde / c.predicted.optical_energy_j


def burst_states(platform: Platform, population: DebrisPopulation, j: int, tp: np.ndarray):
    """Sat position, debris position & velocity at pulse times ``tp`` for object ``j``."""
    r_s, _ = states_at(platform.orbit, tp)
    el = population.elements[j]
    r_d, v_d = states_at(el, tp, population.ballistic_coeff[j])
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
    min_fired = max(1, int(min_fired_frac * res.min_burst_s * las.prf_hz))
    t0 = t_grid[k] + platform.atp.settle_s
    out = []
    for w in open_windows:
        t_end = np.inf if w.stop >= len(t_grid) else t_grid[w.stop - 1] + dt
        t1 = min(t0 + f_max, t_end)
        if t1 - t0 < res.min_burst_s - 1e-9:
            continue
        tp = pulse_times(t0, t1 - t0, las.prf_hz)
        j = w.debris_index
        pred = integrate_burst(*burst_states(platform, population, j, tp), las,
                               population.areal_density[j], params)
        if pred.n_fired >= min_fired and pred.dv_retrograde > 0:
            out.append(Candidate(j, t0, t1, pred))
    return out


def pick(candidates: list[Candidate], score: Callable[[Candidate], float] = score_dv_per_energy) -> Candidate | None:
    return max(candidates, key=score) if candidates else None
