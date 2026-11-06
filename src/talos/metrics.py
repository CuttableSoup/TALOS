"""Module 9 — metrics + sweep driver.

Depletion check: N * R0 = n0 * ln(10) / T for 90% removal in T years.
Sweep driver lives in talos.sweep.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .debris import DebrisPopulation
from .platform import Platform
from .screening import screen_windows


def required_removal_rate(n0: float, years: float, fraction: float = 0.9) -> float:
    """Total removals/yr (N * R0) to remove ``fraction`` of n0 in ``years``."""
    return -n0 * math.log(1.0 - fraction) / years


def raw_encounter_rate(
    platform: Platform,
    population: DebrisPopulation,
    t: np.ndarray,
    range_m: float = 325e3,
) -> float:
    """Real objects/min entering ``range_m`` of the platform (gate 4: ~25/min at 325 km).

    Pure geometry: no slew, field-of-regard or minimum-range gates. Windows
    already open at t[0] are not counted as entries.
    """
    atp = replace(
        platform.atp,
        engage_range_min_m=0.0,
        engage_range_max_m=range_m,
        slew_rate_limit_rad_s=np.inf,
        field_of_regard_half_angle_rad=np.pi,
        earth_clearance_alt_m=-6.4e6,
    )
    wins = screen_windows(replace(platform, atp=atp), population, t)
    entries = sum(population.weight[w.debris_index] for w in wins if w.start > 0)
    return float(entries) / ((t[-1] - t[0]) / 60.0)


def run_sweep(*args, **kwargs):
    """See :func:`talos.sweep.run_sweep`."""
    from .sweep import run_sweep as _run

    return _run(*args, **kwargs)
