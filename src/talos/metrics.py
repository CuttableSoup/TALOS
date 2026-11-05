"""Module 9 — metrics + sweep driver (pandas/parquet/joblib).

Depletion check: N * R0 = n0 * ln(10) / T for 90% removal in T years.
"""

import math


def required_removal_rate(n0: float, years: float, fraction: float = 0.9) -> float:
    """Total removals/yr (N * R0) to remove ``fraction`` of n0 in ``years``."""
    return -n0 * math.log(1.0 - fraction) / years


def run_sweep(*args, **kwargs):
    raise NotImplementedError("module 9")
