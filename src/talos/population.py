"""Module 1 — debris population sampling.

TODO: fit to ESA MASTER-2408 (alt, inc, RAAN, size, mu) once SDUP access is in.
``toy_population`` is a uniform stand-in for development and tests only.
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH
from .debris import DebrisPopulation
from .elements import KeplerElements


def toy_population(
    n: int,
    rng: np.random.Generator,
    alt_km: tuple[float, float] = (600.0, 1000.0),
    n_real: float = 1.0e6,
    areal_density: float = 1.0,
) -> DebrisPopulation:
    """Uniform-in-altitude, near-circular, isotropic-ish population. Not physical."""
    a = R_EARTH + rng.uniform(*alt_km, n) * 1e3
    el = KeplerElements(
        a=a,
        e=rng.uniform(0.0, 0.01, n),
        i=np.arccos(rng.uniform(-1.0, 1.0, n)),
        raan=rng.uniform(0, 2 * np.pi, n),
        argp=rng.uniform(0, 2 * np.pi, n),
        M=rng.uniform(0, 2 * np.pi, n),
    )
    return DebrisPopulation(
        elements=el,
        diameter_m=10 ** rng.uniform(-2, -1, n),  # 1-10 cm, log-uniform
        areal_density=areal_density,
        weight=n_real / n,
    )


def sample_master(n: int, rng: np.random.Generator, **kwargs) -> DebrisPopulation:
    raise NotImplementedError("MASTER-2408 fit pending (module 1)")
