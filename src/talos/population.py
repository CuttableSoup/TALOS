"""Module 1 — debris population sampling.

``ordem_population`` samples NASA ORDEM 3.2 (altitude, inclination, size,
material); see talos.ordem for how the table is derived. ``toy_population`` is
a uniform stand-in for development and tests.
TODO: ESA MASTER-2408 sampler once SDUP access is in (cross-check vs ORDEM).
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH
from .debris import DebrisPopulation
from .elements import KeplerElements
from .constants import YEAR
from .lifetime import orbital_lifetime


def toy_population(
    n: int,
    rng: np.random.Generator,
    alt_km: tuple[float, float] = (600.0, 1000.0),
    n_real: float = 1.0e5,  # Phipps' ~100k; MASTER all-LEO 1-10 cm is ~1.2M, band share unsourced
    areal_density: float = 1.0,
    min_lifetime_yr: float | None = None,
    albedo_median: float = 0.1,
    albedo_log_sigma: float = 0.5,
) -> DebrisPopulation:
    """Uniform-in-altitude, near-circular, isotropic-ish population. Not physical.

    ``min_lifetime_yr`` drops objects whose natural lifetime is shorter (a real,
    roughly steady-state population holds few of them). Survivors keep weight
    n_real / n, so the represented count shrinks accordingly.
    """
    a = R_EARTH + rng.uniform(*alt_km, n) * 1e3
    el = KeplerElements(
        a=a,
        e=rng.uniform(0.0, 0.01, n),
        i=np.arccos(rng.uniform(-1.0, 1.0, n)),
        raan=rng.uniform(0, 2 * np.pi, n),
        argp=rng.uniform(0, 2 * np.pi, n),
        M=rng.uniform(0, 2 * np.pi, n),
    )
    pop = DebrisPopulation(
        elements=el,
        diameter_m=10 ** rng.uniform(-2, -1, n),  # 1-10 cm, log-uniform
        areal_density=areal_density,
        weight=n_real / n,
    )
    # drawn last so earlier draws (and existing seeds) are unchanged
    pop.albedo = np.clip(albedo_median * np.exp(albedo_log_sigma * rng.standard_normal(n)), 0.02, 0.6)
    if min_lifetime_yr is not None:
        life = orbital_lifetime(el.a, el.e, pop.ballistic_coeff, max_s=2 * min_lifetime_yr * YEAR)
        pop = pop.subset(np.flatnonzero(life > min_lifetime_yr * YEAR))
    return pop


def ordem_population(
    n: int,
    rng: np.random.Generator,
    table,
    alt_km: tuple[float, float] = (600.0, 1000.0),
    areal_density: float = 1.0,
    areal_density_model: str = "fixed",
    materials: tuple[str, ...] | None = None,
    min_lifetime_yr: float | None = None,
    albedo_median: float = 0.1,
    albedo_log_sigma: float = 0.5,
    epoch_s: float = 0.0,
) -> DebrisPopulation:
    """1-10 cm population drawn from an ORDEM table (talos.ordem.OrdemTable).

    Altitude, inclination, size and material class come from ORDEM; RAAN, argument of
    perigee and mean anomaly are uniform and e is uniform 0-0.01 (ORDEM's eccentricity
    structure is folded into the altitude it is seen at). Each sample represents
    table count / n real objects.

    ``areal_density_model``: "fixed" uses ``areal_density`` for every object; "sphere"
    uses a solid sphere of the ORDEM material density, mu = (2/3) rho d [pick; real
    fragments are flatter, i.e. lower mu].
    """
    from .ordem import MATERIAL_DENSITY, sample_cells

    s = sample_cells(table, n, rng, alt_km, materials)
    el = KeplerElements(
        a=R_EARTH + s["alt_km"] * 1e3,
        e=rng.uniform(0.0, 0.01, n),
        i=s["inc"],
        raan=rng.uniform(0, 2 * np.pi, n),
        argp=rng.uniform(0, 2 * np.pi, n),
        M=rng.uniform(0, 2 * np.pi, n),
    )
    if areal_density_model == "fixed":
        mu = areal_density
    elif areal_density_model == "sphere":
        rho = np.array([MATERIAL_DENSITY[m] for m in table.materials])[s["material"]]
        rho = np.where(np.isfinite(rho), rho, MATERIAL_DENSITY["MD"])  # intacts: medium density
        mu = 2.0 / 3.0 * rho * s["diameter_m"]
    else:
        raise ValueError(f"areal_density_model {areal_density_model!r}: expected 'fixed' or 'sphere'")
    pop = DebrisPopulation(elements=el, diameter_m=s["diameter_m"], areal_density=mu,
                           weight=s["represented"] / n, epoch_s=epoch_s)
    pop.albedo = np.clip(albedo_median * np.exp(albedo_log_sigma * rng.standard_normal(n)), 0.02, 0.6)
    if min_lifetime_yr is not None:
        life = orbital_lifetime(el.a, el.e, pop.ballistic_coeff, max_s=2 * min_lifetime_yr * YEAR)
        pop = pop.subset(np.flatnonzero(life > min_lifetime_yr * YEAR))
    return pop


def sample_master(n: int, rng: np.random.Generator, **kwargs) -> DebrisPopulation:
    raise NotImplementedError("MASTER-2408 fit pending (module 1)")
