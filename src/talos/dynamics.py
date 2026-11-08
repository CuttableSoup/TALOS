"""Population sources and sinks (roadmap item 2).

Applied every ``update_s`` of sim time to the whole population:

* Natural decay (sink): every live object is advanced to the update time under
  J2 + orbit-averaged drag in the installed atmosphere (solar-cycle aware) and
  re-epoched there. Objects meeting the removal criterion (perigee < 200 km or
  lifetime < 30 d) count as ``decayed`` — gone, but not credited to the laser.
* Background source: new objects at ``background_frac_per_yr`` x n_real per
  year (Poisson), drawn from the same toy distribution as the initial population
  (launch debris, small fragmentations, MLI/paint shedding).
* Breakups: Poisson events at ``breakup_rate_per_yr``; each puts
  ``breakup_fragments`` 1-10 cm fragments around a random parent orbit in the
  band with an isotropic Gaussian dv (``breakup_dv_sigma_m_s`` per axis).

New objects get sample weight n_real / n_samples like the rest, so at full
density (N = n_real) each sample is one physical object.
All rates are PICKS pending sourced values (MASTER/ORDEM source terms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import R_EARTH, YEAR
from .debris import DebrisPopulation
from .elements import KeplerElements, elements_to_rv, rv_to_elements
from .lifetime import decay_to, is_removed


@dataclass(frozen=True)
class DynamicsConfig:
    enabled: bool = True
    update_s: float = 86400.0
    background_frac_per_yr: float = 0.02  # PICK: new objects / yr as a fraction of n_real
    breakup_rate_per_yr: float = 0.5  # PICK: breakups / yr in the band
    breakup_fragments: int = 2000  # PICK: 1-10 cm fragments per breakup
    breakup_dv_sigma_m_s: float = 50.0  # PICK: per-axis fragment dv spread


def toy_elements(n: int, rng: np.random.Generator, alt_km=(600.0, 1000.0)) -> KeplerElements:
    """Same distribution as population.toy_population (uniform altitude, isotropic)."""
    a = R_EARTH + rng.uniform(*alt_km, n) * 1e3
    return KeplerElements(a=a, e=rng.uniform(0.0, 0.01, n), i=np.arccos(rng.uniform(-1.0, 1.0, n)),
                          raan=rng.uniform(0, 2 * np.pi, n), argp=rng.uniform(0, 2 * np.pi, n),
                          M=rng.uniform(0, 2 * np.pi, n))


def _objects(el: KeplerElements, rng, t: float, weight: float, areal_density: float,
             albedo_median: float, albedo_log_sigma: float) -> DebrisPopulation:
    n = len(el)
    return DebrisPopulation(
        elements=el, diameter_m=10 ** rng.uniform(-2, -1, n), areal_density=areal_density, weight=weight,
        albedo=np.clip(albedo_median * np.exp(albedo_log_sigma * rng.standard_normal(n)), 0.02, 0.6),
        epoch_s=t,
    )


def breakup_elements(rng, n: int, dv_sigma: float, alt_km=(600.0, 1000.0)) -> KeplerElements:
    parent = toy_elements(1, rng, alt_km)
    r, v = elements_to_rv(parent)
    v_frag = v + rng.normal(0.0, dv_sigma, (n, 3))
    el = rv_to_elements(np.broadcast_to(r, (n, 3)).copy(), v_frag)
    keep = el.rp - R_EARTH > 150e3  # fragments that re-enter at once never enter the population
    return el[np.flatnonzero(keep)]


class PopulationDynamics:
    def __init__(self, cfg: DynamicsConfig, pop_cfg, weight: float, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.pop_cfg = pop_cfg
        self.weight = weight
        self.rng = rng
        self.next_update = cfg.update_s
        self.stats = {"natural_decays": 0, "added_background": 0, "added_breakup": 0, "breakups": 0}

    def due(self, t: float) -> bool:
        return self.cfg.enabled and t >= self.next_update - 1e-9

    def step(self, pop: DebrisPopulation, gone: np.ndarray, t: float) -> tuple[DebrisPopulation, np.ndarray]:
        """Decay + re-epoch live objects to ``t`` and add new ones. Returns (population,
        bool mask of objects that decayed naturally in this step, sized to the new population)."""
        dt = t - (self.next_update - self.cfg.update_s)
        self.next_update += self.cfg.update_s
        live = np.flatnonzero(~gone)
        decayed = np.zeros(len(pop), dtype=bool)
        if len(live):
            el = decay_to(pop.elements[live], pop.ballistic_coeff[live], pop.epoch_s[live], t)
            for k in ("a", "e", "i", "raan", "argp", "M"):
                getattr(pop.elements, k)[live] = getattr(el, k)
            pop.epoch_s[live] = t
            low = np.flatnonzero(el.rp - R_EARTH < 400e3)  # lifetime check only where it can matter
            if len(low):
                rem = is_removed(el[low], pop.ballistic_coeff[live[low]], t)
                decayed[live[low[rem]]] = True
            self.stats["natural_decays"] += int(decayed.sum())
        pc, rng, w = self.pop_cfg, self.rng, self.weight
        new = []
        n_bg = rng.poisson(self.cfg.background_frac_per_yr * pc.n_real * dt / YEAR / w)
        if n_bg:
            new.append(_objects(toy_elements(n_bg, rng, pc.alt_km), rng, t, w, pc.areal_density,
                                pc.albedo_median, pc.albedo_log_sigma))
            self.stats["added_background"] += n_bg
        for _ in range(rng.poisson(self.cfg.breakup_rate_per_yr * dt / YEAR)):
            el = breakup_elements(rng, max(1, int(round(self.cfg.breakup_fragments / w))),
                                  self.cfg.breakup_dv_sigma_m_s, pc.alt_km)
            new.append(_objects(el, rng, t, w, pc.areal_density, pc.albedo_median, pc.albedo_log_sigma))
            self.stats["breakups"] += 1
            self.stats["added_breakup"] += len(el)
        for p in new:
            pop = pop.append(p)
        decayed = np.concatenate([decayed, np.zeros(len(pop) - len(decayed), dtype=bool)])
        return pop, decayed
