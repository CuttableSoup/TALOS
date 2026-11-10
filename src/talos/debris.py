"""Debris population container (structure-of-arrays for vectorized screening)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import KeplerElements


@dataclass
class DebrisPopulation:
    """N sampled debris objects.

    ``weight`` is the number of real objects each sample represents
    (n0 / N for a uniform sample), used to scale results linearly.
    """

    elements: KeplerElements  # shape (N,)
    diameter_m: np.ndarray
    areal_density: np.ndarray  # mu [kg/m^2]
    weight: np.ndarray
    ids: np.ndarray | None = None
    cd: float = 2.2
    albedo: np.ndarray | float = 0.1  # geometric (Lambertian) albedo, for optical detection
    epoch_s: np.ndarray | float = 0.0  # sim time [s] each object's elements refer to
    material: np.ndarray | int = -1  # ORDEM class index (talos.ordem.MATERIALS); -1 = unknown
    cm_scale: np.ndarray | float = 1.0  # laser coupling relative to the nominal Cm (0 = immune)

    def __post_init__(self) -> None:
        n = len(self.elements)
        self.diameter_m = np.broadcast_to(np.asarray(self.diameter_m, float), (n,)).copy()
        self.areal_density = np.broadcast_to(np.asarray(self.areal_density, float), (n,)).copy()
        self.weight = np.broadcast_to(np.asarray(self.weight, float), (n,)).copy()
        self.albedo = np.broadcast_to(np.asarray(self.albedo, float), (n,)).copy()
        self.epoch_s = np.broadcast_to(np.asarray(self.epoch_s, float), (n,)).copy()
        self.material = np.broadcast_to(np.asarray(self.material, np.int8), (n,)).copy()
        self.cm_scale = np.broadcast_to(np.asarray(self.cm_scale, float), (n,)).copy()
        self.ids = np.arange(n) if self.ids is None else np.asarray(self.ids)
        if self.ids.shape != (n,):
            raise ValueError(f"ids shape {self.ids.shape} != ({n},)")
        if np.any(self.areal_density <= 0):
            raise ValueError("areal_density must be positive")

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def area_m2(self) -> np.ndarray:
        return np.pi * self.diameter_m**2 / 4.0

    @property
    def mass_kg(self) -> np.ndarray:
        return self.areal_density * self.area_m2

    @property
    def ballistic_coeff(self) -> np.ndarray:
        """Cd*A/m [m^2/kg]."""
        return self.cd / self.areal_density

    def laser_areal_density(self, idx) -> np.ndarray:
        """mu as seen by the laser for objects ``idx``: dv = eta1 Cm phi / mu with Cm scaled
        per object (cm_scale), i.e. mu / cm_scale (inf for an immune object: no dv)."""
        mu, s = self.areal_density[idx], self.cm_scale[idx]
        return np.where(s > 0, mu / np.where(s > 0, s, 1.0), np.inf)

    @property
    def represented_count(self) -> float:
        return float(self.weight.sum())

    def subset(self, idx) -> DebrisPopulation:
        return DebrisPopulation(
            elements=self.elements[idx],
            diameter_m=self.diameter_m[idx],
            areal_density=self.areal_density[idx],
            weight=self.weight[idx],
            ids=self.ids[idx],
            cd=self.cd,
            albedo=self.albedo[idx],
            epoch_s=self.epoch_s[idx],
            material=self.material[idx],
            cm_scale=self.cm_scale[idx],
        )

    def append(self, other: "DebrisPopulation") -> "DebrisPopulation":
        """New population with ``other``'s objects after this one's (indices preserved)."""
        cat = lambda f: np.concatenate([getattr(self, f), getattr(other, f)])
        el = KeplerElements(**{k: np.concatenate([getattr(self.elements, k), getattr(other.elements, k)])
                               for k in ("a", "e", "i", "raan", "argp", "M")})
        start = int(self.ids.max()) + 1 if len(self) else 0
        return DebrisPopulation(elements=el, diameter_m=cat("diameter_m"), areal_density=cat("areal_density"),
                                weight=cat("weight"), ids=np.concatenate([self.ids, start + np.arange(len(other))]),
                                cd=self.cd, albedo=cat("albedo"), epoch_s=cat("epoch_s"),
                                material=cat("material"), cm_scale=cat("cm_scale"))
