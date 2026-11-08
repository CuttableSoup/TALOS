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

    def __post_init__(self) -> None:
        n = len(self.elements)
        self.diameter_m = np.broadcast_to(np.asarray(self.diameter_m, float), (n,)).copy()
        self.areal_density = np.broadcast_to(np.asarray(self.areal_density, float), (n,)).copy()
        self.weight = np.broadcast_to(np.asarray(self.weight, float), (n,)).copy()
        self.albedo = np.broadcast_to(np.asarray(self.albedo, float), (n,)).copy()
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
        )
