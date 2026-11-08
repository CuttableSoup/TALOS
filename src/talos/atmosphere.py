"""Solar-activity-dependent atmosphere (roadmap item 3).

* ``exponential``: the static Vallado/CIRA-72 table (legacy; ~F10.7 130-150).
* ``msis_table``: global-mean NRLMSIS 2.1 density vs altitude and F10.7
  (src/talos/data/msis_density.npz, built by scripts/build_density_table.py),
  log-bilinear interpolation, log-linear extrapolation above 1500 km.

Solar activity: ``constant`` F10.7, or ``cycle`` — a smoothed 11-yr cycle
anchored on cycle 25 (minimum Dec 2019, maximum ~Oct 2024), cosine rise and
fall between ``f107_min`` and ``f107_max``, repeating. F10.7A = F10.7 (the
table assumes the 81-day mean equals the daily value); Ap fixed in the table.

The propagator and lifetime code call ``propagator.density(alt, t_s)``, which
uses the atmosphere installed with ``set_atmosphere`` (default: exponential).
Time ``t_s`` is seconds from the scenario epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import YEAR

_TABLE = Path(__file__).resolve().parent / "data" / "msis_density.npz"


@dataclass(frozen=True)
class AtmosphereConfig:
    model: str = "msis_table"  # msis_table | exponential
    activity: str = "cycle"  # cycle | constant
    f107_constant: float = 130.0
    f107_min: float = 70.0  # smoothed cycle minimum (PICK)
    f107_max: float = 155.0  # smoothed cycle maximum (PICK; cycle 25 peak ~150-165)
    cycle_min_year: float = 2019.96  # cycle 25 minimum, Dec 2019
    cycle_max_year: float = 2024.8  # cycle 25 smoothed maximum, ~Oct 2024
    cycle_period_yr: float = 11.0


def decimal_year(epoch: datetime) -> float:
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    y0 = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
    y1 = datetime(epoch.year + 1, 1, 1, tzinfo=timezone.utc)
    return epoch.year + (epoch - y0).total_seconds() / (y1 - y0).total_seconds()


@lru_cache(maxsize=1)
def _load_table():
    d = np.load(_TABLE)
    return d["alt_km"], d["f107"], d["log_rho"]


class Atmosphere:
    def __init__(self, cfg: AtmosphereConfig = AtmosphereConfig(), epoch_year: float = 2026.97) -> None:
        if cfg.model not in ("msis_table", "exponential"):
            raise ValueError(f"unknown atmosphere model {cfg.model!r}")
        if cfg.activity not in ("cycle", "constant"):
            raise ValueError(f"unknown solar activity {cfg.activity!r}")
        self.cfg = cfg
        self.epoch_year = epoch_year
        if cfg.model == "msis_table":
            self._alt, self._f, self._lr = _load_table()

    def f107(self, t_s=0.0) -> np.ndarray:
        c = self.cfg
        if c.activity == "constant":
            return np.full(np.shape(t_s), c.f107_constant, dtype=float)
        year = self.epoch_year + np.asarray(t_s, float) / YEAR
        x = np.mod((year - c.cycle_min_year) / c.cycle_period_yr, 1.0)
        xr = (c.cycle_max_year - c.cycle_min_year) / c.cycle_period_yr
        g = np.where(x < xr, 0.5 * (1 - np.cos(np.pi * x / xr)),
                     0.5 * (1 + np.cos(np.pi * (x - xr) / (1 - xr))))
        return c.f107_min + (c.f107_max - c.f107_min) * g

    def density(self, alt_m, t_s=0.0) -> np.ndarray:
        """Mass density [kg/m^3]; ``t_s`` broadcasts against ``alt_m``."""
        if self.cfg.model == "exponential":
            from .propagator import exponential_density

            return exponential_density(alt_m)
        h = np.asarray(alt_m, float) / 1e3
        f = np.broadcast_to(self.f107(t_s), np.broadcast_shapes(np.shape(h), np.shape(t_s)))
        h = np.broadcast_to(h, f.shape)
        A, F, L = self._alt, self._f, self._lr
        i = np.clip(np.searchsorted(A, h) - 1, 0, len(A) - 2)
        wa = (h - A[i]) / (A[i + 1] - A[i])  # may exceed [0, 1]: log-linear extrapolation
        wa = np.maximum(wa, 0.0)  # clamp below 100 km
        fc = np.clip(f, F[0], F[-1])
        j = np.clip(np.searchsorted(F, fc) - 1, 0, len(F) - 2)
        wf = (fc - F[j]) / (F[j + 1] - F[j])
        lr = ((1 - wa) * (1 - wf) * L[i, j] + wa * (1 - wf) * L[i + 1, j]
              + (1 - wa) * wf * L[i, j + 1] + wa * wf * L[i + 1, j + 1])
        return np.exp(lr)

    def scale_height(self, alt_m, t_s=0.0) -> np.ndarray:
        """Local density scale height [m] = -dh / dln(rho)."""
        h = np.asarray(alt_m, float)
        dh = 5e3
        r1, r2 = self.density(h - dh, t_s), self.density(h + dh, t_s)
        return 2 * dh / np.maximum(np.log(r1 / r2), 1e-9)
