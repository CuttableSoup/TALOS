"""Platform survivability against orbital-debris impacts.

Each platform carries ~100-300 m^2 of arrays and radiator in the band it is clearing.
ORDEM 3.2 spacecraft mode gives the debris flux on the platform's own orbit by impact
direction and size (cross-sectional-area flux, #/m^2/yr). A two-sided flat surface of
area A with unit normal n is hit at A * sum_k F_k |n . d_k| over the igloo elements k
(direction d_k); a convex body of cross-section A_c at A_c * sum_k F_k.

Frame (ORDEM User Guide): x = ram, y = starboard (azimuth positive to the right of the
velocity vector), z = local zenith (elevation up from the local horizontal).
In a dawn-dusk SSO the sun-tracking arrays face the orbit normal (+-y).

Hazard: mission-ending events are (a) a hit >= ``lethal_size_m`` on the bus + beam
director cross-section and (b) a hit >= ``radiator_crit_size_m`` on the radiator's
vulnerable (tube / manifold) area. With hazard rate lam the expected service life is
(1 - exp(-lam T)) / lam for a deterministic life T, and the survival probability
exp(-lam T). Array hits are reported, not costed (cratering degrades cells locally).
Debris only: ORDEM has no meteoroids (NASA MEM would add a comparable mm-scale flux).

Tables (src/talos/data/platform_flux_<year>.npz, gitignored: derived from NASA SUA
software) are built by scripts/build_platform_flux.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ordem import DATA_DIR, MATERIALS

SIZE_CODES_ALL = ("10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60")
SIZES_M = np.array([10 ** (int(c) / 10) * 1e-6 for c in SIZE_CODES_ALL])  # cumulative >= size
NORMALS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
SURFACE_AXES = {"along_track": "x", "orbit_normal": "y", "radial": "z"}


@dataclass(frozen=True)
class SurvivalConfig:
    enabled: bool = True  # needs the platform flux table; off (columns NaN) without it
    year: int | None = None  # ORDEM year of the table; None -> the built table nearest mid-mission
    lethal_size_m: float = 0.01  # [sourced, weak] >= 1 cm destroys a spacecraft in LEO (SDC9 paper 330)
    bus_area_m2: float = 10.0  # PICK: bus + beam director cross-section (as cost.drag_area_fixed_m2)
    radiator_crit_size_m: float = 1e-3  # PICK: particle that punctures a radiator tube / manifold
    radiator_vulnerable_frac: float = 0.1  # PICK: tube + manifold share of radiator area
    radiator_normal: str = "radial"  # PICK: radial | along_track | orbit_normal (two-sided panel)
    array_normal: str = "orbit_normal"  # dawn-dusk SSO, sun-tracking


def table_path(year: int) -> Path:
    return DATA_DIR / f"platform_flux_{int(year)}.npz"


def nearest_table_year(target_year: float) -> int | None:
    years = [int(p.stem.rsplit("_", 1)[1]) for p in DATA_DIR.glob("platform_flux_*.npz")]
    return min(years, key=lambda y: abs(y - target_year)) if years else None


def read_sizeflux(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """SIZEFLUX_SC.OUT -> (size [m], cumulative cross-section flux [#/m^2/yr])."""
    text = Path(path).read_text().splitlines()
    i = next(k for k, line in enumerate(text) if line.split()[:2] == ["Size", "(m)"])
    d = np.loadtxt(text[i + 2:], ndmin=2)
    return d[:, 0], d[:, 1]


def reduce_directional(az_lo, az_hi, el_lo, el_hi, cols, flux) -> dict:
    """IGLOOFLUX columns -> {"sphere": (sizes,), "x"/"y"/"z": (sizes,)} summed over materials
    and speed: cross-section flux, and the |n . d|-weighted flux on a two-sided plate normal
    to each axis."""
    k = {c: j for j, c in enumerate(cols)}
    cum = np.stack([sum(flux[:, k[m + s]] for m in MATERIALS) for s in SIZE_CODES_ALL])  # (sizes, elements)
    az = np.deg2rad(0.5 * (az_lo + az_hi))
    el = np.deg2rad(0.5 * (el_lo + el_hi))
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    out = {"sphere": cum.sum(-1)}
    for name, n in NORMALS.items():
        out[name] = (cum * np.abs(d @ np.asarray(n))).sum(-1)
    return out


def read_igloo_directional(path: Path):
    text = Path(path).read_text().splitlines()
    hdr = text[10].split()
    if hdr[:7] != ["Element", "az_low", "az_high", "el_low", "el_high", "vel_low", "vel_high"]:
        raise ValueError(f"unexpected IGLOOFLUX header in {path}: {hdr[:7]}")
    cols = [c.removeprefix("Flux_") for c in hdr[7:]]
    dat = np.loadtxt(text[12:], ndmin=2)
    return dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], cols, dat[:, 7:7 + len(cols)]


@dataclass
class PlatformFluxTable:
    year: int
    alt_km: np.ndarray  # (n_alt,)
    sizes_m: np.ndarray  # (n_size,) cumulative thresholds
    flux: dict  # key -> (n_alt, n_size): "sphere", "x", "y", "z"
    fine_sizes_m: np.ndarray | None = None  # SIZEFLUX_SC.OUT grid (ORDEM's curve is steep at 1-3 mm)
    fine_sphere: np.ndarray | None = None  # (n_alt, n_fine) cross-section flux on that grid

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or table_path(self.year))
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = ({} if self.fine_sizes_m is None
                 else {"fine_sizes_m": self.fine_sizes_m, "fine_sphere": self.fine_sphere})
        np.savez(path, year=self.year, alt_km=self.alt_km, sizes_m=self.sizes_m, **extra,
                 **{f"flux_{k}": v for k, v in self.flux.items()})
        return path

    @classmethod
    def load(cls, path_or_year) -> "PlatformFluxTable":
        path = table_path(path_or_year) if isinstance(path_or_year, (int, np.integer)) else Path(path_or_year)
        with np.load(path) as f:
            return cls(int(f["year"]), f["alt_km"], f["sizes_m"],
                       {k.removeprefix("flux_"): f[k] for k in f.files if k.startswith("flux_")},
                       f["fine_sizes_m"] if "fine_sizes_m" in f.files else None,
                       f["fine_sphere"] if "fine_sphere" in f.files else None)

    def at(self, key: str, alt_km, size_m) -> np.ndarray:
        """Cumulative flux >= size_m [#/m^2/yr], log-linear in altitude. With the fine curve:
        fine cross-section flux x the directional fraction key / sphere (log-interpolated
        between the half-decade sizes, where it varies slowly); else log-log on the coarse grid."""
        ls = np.log10(self.sizes_m)
        x = np.log10(size_m)
        lf = np.log10(np.maximum(self.flux[key], 1e-30))
        if self.fine_sizes_m is None:
            per_alt = np.array([np.interp(x, ls, row) for row in lf])
        else:
            frac = lf - np.log10(np.maximum(self.flux["sphere"], 1e-30))
            fine = np.log10(np.maximum(self.fine_sphere, 1e-30))
            lsf = np.log10(self.fine_sizes_m)
            per_alt = np.array([np.interp(x, lsf, fr) + np.interp(x, ls, fc) for fr, fc in zip(fine, frac)])
        return 10.0 ** np.interp(np.asarray(alt_km, float), self.alt_km, per_alt)


def impact_rates(table: PlatformFluxTable, cfg: SurvivalConfig, alt_km, array_area_m2: float,
                 radiator_area_m2: float) -> dict:
    """Impacts per year per platform at ``alt_km`` (array, one entry per platform)."""
    alt = np.asarray(alt_km, float)
    ax_r = SURFACE_AXES[cfg.radiator_normal]
    ax_a = SURFACE_AXES[cfg.array_normal]
    bus = cfg.bus_area_m2 * table.at("sphere", alt, cfg.lethal_size_m)
    rad = cfg.radiator_vulnerable_frac * radiator_area_m2 * table.at(ax_r, alt, cfg.radiator_crit_size_m)
    return {
        "bus_lethal": bus,
        "radiator_puncture": rad,
        "array_1mm": array_area_m2 * table.at(ax_a, alt, 1e-3),
        "array_1cm": array_area_m2 * table.at(ax_a, alt, 1e-2),
        "hazard": bus + rad,
    }


def expected_life(hazard_per_yr, life_yr) -> tuple[np.ndarray, np.ndarray]:
    """(expected service life, survival probability) for a constant hazard over ``life_yr``."""
    lam = np.asarray(hazard_per_yr, float)
    T = float(life_yr)
    x = lam * T
    surv = np.exp(-x)
    el = np.where(x > 1e-12, -np.expm1(-x) / np.where(lam > 0, lam, 1.0), T)
    return el, surv
