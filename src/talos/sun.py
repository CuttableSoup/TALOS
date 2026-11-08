"""Sun ephemeris, Earth shadow and per-platform illumination timelines.

* Sun direction: Astronomical Almanac low-precision formula (~0.01 deg,
  1950-2050), in the mean-equator inertial frame the propagator uses
  (treated as J2000; precession over a sim is negligible here).
* Earth shadow on a platform: conical umbra/penumbra (Montenbruck & Gill
  3.4.2), returning the illuminated fraction of the solar disc.
* Earth shadow on a debris target (screening): cylindrical shadow, cheaper and
  within seconds of the conical result at LEO.

Sim time t [s] is measured from the scenario epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .constants import R_EARTH
from .propagator import states_at

AU = 1.495978707e11
R_SUN = 6.957e8
J2000_JD = 2451545.0


@dataclass(frozen=True)
class SunConfig:
    enabled: bool = True  # False: always sunlit, no sun gates (legacy behaviour)
    solar_constant_w_m2: float = 1361.0  # at 1 AU
    albedo: float = 0.30  # Earth Bond albedo (range 0.20-0.35)
    # "two_axis": array normal tracks the sun; "orbit_normal": fixed along the orbit
    # normal (natural for dawn-dusk), output ~ |cos(sun, normal)|
    array_pointing: str = "two_axis"
    fire_in_eclipse: bool = False  # True: laser may fire from battery in eclipse
    lit_threshold: float = 0.5  # platform counts as sunlit when this fraction of the disc is visible
    timeline_step_s: float = 10.0  # resolution of the illumination timeline


def julian_date(epoch: datetime) -> float:
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    unix = epoch.astimezone(timezone.utc).timestamp()
    return unix / 86400.0 + 2440587.5


def sun_vector(jd) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric sun unit vector (..., 3) and distance [m] at Julian date(s) ``jd``."""
    n = np.asarray(jd, dtype=float) - J2000_JD
    L = np.deg2rad(280.460 + 0.9856474 * n)
    g = np.deg2rad(357.528 + 0.9856003 * n)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 4e-7 * n)
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    u = np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)
    return u, r_au * AU


def lit_fraction(r_obj, sun_hat, sun_dist) -> np.ndarray:
    """Visible fraction of the solar disc from position(s) ``r_obj`` (..., 3): conical shadow."""
    r_obj = np.asarray(r_obj, float)
    r_sun = sun_hat * np.asarray(sun_dist)[..., None]
    d_sun = r_sun - r_obj
    ds = np.linalg.norm(d_sun, axis=-1)
    ro = np.linalg.norm(r_obj, axis=-1)
    a = np.arcsin(np.clip(R_SUN / ds, -1, 1))  # apparent sun radius
    b = np.arcsin(np.clip(R_EARTH / ro, -1, 1))  # apparent earth radius
    cosc = np.sum(-r_obj * d_sun, -1) / (ro * ds)
    c = np.arccos(np.clip(cosc, -1, 1))  # separation of the discs
    frac = np.ones_like(c)
    umbra = c <= b - a  # sun fully hidden (a < b always at LEO)
    part = (c < a + b) & ~umbra
    if np.any(part):
        ap, bp, cp = a[part], b[part], c[part]
        x = (cp ** 2 + ap ** 2 - bp ** 2) / (2 * cp)
        y = np.sqrt(np.maximum(ap ** 2 - x ** 2, 0.0))
        area = (ap ** 2 * np.arccos(np.clip(x / ap, -1, 1)) + bp ** 2 * np.arccos(np.clip((cp - x) / bp, -1, 1))
                - cp * y)
        frac[part] = 1.0 - area / (np.pi * ap ** 2)
    frac[umbra] = 0.0
    return frac


@dataclass(frozen=True)
class Timeline:
    """Piecewise-constant illumination of one platform: segment k covers [k*step, (k+1)*step)."""

    step_s: float
    lit: np.ndarray  # bool: platform sunlit (fraction >= threshold)
    array_factor: np.ndarray  # array output / nominal (1 AU, normal incidence, fully lit)
    albedo_w_m2: np.ndarray  # albedo flux incident on one radiator side (before absorptivity)

    def index(self, t: float) -> int:
        return int(min(max(t // self.step_s, 0), len(self.lit) - 1))

    def seg_end(self, t: float) -> float:
        k = self.index(t)
        return np.inf if k == len(self.lit) - 1 else (k + 1) * self.step_s

    @property
    def eclipse_fraction(self) -> float:
        return float(1.0 - self.lit.mean())


class SunEnvironment:
    def __init__(self, cfg: SunConfig, epoch: datetime) -> None:
        self.cfg = cfg
        self.epoch = epoch
        self.jd0 = julian_date(epoch)

    def sun(self, t) -> tuple[np.ndarray, np.ndarray]:
        return sun_vector(self.jd0 + np.asarray(t, float) / 86400.0)

    def sun_hat(self, t) -> np.ndarray:
        return np.ascontiguousarray(self.sun(t)[0])

    def raan_for_ltan(self, ltan_hours: float) -> float:
        """RAAN giving local time of ascending node ``ltan_hours`` at the epoch."""
        u, _ = self.sun(0.0)
        ra_sun = np.arctan2(u[1], u[0])
        return float((ra_sun + np.deg2rad(15.0 * (ltan_hours - 12.0))) % (2 * np.pi))

    def timeline(self, platform, duration_s: float, earth_view: float = 0.22) -> Timeline:
        step = self.cfg.timeline_step_s
        n = int(np.ceil(duration_s / step)) + 2
        tm = (np.arange(n) + 0.5) * step
        r, v = states_at(platform.orbit, tm)
        u, dist = self.sun(tm)
        frac = lit_fraction(r, u, dist)
        flux_scale = (AU / dist) ** 2
        if self.cfg.array_pointing == "two_axis":
            cosang = np.ones(n)
        elif self.cfg.array_pointing == "orbit_normal":
            h = np.cross(r, v)
            h /= np.linalg.norm(h, axis=-1, keepdims=True)
            cosang = np.abs(np.sum(h * u, -1))
        else:
            raise ValueError(f"unknown array_pointing {self.cfg.array_pointing!r}")
        cos_zen = np.sum(r * u, -1) / np.linalg.norm(r, axis=-1)  # sun elevation at the sub-satellite point
        albedo = (self.cfg.albedo * self.cfg.solar_constant_w_m2 * flux_scale * earth_view
                  * np.clip(cos_zen, 0.0, None))
        return Timeline(step, frac >= self.cfg.lit_threshold, frac * flux_scale * cosang, albedo)


def target_lit(r, sun_hat) -> np.ndarray:
    """Cylindrical-shadow test for positions ``r`` (..., 3) with sun direction (..., 3)."""
    d = np.sum(r * sun_hat, -1)
    perp2 = np.sum(r * r, -1) - d * d
    return (d >= 0) | (perp2 >= R_EARTH ** 2)
