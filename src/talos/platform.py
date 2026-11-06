"""Module 3 — platform / ATP configuration dataclasses and baseline point design.

Defaults follow the README's L'ADROIT-class baseline. Values marked PLACEHOLDER
have not been calibrated against a reference and should be before sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import FT2_TO_M2, R_EARTH
from .elements import KeplerElements
from .propagator import sso_inclination


@dataclass(frozen=True)
class LaserConfig:
    wavelength_m: float = 355e-9
    pulse_width_s: float = 100e-12
    aperture_d_m: float = 1.5
    m2: float = 2.0
    beam_a: float = 1.27  # diffraction constant "a" in the fluence equation
    prf_hz: float = 33.0
    burst_power_w: float = 25e3  # optical power while firing
    # PLACEHOLDER: optical-train transmission x Strehl. 0.35 is back-solved so the
    # baseline threshold range is ~325 km (L'ADROIT upper bound) -- replace with
    # a sourced value (research item 1); gate 1 is not independently checked yet.
    eta2: float = 0.35

    @property
    def pulse_energy_j(self) -> float:
        return self.burst_power_w / self.prf_hz


@dataclass(frozen=True)
class ATPConfig:
    """Acquisition, tracking & pointing limits."""

    detection_range_m: float = 800e3
    engage_range_min_m: float = 100e3
    engage_range_max_m: float = 500e3
    slew_rate_limit_rad_s: float = np.deg2rad(2.0)
    # Field of regard about the ram direction; pi = unrestricted (PLACEHOLDER).
    field_of_regard_half_angle_rad: float = np.pi
    earth_clearance_alt_m: float = 100e3  # LOS must stay above this altitude
    jitter_rad: float = 0.3e-6  # 1-sigma per-axis pointing jitter
    # PLACEHOLDER: retarget + settle + acquisition before each burst. Conservative:
    # counted after target selection, not overlapped with recovery.
    settle_s: float = 5.0


@dataclass(frozen=True)
class ResourceConfig:
    """Power/thermal inputs for module 8. Duty cycle is an output, not an input."""

    array_area_m2: float = 1600 * FT2_TO_M2
    array_specific_power_w_m2: float = 250.0  # EOL
    wallplug_eff: float = 0.15
    radiator_area_m2: float = 20.0  # PLACEHOLDER
    radiator_net_flux_w_m2: float = 600.0
    # PLACEHOLDER: storable waste heat (PCM / thermal mass). ~1 full burst of net heat.
    heat_buffer_j: float = 1.5e6
    battery_capacity_j: float = 2.0 * 3.6e6  # PLACEHOLDER (2 kWh)
    bus_power_w: float = 1.5e3  # PLACEHOLDER
    burst_s: float = 10.0  # max burst; platform waits until a full burst fits
    min_burst_s: float = 5.0  # PLACEHOLDER: shortest burst worth starting


@dataclass(frozen=True)
class Platform:
    name: str
    orbit: KeplerElements  # scalar elements at sim epoch
    laser: LaserConfig = field(default_factory=LaserConfig)
    atp: ATPConfig = field(default_factory=ATPConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)


def baseline_platform(
    alt_km: float = 800.0,
    raan_rad: float = 0.0,
    mean_anomaly_rad: float = 0.0,
    name: str | None = None,
    **overrides,
) -> Platform:
    """Circular SSO platform. ``raan_rad`` should be set from the sun's RA at
    epoch (+/- 90 deg) to make it dawn-dusk; that is left to the caller for now."""
    a = R_EARTH + alt_km * 1e3
    orbit = KeplerElements(a=a, e=0.0, i=sso_inclination(a), raan=raan_rad, argp=0.0, M=mean_anomaly_rad)
    return Platform(name=name or f"sat-{alt_km:.0f}km", orbit=orbit, **overrides)


def baseline_constellation(
    n_sats: int,
    altitudes_km: tuple[float, ...] = (700.0, 800.0, 900.0),
    raan_rad: float = 0.0,
    **overrides,
) -> list[Platform]:
    """Round-robin sats over altitude shells, phased evenly in mean anomaly."""
    return [
        baseline_platform(
            alt_km=altitudes_km[k % len(altitudes_km)],
            raan_rad=raan_rad,
            mean_anomaly_rad=2 * np.pi * k / n_sats,
            name=f"sat{k}",
            **overrides,
        )
        for k in range(n_sats)
    ]
