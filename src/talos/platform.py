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
    # Fluence efficiency. ~0.40 backed out from L'ADROIT's published spot size and
    # fluence (research pass 2026-11); sweep 0.3-0.67.
    eta2: float = 0.40

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
    """Power/thermal inputs for module 8. Duty cycle is an output, not an input.

    Sources: research pass "Thermal and Power Parameters ..." (2026-11), see README.
    Thermal architecture: laser head on an isothermal pumped two-phase loop
    (~20 C) dumping into a PCM store; the radiator runs ``radiator_dT_k`` below
    the store. One lumped node: sensible ``heat_capacity_j_k`` + latent plateau at
    ``pcm_melt_k``; heaters/louvers hold it >= ``t_min_k``; firing must keep it
    <= ``t_max_k``.
    """

    array_area_m2: float = 1600 * FT2_TO_M2
    array_specific_power_w_m2: float = 250.0  # EOL; range 200-300, degradation unsourced
    wallplug_eff: float = 0.10  # electrical -> 355 nm; chain 0.9*0.55*0.45*0.40, range 0.04-0.17
    bus_power_w: float = 1.5e3  # PLACEHOLDER; dissipated into the same thermal node
    # --- radiator: q_net = fin * sides * [eps*sigma*T^4 - eps*F_E*IR_E - other] ---
    radiator_area_m2: float = 20.0  # PLACEHOLDER (sweep it; ~40 m2 needed for 10 s / 2 min)
    radiator_sides: int = 2
    radiator_emissivity: float = 0.90  # Z-93 / AZ-93
    radiator_earth_view: float = 0.22  # per side, panel plane containing nadir, 700-900 km
    earth_ir_w_m2: float = 237.0
    radiator_other_load_w_m2: float = 5.0  # albedo/sun residual per side (dawn-dusk, edge-on)
    fin_efficiency: float = 0.85
    radiator_dT_k: float = 8.0  # store -> radiator, pumped two-phase (loop heat pipe: ~20 K)
    # --- thermal store ---
    pcm_melt_k: float = 288.15  # 15 C, below a ~20 C diode loop (salt-hydrate class; unsourced)
    pcm_mass_kg: float = 25.0
    pcm_specific_j_kg: float = 120e3  # system level incl. fins/container (range 50-170)
    heat_capacity_j_k: float = 0.5e6  # GUESS: laser head + loop sensible capacity (0.2-1 MJ/K)
    t_min_k: float = 283.15  # heater floor (melt - 5 K)
    t_max_k: float = 290.15  # max store temperature while firing (melt + 2 K)
    # --- battery ---
    battery_capacity_j: float = 2.0 * 3.6e6  # PLACEHOLDER (2 kWh); cycle life NOT modelled
    battery_rt_eff: float = 0.85  # discharge losses -> heat (range 0.80-0.90)
    burst_s: float = 10.0  # max burst; platform waits until a full burst fits
    min_burst_s: float = 5.0  # PLACEHOLDER: shortest burst worth starting

    @property
    def pcm_latent_j(self) -> float:
        return self.pcm_mass_kg * self.pcm_specific_j_kg


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
