"""Module 5 — engagement physics (Rogers & Lee, arXiv:2512.14682).

    fluence:  phi = eta2 * (E/pi) * [2D / (M^2 a lambda R)]^2
    dv/pulse: dv  = eta1 * Cm * phi / mu, along the beam line of sight

Pulses fire only when phi >= ablation threshold. Within a burst the target
trajectory is not updated by the accumulated dv (dv << orbital speed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .platform import LaserConfig


@dataclass(frozen=True)
class AblationParams:
    """Ablation / momentum-coupling model. Defaults from the research pass (2026-11):

    * Fire above ~4.5 kJ/m^2; Cm peaks near 8.5 kJ/m^2, then falls as phi^-1/4.
    * Peak Cm ~50 N/MW (sweep 25-100). Only ~100 ps Al data point found is
      <=30 N/MW, and it was IR, not UV.
    * Shape/tumble: coupling efficiency eta1 mean ~0.3; thrust angle off the
      beam axis centred near 45 deg.
    """

    cm: float = 50e-6  # N*s/J (= 50 N/MW), peak at phi_opt for "plasma"; flat for "constant"
    threshold_fluence: float = 4.5e3  # J/m^2, fire only above this
    phi_opt: float = 8.5e3  # J/m^2, Cm peak
    # "plasma": log ramp threshold -> phi_opt, then Cm = cm * (phi_opt/phi)^plasma_exponent
    cm_model: str = "plasma"  # "plasma" | "constant"
    plasma_exponent: float = 0.25
    # eta1 ~ Beta(mean*kappa, (1-mean)*kappa); kappa is a PLACEHOLDER (spread not sourced).
    eta1_mean: float = 0.3
    eta1_kappa: float = 10.0
    # Thrust deflection from beam axis ~ N(mean, sigma) clipped to [0, 90 deg], random azimuth.
    off_axis_mean_rad: float = np.deg2rad(45.0)
    off_axis_sigma_rad: float = np.deg2rad(15.0)  # PLACEHOLDER spread

    @property
    def eta1_nominal(self) -> float:
        return self.eta1_mean

    def sample_eta1(self, rng: np.random.Generator, n: int) -> np.ndarray:
        m = self.eta1_mean
        if m <= 0 or m >= 1 or not np.isfinite(self.eta1_kappa):
            return np.full(n, m)
        return rng.beta(m * self.eta1_kappa, (1 - m) * self.eta1_kappa, size=n)

    def sample_off_axis(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.clip(rng.normal(self.off_axis_mean_rad, self.off_axis_sigma_rad, n), 0.0, np.pi / 2)


def coupling_coefficient(phi, params: AblationParams) -> np.ndarray:
    """Momentum coupling Cm [N*s/J] at fluence ``phi`` [J/m^2]; 0 below threshold."""
    phi = np.asarray(phi, dtype=float)
    th = params.threshold_fluence
    above = phi >= th
    if params.cm_model == "constant":
        return np.where(above, params.cm, 0.0)
    if params.cm_model != "plasma":
        raise ValueError(f"unknown cm_model {params.cm_model!r}")
    opt = params.phi_opt
    if opt <= th:
        raise ValueError("phi_opt must exceed threshold_fluence")
    safe = np.where(above, phi, th)
    rise = params.cm * np.log(safe / th) / np.log(opt / th)
    fall = params.cm * (opt / safe) ** params.plasma_exponent
    return np.where(above, np.where(safe <= opt, rise, fall), 0.0)


def spot_diameter(range_m, laser: LaserConfig) -> np.ndarray:
    """Diffraction-limited spot diameter on target [m]: a M^2 lambda R / D."""
    return laser.beam_a * laser.m2 * laser.wavelength_m * np.asarray(range_m) / laser.aperture_d_m


def fluence(range_m, laser: LaserConfig) -> np.ndarray:
    """Peak-equivalent fluence on target [J/m^2]."""
    k = 2.0 * laser.aperture_d_m / (laser.m2 * laser.beam_a * laser.wavelength_m * np.asarray(range_m, float))
    return laser.eta2 * (laser.pulse_energy_j / np.pi) * k**2


def max_effective_range(laser: LaserConfig, threshold_fluence: float) -> float:
    """Range [m] at which fluence falls to ``threshold_fluence``."""
    return float(
        2.0 * laser.aperture_d_m / (laser.m2 * laser.beam_a * laser.wavelength_m)
        * np.sqrt(laser.eta2 * laser.pulse_energy_j / (np.pi * threshold_fluence))
    )


def dv_per_pulse(phi, areal_density, cm: float, eta1=1.0) -> np.ndarray:
    """Delta-v magnitude per pulse [m/s]."""
    return np.asarray(eta1) * cm * np.asarray(phi) / np.asarray(areal_density)


def pulse_times(t0: float, duration_s: float, prf_hz: float) -> np.ndarray:
    return t0 + np.arange(int(np.floor(duration_s * prf_hz))) / prf_hz


def _deflect(u: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate unit vectors ``u`` (P,3) by ``theta`` toward random perpendicular directions."""
    w = np.cross(u, rng.normal(size=u.shape))
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    return np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * w


@dataclass(frozen=True)
class BurstResult:
    dv_vec: np.ndarray  # (3,) summed ECI dv [m/s]
    dv_retrograde: float  # sum over pulses of -dv . v_hat [m/s]; >0 lowers orbit
    n_fired: int
    n_pulses: int
    optical_energy_j: float  # energy of pulses actually fired


def integrate_burst(
    r_sat: np.ndarray,
    r_deb: np.ndarray,
    v_deb: np.ndarray,
    laser: LaserConfig,
    areal_density: float,
    params: AblationParams = AblationParams(),
    jitter_rad: float = 0.0,
    rng: np.random.Generator | None = None,
) -> BurstResult:
    """Integrate one burst given states (P, 3) sampled at each pulse time.

    With ``rng=None`` the result is the deterministic expectation (eta1 mean,
    cos(mean off-axis angle) along the LOS, no jitter). With an rng, eta1, off-axis deflection and pointing jitter
    are sampled per pulse.
    """
    d = r_deb - r_sat
    R = np.linalg.norm(d, axis=-1)
    u = d / R[:, None]  # push direction: away from the laser
    phi = fluence(R, laser)
    P = len(R)

    if rng is None:
        # Expected value: random-azimuth deflection averages out, leaving cos(mean angle) along LOS.
        eta1 = np.full(P, params.eta1_nominal)
        dirs = u * np.cos(params.off_axis_mean_rad)
    else:
        if jitter_rad > 0:
            # Radial pointing offset (Rayleigh for 2-axis Gaussian jitter) on a
            # Gaussian-profile spot of 1/e^2 radius ~ spot_diameter/2.
            offset = R * jitter_rad * rng.rayleigh(size=P)
            w = spot_diameter(R, laser) / 2.0
            phi = phi * np.exp(-2.0 * (offset / w) ** 2)
        eta1 = params.sample_eta1(rng, P)
        dirs = _deflect(u, params.sample_off_axis(rng, P), rng)

    fire = phi >= params.threshold_fluence
    cm = coupling_coefficient(phi, params)
    dv_mag = np.where(fire, dv_per_pulse(phi, areal_density, cm, eta1), 0.0)
    dv_i = dv_mag[:, None] * dirs
    v_hat = v_deb / np.linalg.norm(v_deb, axis=-1, keepdims=True)
    n_fired = int(fire.sum())
    return BurstResult(
        dv_vec=dv_i.sum(axis=0),
        dv_retrograde=float(-np.sum(dv_i * v_hat)),
        n_fired=n_fired,
        n_pulses=P,
        optical_energy_j=n_fired * laser.pulse_energy_j,
    )
