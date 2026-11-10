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
    # What eta1 measures. "along_beam" (sourced, default): eta1 is the efficiency of the
    # *useful* impulse along the beam, with improper thrust direction, shape and tumble
    # already folded in (Phipps et al. 2012 Eq. 6, eta_c = 0.3 "after Liedahl" for "the
    # combined effects of improper thrust direction, target shape, target tumbling";
    # Rogers & Lee 2025 Eq. 2-4 push along the LOS with no separate angle). The impulse
    # magnitude is then eta1 / cos(off_axis_mean), so its mean along-beam part is eta1 and the
    # off-axis draw only adds the sideways scatter. "magnitude" (pre-2026-11-09 runs): eta1
    # is the impulse magnitude and the deflection removes another cos(45 deg) ~ 29 %.
    eta1_basis: str = "along_beam"
    # Target attitude: "per_pulse" draws eta1 and the thrust direction for every pulse;
    # "per_burst" holds them for the whole burst (slow tumble: same face lit throughout).
    attitude_model: str = "per_pulse"
    # Cm relative to nominal by ORDEM material class (NaK, low / medium / high density,
    # intact); PICKS, all 1 = no material dependence. 0 = immune to this laser.
    cm_scale_nk: float = 1.0
    cm_scale_ld: float = 1.0
    cm_scale_md: float = 1.0
    cm_scale_hd: float = 1.0
    cm_scale_in: float = 1.0
    # Fragmentation by ablation (PICKS; 0 = off): probability per burst that the target
    # sheds fragments; mean number >= 1 cm (Poisson), size fraction of the parent, dv spread.
    fragment_prob_per_burst: float = 0.0
    fragment_count_mean: float = 2.0
    fragment_size_frac: tuple[float, float] = (0.3, 0.7)
    fragment_dv_sigma_m_s: float = 5.0

    def cm_scale_for(self, material) -> np.ndarray:
        """Per-object Cm scale from ORDEM class indices (-1 / unknown -> medium density)."""
        table = np.array([self.cm_scale_nk, self.cm_scale_ld, self.cm_scale_md, self.cm_scale_hd,
                          self.cm_scale_in])
        m = np.asarray(material)
        return np.where(m >= 0, table[np.clip(m, 0, 4)], self.cm_scale_md)

    @property
    def eta1_scale(self) -> float:
        """Impulse magnitude per unit eta1 (see ``eta1_basis``)."""
        if self.eta1_basis == "magnitude":
            return 1.0
        if self.eta1_basis == "along_beam":
            c = float(np.cos(self.off_axis_mean_rad))
            if c < 1e-3:
                raise ValueError("eta1_basis 'along_beam' needs off_axis_mean below 90 deg")
            return 1.0 / c
        raise ValueError(f"eta1_basis {self.eta1_basis!r}: expected along_beam or magnitude")

    @property
    def eta1_nominal(self) -> float:
        """Mean impulse magnitude factor (eta1_mean x eta1_scale)."""
        return self.eta1_mean * self.eta1_scale

    def sample_eta1(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Per-pulse impulse magnitude factors: Beta(eta1_mean, kappa) x eta1_scale."""
        m, s = self.eta1_mean, self.eta1_scale
        if m <= 0 or m >= 1 or not np.isfinite(self.eta1_kappa):
            return np.full(n, m * s)
        return s * rng.beta(m * self.eta1_kappa, (1 - m) * self.eta1_kappa, size=n)

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
    pointing_bias: np.ndarray | None = None,
) -> BurstResult:
    """Integrate one burst given states (P, 3) sampled at each pulse time.

    With ``rng=None`` the result is the deterministic expectation (eta1 mean,
    cos(mean off-axis angle) along the LOS, no jitter). With an rng, eta1, off-axis deflection and pointing jitter
    are sampled per pulse (eta1 and deflection per burst with ``params.attitude_model = "per_burst"``).
    ``pointing_bias``: (2,) constant pointing offset [rad] for the burst (orbit-knowledge /
    centroiding error), added to the per-pulse jitter.
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
        if pointing_bias is not None and np.any(pointing_bias != 0):
            # bias + 2-axis Gaussian jitter, Gaussian spot of 1/e^2 radius ~ spot_diameter/2
            ang = np.asarray(pointing_bias, float)[None, :] + jitter_rad * rng.standard_normal((P, 2))
            offset = R * np.hypot(ang[:, 0], ang[:, 1])
            w = spot_diameter(R, laser) / 2.0
            phi = phi * np.exp(-2.0 * (offset / w) ** 2)
        elif jitter_rad > 0:
            # Radial pointing offset (Rayleigh for 2-axis Gaussian jitter) on a
            # Gaussian-profile spot of 1/e^2 radius ~ spot_diameter/2.
            offset = R * jitter_rad * rng.rayleigh(size=P)
            w = spot_diameter(R, laser) / 2.0
            phi = phi * np.exp(-2.0 * (offset / w) ** 2)
        if params.attitude_model == "per_burst":
            eta1 = np.full(P, params.sample_eta1(rng, 1)[0])
            theta = np.full(P, params.sample_off_axis(rng, 1)[0])
            w = np.cross(u, rng.normal(size=3)[None, :])  # one deflection azimuth for the burst
            w /= np.linalg.norm(w, axis=-1, keepdims=True)
            dirs = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * w
        elif params.attitude_model == "per_pulse":
            eta1 = params.sample_eta1(rng, P)
            dirs = _deflect(u, params.sample_off_axis(rng, P), rng)
        else:
            raise ValueError(f"attitude_model {params.attitude_model!r}: expected per_pulse or per_burst")

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


@dataclass(frozen=True)
class BurstPrediction:
    """Deterministic (expected-value) predictions for n candidate bursts."""

    dv_vec: np.ndarray  # (n, 3)
    dv_retrograde: np.ndarray  # (n,)
    n_fired: np.ndarray  # (n,)
    optical_energy_j: np.ndarray  # (n,)


def predict_bursts(
    r_sat: np.ndarray,
    r_deb: np.ndarray,
    v_deb: np.ndarray,
    active: np.ndarray,
    laser: LaserConfig,
    areal_density: np.ndarray,
    params: AblationParams = AblationParams(),
) -> BurstPrediction:
    """Batched deterministic version of ``integrate_burst``.

    r_sat: (P, 3); r_deb, v_deb: (P, n, 3); active: (P, n) pulses inside each
    candidate's burst; areal_density: (n,). Matches ``integrate_burst(rng=None)``.
    """
    d = r_deb - r_sat[:, None, :]
    R = np.linalg.norm(d, axis=-1)
    phi = fluence(R, laser)
    fire = active & (phi >= params.threshold_fluence)
    cm = coupling_coefficient(phi, params)
    mag = np.where(fire, params.eta1_nominal * cm * phi / np.asarray(areal_density)[None, :], 0.0)
    mag = mag * np.cos(params.off_axis_mean_rad) / R  # fold LOS normalisation into the magnitude
    dv_i = mag[..., None] * d
    v_hat = v_deb / np.linalg.norm(v_deb, axis=-1, keepdims=True)
    n_fired = fire.sum(axis=0)
    return BurstPrediction(
        dv_vec=dv_i.sum(axis=0),
        dv_retrograde=-np.sum(dv_i * v_hat, axis=(0, 2)),
        n_fired=n_fired,
        optical_energy_j=n_fired * laser.pulse_energy_j,
    )
