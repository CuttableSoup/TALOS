"""Validation gate 2: single-pass delta-v against Pieters & Noomen, "Space-based laser
ablation for space debris removal", ESA SDC8 (2021), paper 43.

Their system (Table 1 / 2): 20 kW, 600 J at 33.33 Hz, 100 ps, D 1.5 m, M^2 2, a 1.27,
T_eff 0.9, Cm 128 N/MW, platform in an 800 km circular orbit at 98 deg. Thrust is
T_eff * A * Cm * fluence * f along the beam: no eta1, no thrust angle, no ablation
threshold in the force, and lasing through the whole approach ("ablation is achieved at a
distance of about 500 km") until the tracking rate exceeds 2 deg/s or the object has passed.

What is and is not reproducible from the paper:

* Table 4 (coplanar, counter-rotating, debris 50 / 100 / 200 km above and below): fully
  specified. TALOS's fluence and impulse equations reproduce all six within 15 %.
* Head-on at the same altitude (213 m/s on the 10 cm object, 818 m/s on the 1 cm one):
  NOT fully specified. An exact head-on pass has zero line-of-sight rate, so the stated
  63 km cut-off implies a miss distance the paper does not give, and the result goes as
  1 / (closest lasing range). TALOS gives 303 m/s when lasing stops at 63 km; the paper's
  value corresponds to stopping near 85 km. The test only brackets it.
* Table 1 lists the wavelength as 335 nm. 355 nm (L'ADROIT, which the paper takes its
  parameters from) fits Table 4 with a mean TALOS / paper ratio of 1.01; 335 nm gives 1.14.
* Lifetimes (their exponential atmosphere) and the non-coplanar cases (azimuth geometry
  not defined well enough to rebuild) are not compared.
"""

from dataclasses import replace

import numpy as np
import pytest

from talos.constants import MU_EARTH, R_EARTH
from talos.elements import KeplerElements, elements_to_rv
from talos.engagement import AblationParams, integrate_burst
from talos.platform import LaserConfig

LASER = LaserConfig(wavelength_m=355e-9, pulse_width_s=100e-12, aperture_d_m=1.5, m2=2.0, beam_a=1.27,
                    prf_hz=33.33, burst_power_w=600.0 * 33.33, eta2=0.9)
# Cm 128 N/MW at every fluence, the whole impulse along the beam
PARAMS = AblationParams(cm=128e-6, threshold_fluence=0.0, cm_model="constant", eta1_mean=1.0,
                        eta1_kappa=np.inf, off_axis_mean_rad=0.0, off_axis_sigma_rad=0.0)
H_SAT = 800e3
INC = np.deg2rad(98.0)
SLEW_MAX = np.deg2rad(2.0)
LASE_FROM_M = 500e3
AMR_10CM, AMR_1CM = 0.04, 0.16  # m^2/kg, Table 3

# Table 4: debris altitude [km] -> delta-v [m/s], 10 cm object
TABLE_4 = {600: 34.0, 700: 65.0, 750: 92.3, 850: 93.5, 900: 61.1, 1000: 33.2}


def _circular(alt_m, inc, raan, m0, t):
    a = R_EARTH + alt_m
    n = np.sqrt(MU_EARTH / a**3)
    return elements_to_rv(KeplerElements(np.full(t.shape, a), 0.0, inc, raan, 0.0, m0 + n * t))


def one_pass(debris_alt_m, amr, laser=LASER, stop_range_m=None):
    """Delta-v [m/s] against the velocity over one counter-rotating coplanar pass that meets
    at t = 0, with the range [m] where lasing starts and stops."""
    t = np.arange(-400.0, 0.0, 1.0 / laser.prf_hz)
    r_s, v_s = _circular(H_SAT, INC, 0.0, 0.0, t)
    r_d, v_d = _circular(debris_alt_m, np.pi - INC, np.pi, np.pi, t)  # same plane, opposite sense
    d, v_rel = r_d - r_s, v_d - v_s
    rng = np.linalg.norm(d, axis=1)
    rate = np.linalg.norm(np.cross(d, v_rel), axis=1) / rng**2
    ok = (rng <= LASE_FROM_M) & (np.einsum("ij,ij->i", d, v_rel) < 0) & (rate <= SLEW_MAX)
    if stop_range_m is not None:
        ok &= rng >= stop_range_m
    k = np.flatnonzero(ok)
    gaps = np.flatnonzero(np.diff(k) > 1)
    if len(gaps):  # the first uninterrupted stretch: lasing does not resume after the cut-off
        k = k[: gaps[0] + 1]
    res = integrate_burst(r_s[k], r_d[k], v_d[k], laser, 1.0 / amr, PARAMS)
    return res.dv_retrograde, rng[k[0]], rng[k[-1]]


@pytest.mark.parametrize("alt_km, dv_paper", sorted(TABLE_4.items()))
def test_table4_coplanar_altitude_offset(alt_km, dv_paper):
    dv, _, _ = one_pass(alt_km * 1e3, AMR_10CM)
    assert dv == pytest.approx(dv_paper, rel=0.15)


def test_table4_mean_ratio_and_wavelength():
    def mean_ratio(laser):
        return np.mean([one_pass(h * 1e3, AMR_10CM, laser)[0] / dv for h, dv in TABLE_4.items()])

    assert mean_ratio(LASER) == pytest.approx(1.0, abs=0.05)
    # the 335 nm printed in Table 1 overshoots: 355 nm fits the reported results
    assert mean_ratio(replace(LASER, wavelength_m=335e-9)) > 1.08


def test_slew_cutoff_ranges():
    # "For the objects at +-100 km this is at a relative distance of about 200 km, but the
    # objects at +-200 km already become untrackable at ... about 300 km"
    for alt_km, cutoff_km in ((700, 200), (900, 200), (600, 300), (1000, 300)):
        _, start, stop = one_pass(alt_km * 1e3, AMR_10CM)
        assert start == pytest.approx(LASE_FROM_M, rel=0.01)
        assert stop / 1e3 == pytest.approx(cutoff_km, abs=12)


def test_dv_scales_with_area_to_mass():
    # the paper: 818 m/s on the 1 cm object against 213 m/s on the 10 cm one (x3.84, AMR x4)
    dv10, _, _ = one_pass(H_SAT, AMR_10CM, stop_range_m=63e3)
    dv1, _, _ = one_pass(H_SAT, AMR_1CM, stop_range_m=63e3)
    assert dv1 / dv10 == pytest.approx(4.0, rel=1e-6)
    assert 818.0 / 213.0 == pytest.approx(dv1 / dv10, rel=0.05)


def test_head_on_is_bracketed_not_reproduced():
    # goes as 1 / (closest lasing range); the paper's 213 m/s sits between stopping at the
    # stated 63 km and stopping at 100 km (equivalent stop ~85 km)
    at_63, _, _ = one_pass(H_SAT, AMR_10CM, stop_range_m=63e3)
    at_100, _, _ = one_pass(H_SAT, AMR_10CM, stop_range_m=100e3)
    assert at_100 < 213.0 < at_63
    assert at_63 == pytest.approx(303.0, rel=0.02)
