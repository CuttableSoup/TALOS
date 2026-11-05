import numpy as np
import pytest

from talos.engagement import (
    AblationParams,
    dv_per_pulse,
    fluence,
    integrate_burst,
    max_effective_range,
    pulse_times,
    spot_diameter,
)
from talos.platform import LaserConfig

LASER = LaserConfig()


def test_spot_diameter_baseline():
    # a M^2 lambda R / D = 1.27 * 2 * 355e-9 * 300e3 / 1.5
    assert spot_diameter(300e3, LASER) == pytest.approx(0.18034, rel=1e-4)


def test_fluence_is_energy_over_spot_area():
    R = 250e3
    area = np.pi * (spot_diameter(R, LASER) / 2) ** 2
    assert fluence(R, LASER) == pytest.approx(LASER.eta2 * LASER.pulse_energy_j / area)


def test_fluence_scaling():
    base = fluence(200e3, LASER)
    assert fluence(400e3, LASER) == pytest.approx(base / 4)
    assert fluence(200e3, LaserConfig(aperture_d_m=3.0)) == pytest.approx(base * 4)
    assert fluence(200e3, LaserConfig(m2=4.0)) == pytest.approx(base / 4)
    assert fluence(200e3, LaserConfig(burst_power_w=50e3)) == pytest.approx(base * 2)


def test_max_range_inverts_fluence():
    th = AblationParams().threshold_fluence
    R = max_effective_range(LASER, th)
    assert fluence(R, LASER) == pytest.approx(th)


def test_baseline_range_order_of_magnitude():
    # Gate 1 proper (175-325 km, L'ADROIT) needs eta2 / pulse energy calibrated.
    # With eta2 = 1 this is an upper bound; just check it's the right magnitude.
    R = max_effective_range(LASER, AblationParams().threshold_fluence)
    assert 100e3 < R < 1000e3


def test_dv_per_pulse_units():
    # 30 uN*s/J * 1e4 J/m^2 / (1 kg/m^2) = 0.3 m/s
    assert dv_per_pulse(1e4, 1.0, 30e-6) == pytest.approx(0.3)
    assert dv_per_pulse(1e4, 10.0, 30e-6, eta1=0.5) == pytest.approx(0.015)


def head_on_geometry(R, n=33):
    """Debris dead ahead on +x, moving -x straight at the laser."""
    r_sat = np.zeros((n, 3))
    r_deb = np.tile([R, 0.0, 0.0], (n, 1))
    v_deb = np.tile([-7500.0, 0.0, 0.0], (n, 1))
    return r_sat, r_deb, v_deb


def test_head_on_burst_is_fully_retrograde():
    params = AblationParams(eta1_range=(1.0, 1.0))
    r_s, r_d, v_d = head_on_geometry(200e3)
    res = integrate_burst(r_s, r_d, v_d, LASER, areal_density=1.0, params=params)
    expected = 33 * params.cm * fluence(200e3, LASER) / 1.0
    assert res.n_fired == 33
    assert res.dv_retrograde == pytest.approx(expected)
    assert res.dv_vec[0] == pytest.approx(expected)
    assert res.optical_energy_j == pytest.approx(33 * LASER.pulse_energy_j)


def test_below_threshold_does_not_fire():
    R = 1.01 * max_effective_range(LASER, AblationParams().threshold_fluence)
    res = integrate_burst(*head_on_geometry(R), LASER, areal_density=1.0)
    assert res.n_fired == 0
    assert res.dv_retrograde == 0.0
    assert res.optical_energy_j == 0.0


def test_tail_chase_is_prograde():
    r_s, r_d, v_d = head_on_geometry(200e3)
    res = integrate_burst(r_s, r_d, -v_d, LASER, areal_density=1.0)
    assert res.dv_retrograde < 0


def test_stochastic_burst_bounded_and_reproducible():
    params = AblationParams()
    geom = head_on_geometry(200e3, n=330)
    a = integrate_burst(*geom, LASER, 1.0, params, jitter_rad=0.3e-6, rng=np.random.default_rng(7))
    b = integrate_burst(*geom, LASER, 1.0, params, jitter_rad=0.3e-6, rng=np.random.default_rng(7))
    assert a.dv_retrograde == b.dv_retrograde
    upper = 330 * params.eta1_range[1] * params.cm * fluence(200e3, LASER)
    assert 0 < a.dv_retrograde < upper
    assert np.linalg.norm(a.dv_vec) <= upper


def test_pulse_times():
    t = pulse_times(100.0, 10.0, 33.0)
    assert len(t) == 330
    assert t[1] - t[0] == pytest.approx(1 / 33)
