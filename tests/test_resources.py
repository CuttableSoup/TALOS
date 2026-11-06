import numpy as np
import pytest

from talos.platform import LaserConfig, ResourceConfig
from talos.resources import ResourceState


def state(**kw):
    return ResourceState(ResourceConfig(**kw), LaserConfig())


def test_baseline_rates():
    s = state()
    assert s.p_elec == pytest.approx(25e3 / 0.15)
    assert s.p_heat == pytest.approx(25e3 / 0.15 - 25e3)
    assert s.p_rad == pytest.approx(12e3)
    # thermal-limited: 12 kW / 141.7 kW
    assert s.steady_state_max_duty() == pytest.approx(12e3 / s.p_heat)


def test_fire_then_recover_cycle():
    s = state()
    f = s.cfg.burst_s
    assert s.max_fire_s() >= f
    s.fire(f)
    assert s.heat_j == pytest.approx((s.p_heat - s.p_rad) * f)
    w = s.wait_for(f)
    assert w > 0
    s.advance(s.t + w)
    assert s.max_fire_s() == pytest.approx(f, rel=1e-6)
    # thermal-limited cycle period = f + w, consistent with steady-state duty
    s.fire(f)
    w2 = s.wait_for(f)
    assert f / (f + w2) == pytest.approx(s.steady_state_max_duty(), rel=1e-6)


def test_overfire_raises():
    s = state()
    with pytest.raises(ValueError):
        s.fire(s.max_fire_s() + 1.0)


def test_power_limited_when_array_small():
    s = state(array_area_m2=10.0, battery_capacity_j=2e6, radiator_area_m2=1e3)
    assert s.steady_state_max_duty() == pytest.approx(s.p_charge / s.p_elec)
    f = s.max_fire_s()
    assert f == pytest.approx(2e6 / (s.p_elec - s.p_charge))


def test_burst_that_never_fits():
    s = state(heat_buffer_j=1e5)
    assert np.isinf(s.wait_for(10.0))


def test_advance_backwards_raises():
    s = state()
    s.advance(10.0)
    with pytest.raises(ValueError):
        s.advance(5.0)
