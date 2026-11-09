import numpy as np
import pytest

from talos.platform import LaserConfig, ResourceConfig
from talos.resources import SIGMA, ResourceState


def state(**kw):
    return ResourceState(ResourceConfig(**kw), LaserConfig())


def test_baseline_rates():
    s = state()
    assert s.p_elec == pytest.approx(25e3 / 0.10)
    assert s.p_heat == pytest.approx(225e3)
    # research table: two-sided net flux ~445 W/m^2 at a 280 K radiator
    assert s.q_net(280.0) == pytest.approx(445.0, rel=0.02)
    assert s.q_net(300.0) == pytest.approx(615.0, rel=0.03)


def test_radiator_follows_t4():
    s = state(earth_ir_w_m2=0.0, radiator_other_load_w_m2=0.0, fin_efficiency=1.0, radiator_sides=1,
              radiator_emissivity=1.0)
    assert s.q_net(300.0) == pytest.approx(SIGMA * 300.0 ** 4)
    assert s.p_rad_at(310.0) > s.p_rad_at(300.0)


def test_enthalpy_temperature_map():
    s = state()
    c = s.cfg
    assert s.temp_of(0.0) == pytest.approx(c.t_min_k)
    assert s.temp_of(s._H1 + 0.5 * c.pcm_latent_j) == pytest.approx(c.pcm_melt_k)
    assert s.temp_of(s.h_max) == pytest.approx(c.t_max_k)


def test_fire_then_recover_cycle_matches_steady_state_duty():
    s = state()
    f = s.cfg.burst_s
    assert s.max_fire_s() >= f
    for _ in range(40):
        s.fire(f)
        w = s.wait_for(f)
        assert np.isfinite(w) and w >= 0  # 0 while the store still has spare capacity
        s.advance(s.t + w)
        assert s.max_fire_s() >= f - 1e-6
    # cycling on the melt plateau: duty = (rad - idle) / (fire_in - idle); bursts poke into
    # the sensible band above melt where the radiator runs hotter, so slightly above
    assert s.temperature_k == pytest.approx(s.cfg.pcm_melt_k)
    assert f / (f + w) == pytest.approx(s.steady_state_max_duty(), rel=0.02)


def test_bigger_radiator_raises_duty():
    d = [state(radiator_area_m2=a).steady_state_max_duty() for a in (20.0, 40.0, 60.0)]
    assert d[0] < d[1] < d[2]
    assert d[0] == pytest.approx(0.029, abs=0.003)  # research sanity check: ~3.6 % before battery losses


def test_idle_cools_to_heater_floor():
    s = state()
    s.fire(10.0)
    s.advance(s.t + 1e5)
    assert s.heat_j == 0.0 and s.temperature_k == pytest.approx(s.cfg.t_min_k)


def test_overfire_raises():
    s = state()
    with pytest.raises(ValueError):
        s.fire(s.max_fire_s() + 1.0)


def test_power_limited_when_array_small():
    s = state(array_area_m2=10.0, battery_capacity_j=2e6, radiator_area_m2=1e3)
    assert s.steady_state_max_duty() == pytest.approx(s.p_charge / s.p_elec)
    assert s.max_fire_s() == pytest.approx(2e6 / (s.p_elec - s.p_charge))


def test_burst_that_never_fits():
    s = state(pcm_mass_kg=0.0, heat_capacity_j_k=1e4)
    assert np.isinf(s.wait_for(10.0))


def test_advance_backwards_raises():
    s = state()
    s.advance(10.0)
    with pytest.raises(ValueError):
        s.advance(5.0)


def test_hardware_life_counters():
    rs = ResourceState(ResourceConfig(), LaserConfig())
    cap = rs.cfg.battery_capacity_j
    de = rs.p_elec - rs.p_charge  # battery draw while firing, fully lit
    rs.fire(10.0)
    rs.advance(rs.t + 3600.0)  # recharges; no discharge while idle in sunlight
    rs.fire(5.0)
    assert rs.bursts == 2
    assert rs.discharge_j == pytest.approx(15.0 * de)
    assert rs.burst_dod_max == pytest.approx(10.0 * de / cap)
    assert rs.burst_dod_sum == pytest.approx(15.0 * de / cap)


def test_summary_reports_hardware_life():
    from talos.platform import baseline_platform
    from talos.population import toy_population
    from talos.simulation import run_engagements

    sat = baseline_platform(800.0)
    pop = toy_population(3000, np.random.default_rng(1), n_real=3000)
    res = run_engagements([sat], pop, 1800.0)
    s = res.summary()
    yr = 365.25 * 86400.0 / 1800.0
    assert s["bursts_per_sat_per_yr"] == pytest.approx(len(res.events) * yr)
    assert s["laser_shots_per_sat_per_yr"] == pytest.approx(sum(e.n_pulses for e in res.events) * yr)
    assert 0.0 < s["battery_dod_mean"] <= s["battery_dod_max"] <= 1.0
    assert s["battery_efc_per_sat_per_yr"] > 0
