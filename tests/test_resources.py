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


@pytest.mark.parametrize("with_env", [False, True])
def test_numba_state_machine_matches_python(monkeypatch, with_env):
    """Same random sequence of idle / wait / fire through both paths: bit-identical state."""
    from talos import _kernels
    from talos.sun import Timeline

    if not _kernels.HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(4)
    env = None
    if with_env:
        n = 4000
        lit = (np.arange(n) % 590) < 480  # ~98-min orbit at 10 s steps with an eclipse
        env = Timeline(10.0, lit, np.where(lit, rng.uniform(0.8, 1.0, n), 0.0), rng.uniform(0, 400, n))
    ops = [(rng.choice(["idle", "wait", "fire"]), rng.uniform(1, 400)) for _ in range(250)]

    def run():
        s = ResourceState(ResourceConfig(radiator_area_m2=8.0), LaserConfig(), env=env, fire_in_eclipse=False)
        log = []
        for op, x in ops:
            if op == "idle":
                s.advance(s.t + x)
            elif op == "wait":
                w = s.wait_for(10.0)
                log.append(w)
                if np.isfinite(w):
                    s.advance(s.t + w)
            else:
                f = min(10.0, s.max_fire_s())
                log.append(f)
                if f > 0.5:
                    s.fire(f)
            log += [s.t, s.heat_j, s.soc_j, s.heater_j, s.brownout_s, s.discharge_j]
        return log

    got = run()
    monkeypatch.setattr(_kernels, "HAVE_NUMBA", False)
    ref = run()
    assert got == ref  # exact
    assert any(v > 0 for v in got[1::7])  # the run does wait / fire


def test_ready_burst_capped_at_feasible():
    las = LaserConfig()
    feasible = ResourceState(ResourceConfig(), las).max_fire_s()
    assert ResourceConfig(burst_s=10.0).ready_s(las) == 10.0  # fits: unchanged
    assert ResourceConfig(burst_s=60.0).ready_s(las) == pytest.approx(0.95 * feasible)
    assert ResourceConfig(burst_s=60.0, ready_burst_s=15.0).ready_s(las) == 15.0
    # a wait for the capped length terminates; for the uncapped one it never does
    s = ResourceState(ResourceConfig(burst_s=60.0), las)
    s.fire(8.0)
    assert np.isfinite(s.wait_for(ResourceConfig(burst_s=60.0).ready_s(las)))
    assert np.isinf(s.wait_for(60.0))


def test_storage_damage_miner_rule():
    cfg = ResourceConfig(storage_cycles_100=1000.0, storage_cycle_exponent=2.0)
    s = ResourceState(cfg, LaserConfig())
    s.fire(10.0)
    dod = s.burst_dod_max
    assert 0 < dod < 1
    assert s.storage_damage == pytest.approx(dod**2 / 1000.0)
    flat = ResourceState(ResourceConfig(storage_cycles_100=1e6, storage_cycle_exponent=0.0), LaserConfig())
    flat.fire(10.0)
    assert flat.storage_damage == pytest.approx(1e-6)  # supercap-like: one cycle per burst regardless of DoD
