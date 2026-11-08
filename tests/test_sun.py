from datetime import datetime, timezone

import numpy as np
import pytest

from talos.constants import R_EARTH
from talos.platform import baseline_platform
from talos.resources import ResourceState
from talos.screening import screen_windows
from talos.sun import AU, SunConfig, SunEnvironment, julian_date, lit_fraction, sun_vector, target_lit


def env_at(date: str, **kw) -> SunEnvironment:
    return SunEnvironment(SunConfig(**kw), datetime.fromisoformat(date + "T00:00:00+00:00"))


def test_julian_date_j2000():
    assert julian_date(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)) == pytest.approx(2451545.0)


def test_sun_declination_and_distance():
    for date, dec in (("2026-12-21", -23.44), ("2026-06-21", 23.44), ("2026-03-20", 0.0)):
        u, d = env_at(date).sun(0.0)
        assert np.degrees(np.arcsin(u[2])) == pytest.approx(dec, abs=0.5)
    _, d_jan = sun_vector(julian_date(datetime(2027, 1, 3, tzinfo=timezone.utc)))
    _, d_jul = sun_vector(julian_date(datetime(2027, 7, 4, tzinfo=timezone.utc)))
    assert d_jan == pytest.approx(0.9833 * AU, rel=1e-3) and d_jul == pytest.approx(1.0167 * AU, rel=1e-3)


def test_shadow_geometry():
    s = np.array([1.0, 0.0, 0.0])
    r = R_EARTH + 800e3
    assert lit_fraction(np.array([r, 0, 0]), s, AU) == 1.0  # sunward
    assert lit_fraction(np.array([-r, 0, 0]), s, AU) == 0.0  # behind Earth
    assert lit_fraction(np.array([0, r, 0]), s, AU) == 1.0  # terminator
    # cylindrical target test agrees away from the penumbra
    assert target_lit(np.array([-r, 0, 0]), s) == False  # noqa: E712
    assert target_lit(np.array([-r, 1.2 * R_EARTH, 0]), s) == True  # noqa: E712


def test_dawn_dusk_eclipse_season():
    """06:00 LTAN at 800 km: ~17 min eclipse/orbit at the December solstice, none in June."""
    for date, lo, hi in (("2026-12-21", 0.14, 0.20), ("2026-06-21", 0.0, 0.0)):
        env = env_at(date)
        p = baseline_platform(800.0, raan_rad=env.raan_for_ltan(6.0))
        tl = env.timeline(p, 86400.0)
        assert lo <= tl.eclipse_fraction <= hi


def test_ltan_sets_raan_relative_to_sun():
    env = env_at("2026-03-20")
    u, _ = env.sun(0.0)
    ra = np.arctan2(u[1], u[0])
    assert np.angle(np.exp(1j * (env.raan_for_ltan(6.0) - ra))) == pytest.approx(-np.pi / 2, abs=1e-9)


def eclipsed_state():
    env = env_at("2026-12-21")
    p = baseline_platform(800.0, raan_rad=env.raan_for_ltan(6.0))
    tl = env.timeline(p, 4 * 3600.0)
    return ResourceState(p.resources, p.laser, env=tl), tl


def test_no_firing_in_eclipse():
    rs, tl = eclipsed_state()
    k = int(np.argmax(~tl.lit))  # first eclipsed segment
    rs.advance(k * tl.step_s + 1.0)
    assert not rs.sunlit and rs.max_fire_s() == 0.0
    w = rs.wait_for(10.0)
    rs.advance(rs.t + w)
    assert rs.sunlit and rs.max_fire_s() >= 10.0 - 1e-6
    exit_t = (k + int(np.argmax(tl.lit[k:]))) * tl.step_s
    assert rs.t == pytest.approx(exit_t, abs=0.5)


def test_battery_drains_in_eclipse():
    rs, tl = eclipsed_state()
    k = int(np.argmax(~tl.lit))
    n = int(np.argmax(tl.lit[k:]))
    # measure across the umbra only (penumbra segments still deliver partial array power)
    assert np.all(tl.array_factor[k + 1:k + n - 1] == 0.0)
    rs.advance((k + 1) * tl.step_s)
    soc0, heat0 = rs.soc_j, rs.heater_j
    rs.advance((k + n - 1) * tl.step_s)
    drain = rs.cfg.bus_power_w * (n - 2) * tl.step_s + (rs.heater_j - heat0)
    assert rs.soc_j == pytest.approx(soc0 - drain, rel=1e-6)
    assert rs.brownout_s == 0.0


def test_sun_gates_remove_windows():
    env = env_at("2026-12-21")
    sat = baseline_platform(800.0, raan_rad=env.raan_for_ltan(6.0))
    from talos.population import toy_population

    pop = toy_population(20000, np.random.default_rng(3))
    t = np.arange(0.0, 3600.0)
    sun = env.sun_hat(t)
    n0 = sum(w.n_steps for w in screen_windows(sat, pop, t))
    for kernel in (True, False):
        n1 = sum(w.n_steps for w in screen_windows(sat, pop, t, sun_hat=sun, use_kernel=kernel))
        assert 0 < n1 < n0
    a = [(w.debris_index, w.start, w.stop) for w in screen_windows(sat, pop, t, sun_hat=sun, use_kernel=True)]
    b = [(w.debris_index, w.start, w.stop) for w in screen_windows(sat, pop, t, sun_hat=sun, use_kernel=False)]
    assert a == b
