import numpy as np
import pytest

from talos.constants import R_EARTH
from talos.debris import DebrisPopulation
from talos.elements import KeplerElements
from talos.platform import ATPConfig, baseline_constellation, baseline_platform
from talos.population import toy_population
from talos.screening import (
    apsis_filter,
    find_windows,
    los_angular_rate,
    los_clear,
    screen,
    screen_fine,
)


def circ(alt_km, **kw):
    return KeplerElements(a=R_EARTH + np.asarray(alt_km) * 1e3, e=0.0, i=1.7, raan=0.0, argp=0.0, M=0.0, **kw)


def test_apsis_filter():
    sat = circ(700.0)
    deb = circ([1000.0, 1300.0, 650.0])
    assert apsis_filter(sat, deb, 500e3).tolist() == [True, False, True]


def test_apsis_filter_eccentric_band_overlap():
    sat = circ(800.0)
    # perigee 300 km, apogee ~ 1700 km straddles the sat shell
    a = R_EARTH + 1000e3
    e = 700e3 / a
    deb = KeplerElements(a=a, e=e, i=1.0, raan=0, argp=0, M=0)
    assert apsis_filter(sat, deb, 10e3)


def test_los_clear():
    r = R_EARTH + 800e3
    obs = np.array([r, 0, 0])
    assert los_clear(obs, np.array([r, 300e3, 0]), R_EARTH + 100e3)
    assert not los_clear(obs, np.array([-r, 0, 0]), R_EARTH + 100e3)


def test_los_angular_rate_straight_pass():
    b, v = 10e3, 15e3
    t = np.array([-5.0, 0.0, 5.0])
    r_rel = np.stack([np.full(3, b), v * t, np.zeros(3)], -1)
    v_rel = np.tile([0.0, v, 0.0], (3, 1))
    w = los_angular_rate(r_rel, v_rel)
    assert w[1] == pytest.approx(v / b)
    assert np.allclose(w, v * b / (b**2 + (v * t) ** 2))


def test_find_windows():
    mask = np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    w = [(x.debris_index, x.start, x.stop) for x in find_windows(mask)]
    assert w == [(0, 1, 3), (1, 0, 2), (2, 0, 1), (2, 3, 4)]
    assert [(x.debris_index, x.start) for x in find_windows(mask, min_steps=2)] == [(0, 1), (1, 0)]


def test_screen_fine_gates_individually():
    atp = ATPConfig()
    t = np.zeros(1)
    r_sat = np.array([[R_EARTH + 800e3, 0, 0]])
    v_sat = np.array([[0, 7450.0, 0]])
    # 0: 200 km ahead, co-moving (pass); 1: 50 km (too close);
    # 2: 200 km ahead, but fast crossing -> slew limit; 3: behind Earth
    r_deb = r_sat[:, None, :] + np.array([[[0, 200e3, 0], [0, 50e3, 0], [0, 200e3, 0], [0, 0, 0]]])
    r_deb[0, 3] = [-(R_EARTH + 800e3), 0, 0]
    v_deb = np.repeat(v_sat[:, None, :], 4, axis=1).copy()
    v_deb[0, 2] += [15e3, 0, 0]  # 15 km/s / 200 km = 4.3 deg/s
    res = screen_fine(t, r_sat, v_sat, r_deb, v_deb, atp)
    assert res.mask[0].tolist() == [True, False, False, False]
    assert not res.slew_ok[0, 2]
    assert not res.los_ok[0, 3]


def test_screen_coorbital_trailing_object():
    sat = baseline_platform(800.0)
    o = sat.orbit
    deb_el = KeplerElements(
        a=o.a,
        e=0.0,
        i=o.i,
        raan=o.raan,
        argp=0.0,
        M=np.array([-np.deg2rad(1.0), np.pi, 0.0]),  # ~125 km behind; antipodal; filtered by apsis
    )
    deb_el.a[2] = R_EARTH + 2000e3
    pop = DebrisPopulation(elements=deb_el, diameter_m=0.05, areal_density=1.0, weight=1.0)
    t = np.arange(0.0, 600.0, 1.0)
    keep, res, windows = screen(sat, pop, t)
    assert keep.tolist() == [0, 1]
    assert res.mask.shape == (600, 2)
    assert np.allclose(res.range_m[:, 0], 2 * o.a * np.sin(np.deg2rad(0.5)), rtol=1e-3)
    assert [(w.debris_index, w.start, w.stop) for w in windows] == [(0, 0, 600)]


def test_screen_toy_population_runs(rng):
    pop = toy_population(500, rng)
    sat = baseline_constellation(4)[1]
    t = np.arange(0.0, 300.0, 1.0)
    keep, res, windows = screen(sat, pop, t)
    assert res.mask.shape == (300, len(keep))
    assert all(0 <= w.debris_index < len(pop) for w in windows)


@pytest.mark.skip(reason="Gate 3: needs L'ADROIT head-on geometry (miss distance); "
                  "w = v_rel*b/r^2 implies b ~ 9 km for 2 deg/s at 63 km with v_rel ~ 15 km/s")
def test_gate3_slew_limit_head_on():
    pass
