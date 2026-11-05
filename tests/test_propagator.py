import numpy as np
import pytest

from talos.constants import DAY, MU_EARTH, R_EARTH, SSO_RAAN_RATE
from talos.elements import KeplerElements, elements_to_rv, rv_to_elements, solve_kepler
from talos.propagator import (
    drag_da_dt,
    exponential_density,
    j2_secular_rates,
    propagate,
    sso_inclination,
    states_at,
)


def random_elements(rng, n=200):
    return KeplerElements(
        a=R_EARTH + rng.uniform(300e3, 2000e3, n),
        e=rng.uniform(0.0, 0.2, n),
        i=rng.uniform(0.05, np.pi - 0.05, n),
        raan=rng.uniform(0, 2 * np.pi, n),
        argp=rng.uniform(0, 2 * np.pi, n),
        M=rng.uniform(0, 2 * np.pi, n),
    )


def test_kepler_residual(rng):
    M = rng.uniform(-10, 10, 1000)
    e = rng.uniform(0, 0.95, 1000)
    E = solve_kepler(M, e)
    assert np.allclose(np.mod(E - e * np.sin(E), 2 * np.pi), np.mod(M, 2 * np.pi), atol=1e-10)


def test_circular_orbit_radius_and_speed():
    a = R_EARTH + 800e3
    el = KeplerElements(a=a, e=0.0, i=1.0, raan=0.3, argp=0.0, M=np.linspace(0, 6, 7))
    r, v = elements_to_rv(el)
    assert r.shape == (7, 3)
    assert np.allclose(np.linalg.norm(r, axis=-1), a)
    assert np.allclose(np.linalg.norm(v, axis=-1), np.sqrt(MU_EARTH / a))
    assert np.allclose(np.sum(r * v, -1), 0.0, atol=1e-3)


def test_vis_viva(rng):
    el = random_elements(rng)
    r, v = elements_to_rv(el)
    rn = np.linalg.norm(r, axis=-1)
    assert np.allclose(np.sum(v * v, -1), MU_EARTH * (2 / rn - 1 / el.a), rtol=1e-10)


def test_rv_roundtrip(rng):
    el = random_elements(rng)
    r, v = elements_to_rv(el)
    el2 = rv_to_elements(r, v)
    assert np.allclose(el2.a, el.a, rtol=1e-9)
    assert np.allclose(el2.e, el.e, atol=1e-9)
    assert np.allclose(el2.i, el.i, atol=1e-9)
    r2, v2 = elements_to_rv(el2)
    assert np.allclose(r2, r, atol=1e-3)
    assert np.allclose(v2, v, atol=1e-6)


def test_rv_roundtrip_circular():
    el = KeplerElements(a=R_EARTH + 700e3, e=0.0, i=1.7, raan=2.0, argp=0.0, M=[0.1, 3.0, 5.5])
    r, v = elements_to_rv(el)
    r2, _ = elements_to_rv(rv_to_elements(r, v))
    assert np.allclose(r2, r, atol=1e-3)


def test_sso_inclination_800km():
    assert np.rad2deg(sso_inclination(R_EARTH + 800e3)) == pytest.approx(98.6, abs=0.1)


def test_sso_raan_rate():
    a = R_EARTH + 800e3
    el = KeplerElements(a=a, e=0.0, i=sso_inclination(a), raan=0.0, argp=0.0, M=0.0)
    raan_dot, _, _ = j2_secular_rates(el)
    assert raan_dot == pytest.approx(SSO_RAAN_RATE, rel=1e-9)
    assert np.rad2deg(propagate(el, DAY).raan) == pytest.approx(0.9856, abs=1e-3)


def test_critical_inclination_freezes_argp():
    el = KeplerElements(a=R_EARTH + 800e3, e=0.01, i=np.arccos(np.sqrt(1 / 5)), raan=0, argp=0, M=0)
    _, argp_dot, _ = j2_secular_rates(el)
    assert abs(argp_dot) < 1e-15


def test_states_at_shape(rng):
    el = random_elements(rng, n=5)
    t = np.arange(0.0, 60.0, 1.0)
    r, v = states_at(el, t)
    assert r.shape == v.shape == (60, 5, 3)
    r0, _ = elements_to_rv(el)
    assert np.allclose(r[0], r0)
    r_s, _ = states_at(el[0], t)
    assert r_s.shape == (60, 3)
    assert np.allclose(r_s, r[:, 0])


def test_density_table_anchors_and_monotone():
    assert exponential_density(800e3) == pytest.approx(1.170e-14)
    h = np.linspace(150e3, 1200e3, 500)
    assert np.all(np.diff(exponential_density(h)) < 0)


def test_drag_decay():
    a = R_EARTH + np.array([600e3, 900e3])
    da = drag_da_dt(a, ballistic_coeff=2.2)
    assert np.all(da < 0)
    assert abs(da[0]) > abs(da[1])
    el = KeplerElements(a=a, e=0, i=1.7, raan=0, argp=0, M=0)
    assert np.all(propagate(el, DAY, ballistic_coeff=2.2).a < a)
    assert np.allclose(propagate(el, DAY).a, a)
