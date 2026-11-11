"""Solar radiation pressure on debris eccentricity (talos.srp)."""

import numpy as np
import pytest

from talos.constants import MU_EARTH, R_EARTH
from talos.elements import KeplerElements
from talos.srp import P_SUN, lit_fraction_circular, srp_step


def _el(n=1, e=0.0, i=0.0, raan=0.0, argp=0.0, M=0.0, alt=800e3):
    f = lambda x: np.full(n, float(x))  # noqa: E731
    return KeplerElements(a=f(R_EARTH + alt), e=f(e), i=f(i), raan=f(raan), argp=f(argp), M=f(M))


def test_in_plane_sun_drives_e_perpendicular_at_the_averaged_rate():
    el = _el(alt=20000e3)
    dt = 86400.0
    out = srp_step(el, 2.0, np.array([1.0, 0.0, 0.0]), dt)
    a = el.a[0]
    lit = 1 - np.arcsin(R_EARTH / a) / np.pi  # sun in the orbit plane: cylindrical shadow arc
    rate = 1.5 / (np.sqrt(MU_EARTH / a**3) * a) * P_SUN * 1.2 * 2.0 * lit
    assert out.e[0] == pytest.approx(rate * dt, rel=1e-9)
    # F = -k x_hat, h_hat = z_hat: F x h_hat = +k y_hat -> perigee at +90 deg (perpendicular to the sun line)
    assert out.argp[0] == pytest.approx(np.pi / 2)
    du = np.angle(np.exp(1j * ((out.argp + out.M) - (el.argp + el.M))))[0]
    assert du == pytest.approx(0.0, abs=1e-12)  # mean argument of latitude kept
    np.testing.assert_array_equal(out.a, el.a)


def test_sun_along_orbit_normal_does_nothing():
    out = srp_step(_el(e=0.003, argp=1.0), 2.0, np.array([0.0, 0.0, 1.0]), 86400.0)
    assert out.e[0] == pytest.approx(0.003) and out.argp[0] == pytest.approx(1.0)


def test_cylindrical_shadow_fraction():
    a = np.array(R_EARTH + 800e3)
    f = lit_fraction_circular(a, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    assert f == pytest.approx(1 - np.arcsin(R_EARTH / a) / np.pi)  # sun in plane
    assert lit_fraction_circular(a, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])) == 1.0


def test_population_step_applies_srp_to_light_objects_only():
    from talos.config import build, load_scenario

    over = {"population.kind": "toy", "population.n_real": 400, "population.min_lifetime_yr": None,
            "population.areal_density_model": "sbm", "constellation.n_sats": 1, "dynamics.breakup_rate_per_yr": 0.0}
    res = {}
    for srp in (False, True):
        b = build(load_scenario(overrides={**over, "dynamics.srp": srp}))
        pop = b.population
        e0 = pop.elements.e.copy()
        gone = np.zeros(len(pop), bool)
        for d in range(1, 11):
            pop, _ = b.dynamics.step(pop, gone, d * 86400.0, initial=pop, engaged=np.ones(len(pop), bool))
        res[srp] = (pop.elements.e[: len(e0)] - e0, pop.areal_density[: len(e0)])
    de = np.abs(res[True][0] - res[False][0])
    mu = res[True][1]
    assert de.max() > 1e-4  # ~2e-4 / day for A/m ~ 2
    assert np.median(de[mu < 1]) > 5 * np.median(de[mu > 5])
