"""Risk transfer: shell residence from the lifetime integrator."""

import numpy as np
import pytest

from talos.config import load_scenario, run_scenario
from talos.constants import R_EARTH, YEAR
from talos.lifetime import orbital_lifetime, shell_fraction


def test_shell_fraction_circular_and_elliptic():
    sh = np.array([[370e3, 430e3], [530e3, 575e3]])
    f = shell_fraction(R_EARTH + np.array([400e3, 550e3, 700e3]), np.zeros(3), sh)
    np.testing.assert_array_equal(f, [[1, 0], [0, 1], [0, 0]])
    # perigee 400 km, apogee 800 km: Monte Carlo over mean anomaly
    rp, ra = R_EARTH + 400e3, R_EARTH + 800e3
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    M = np.random.default_rng(0).uniform(0, 2 * np.pi, 200000)
    E = M.copy()
    for _ in range(30):
        E = M + e * np.sin(E)
    alt = a * (1 - e * np.cos(E)) - R_EARTH
    mc = [np.mean((alt >= lo) & (alt < hi)) for lo, hi in sh]
    np.testing.assert_allclose(shell_fraction(np.array([a]), np.array([e]), sh, n_quad=512)[0], mc, atol=0.005)


def test_circular_residence_equals_crossing_time():
    """A circular orbit starting at the shell top stays in the shell until perigee leaves its bottom."""
    B = 2.2 / 2.0
    _, res = orbital_lifetime(R_EARTH + 575e3, 0.0, B, shells_m=np.array([[530e3, 576e3]]))
    t530 = orbital_lifetime(R_EARTH + 575e3, 0.0, B, reentry_alt_m=530e3)
    assert res[0] == pytest.approx(t530, rel=0.02)


def test_residence_capped_at_horizon():
    life, res = orbital_lifetime(R_EARTH + 400e3, 0.0, 0.001, max_s=YEAR, shells_m=np.array([[370e3, 430e3]]))
    assert np.isinf(life) and res[0] == pytest.approx(YEAR, rel=1e-6)


def test_summary_reports_risk_shells():
    sc = load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000, "constellation.n_sats": 1,
                                  "population.min_lifetime_yr": None, "run.duration_s": 1800.0,
                                  "population.areal_density": 0.2, "population.areal_density_model": "fixed"})
    s = run_scenario(sc).summary()
    for n in ("crewed", "starlink"):
        assert {f"risk_{n}_obj_yr_natural", f"risk_{n}_obj_yr_after", f"risk_{n}_delta_obj_yr_per_yr"} <= set(s)
        assert s[f"risk_{n}_obj_yr_natural"] >= 0 and s[f"risk_{n}_obj_yr_after"] >= 0
    off = run_scenario(load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000,
                                                "constellation.n_sats": 1, "population.min_lifetime_yr": None,
                                                "run.duration_s": 600.0, "risk.enabled": False})).summary()
    assert not any(k.startswith("risk_") for k in off)
