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


def test_lasered_path_includes_in_run_residence():
    """With one shell covering every altitude, residence = time alive. Natural path:
    residence = lifetime. Lasered path: the whole run (in-run accrual) + the remaining life
    from the last engagement, which is where the final state's epoch sits."""
    from talos.platform import baseline_platform
    from talos.population import toy_population
    from talos.simulation import run_engagements

    pop = toy_population(4000, np.random.default_rng(3), alt_km=(900.0, 1000.0), n_real=4000, areal_density=10.0)
    res = run_engagements([baseline_platform(950.0)], pop, 1800.0, risk_shells={"all": (0.0, 5e6)})
    assert len(res.events) > 0 and not res.removed.any()
    hit = np.array(list(res.lifetime_initial_s))
    li = np.array([res.lifetime_initial_s[j] for j in hit])
    lf = np.array([res.lifetime_final_s[j] for j in hit])  # measured from t = 0
    tm = {e.debris_index: 0.5 * (e.t_start + e.t_stop) for e in res.events}  # last engagement wins
    tm = np.array([tm[j] for j in hit])
    assert np.all(np.isfinite(li))
    np.testing.assert_allclose(res.shell_res_initial_s[:, 0], li, rtol=1e-6)
    np.testing.assert_allclose(res.shell_res_final_s[:, 0], 1800.0 + (lf - tm), rtol=1e-6)


def test_active_collision_metric():
    from talos import risk
    from talos.constants import R_EARTH, YEAR

    sc = load_scenario()
    k = risk.shell_rates_per_yr(sc.risk)
    r = R_EARTH + 552.5e3
    assert k["starlink"] == pytest.approx(2.4e5 * 1e4 / (4 * np.pi * r**2 * 45e3) * YEAR)
    row = {"sim_hours": 24.0 * 365.25 / 12,  # one month
           "risk_crewed_obj_yr_natural": 1.0, "risk_crewed_obj_yr_after": 3.0,
           "risk_starlink_obj_yr_natural": 2.0, "risk_starlink_obj_yr_after": 4.0,
           "risk_band_obj_yr_natural": 100.0, "risk_band_obj_yr_after": 20.0}
    out = risk.evaluate(sc, row)
    nat = k["crewed"] * 1 + k["starlink"] * 2 + k["band"] * 100
    aft = k["crewed"] * 3 + k["starlink"] * 4 + k["band"] * 20
    assert out["active_collisions_avoided"] == pytest.approx(nat - aft)
    assert out["active_collisions_avoided_per_yr"] == pytest.approx(12 * (nat - aft))
    assert out["risk_starlink_collisions_after"] > out["risk_starlink_collisions_natural"]  # risk transfer


def test_fragment_risk_counts_against_the_laser():
    from talos import risk

    sc = load_scenario()
    k = risk.shell_rates_per_yr(sc.risk)
    row = {"sim_hours": 24.0, "risk_band_obj_yr_natural": 10.0, "risk_band_obj_yr_after": 4.0,
           "risk_band_obj_yr_fragments": 5.0}
    out = risk.evaluate(sc, row)
    assert out["active_collisions_added_by_fragments"] == pytest.approx(5 * k["band"])
    assert out["active_collisions_avoided"] == pytest.approx((10 - 4 - 5) * k["band"])
    s = run_scenario(load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000,
                                              "constellation.n_sats": 1, "population.min_lifetime_yr": None,
                                              "run.duration_s": 3600.0, "ablation.fragment_prob_per_burst": 1.0,
                                              "population.areal_density_model": "sbm"})).summary()
    assert s["risk_band_obj_yr_fragments"] > 0
