"""Platform debris-impact survivability (talos.survival) with a synthetic flux table."""

import numpy as np
import pytest

from talos import survival
from talos.survival import (SIZES_M, PlatformFluxTable, SurvivalConfig, expected_life, impact_rates,
                            reduce_directional)


def _table(scale=1.0):
    alts = np.array([700.0, 900.0])
    cum = scale * 1e-4 * (SIZES_M / 1e-2) ** -2.5  # power law, 1e-4 /m2/yr at 1 cm
    sphere = np.stack([2 * cum, cum])  # 700 km twice 900 km
    return PlatformFluxTable(2027, alts, SIZES_M,
                             {"sphere": sphere, "x": 0.8 * sphere, "y": 0.4 * sphere, "z": 0.01 * sphere})


def test_directional_reduction_weights_by_cosine():
    cols = [m + s for m in survival.MATERIALS for s in survival.SIZE_CODES_ALL]
    flux = np.zeros((2, len(cols)))
    flux[:, cols.index("MD40")] = [3.0, 1.0]  # >= 1 cm, MD only
    # element 0 from ram (az 0, el 0); element 1 from 60 deg starboard
    r = reduce_directional(np.array([-1.0, 59.0]), np.array([1.0, 61.0]), np.array([-1.0, -1.0]),
                           np.array([1.0, 1.0]), cols, flux)
    k = list(survival.SIZE_CODES_ALL).index("40")
    assert r["sphere"][k] == pytest.approx(4.0)
    assert r["x"][k] == pytest.approx(3.0 + 0.5, rel=1e-3)
    assert r["y"][k] == pytest.approx(np.sin(np.deg2rad(60)), rel=1e-3)
    assert r["z"][k] == pytest.approx(0.0, abs=1e-12)


def test_flux_interpolation_and_fine_curve():
    t = _table()
    assert t.at("sphere", 700.0, 1e-2) == pytest.approx(2e-4)
    assert t.at("sphere", 800.0, 1e-2) == pytest.approx(np.sqrt(2) * 1e-4)  # log-linear in altitude
    assert t.at("y", 900.0, 10**-2.25) == pytest.approx(0.4e-4 * 10 ** (0.25 * 2.5), rel=1e-9)  # log-log
    fine = np.geomspace(1e-5, 1.0, 101)
    t.fine_sizes_m, t.fine_sphere = fine, np.stack([np.full(101, 7.0), np.full(101, 5.0)])
    assert t.at("z", 900.0, 3e-3) == pytest.approx(0.05)  # fine flux x directional fraction


def test_rates_and_expected_life():
    cfg = SurvivalConfig()
    r = impact_rates(_table(), cfg, np.array([700.0, 900.0]), array_area_m2=100.0, radiator_area_m2=50.0)
    np.testing.assert_allclose(r["bus_lethal"], 10.0 * np.array([2e-4, 1e-4]))
    rad = 0.1 * 50.0 * 0.01 * 1e-4 * 10**2.5 * np.array([2.0, 1.0])  # >= 1 mm, radial normal
    np.testing.assert_allclose(r["radiator_puncture"], rad)
    np.testing.assert_allclose(r["hazard"], r["bus_lethal"] + r["radiator_puncture"])
    life, p = expected_life(np.array([0.0, 0.5]), 2.0)
    assert life[0] == pytest.approx(2.0) and p[0] == 1.0
    assert life[1] == pytest.approx((1 - np.exp(-1.0)) / 0.5) and p[1] == pytest.approx(np.exp(-1.0))


def test_cost_uses_service_life(tmp_path, monkeypatch):
    from talos.config import load_scenario
    from talos.cost import evaluate

    monkeypatch.setattr(survival, "DATA_DIR", tmp_path)
    row = {"n_platforms": 4, "storage_life_yr": 3.0, "laser_shots_per_sat_per_yr": 3e7,
           "debris_years_removed_per_yr": 4e5, "removed_credited_per_yr": 1000.0}
    sc = load_scenario()
    base = evaluate(sc, row)
    assert np.isnan(base["impact_hazard_per_yr"]) and base["service_life_yr"] == base["mission_life_yr"]
    _table(scale=300.0).save(tmp_path / "platform_flux_2027.npz")
    hit = evaluate(sc, row)
    assert hit["impact_flux_year"] == 2027 and hit["impact_hazard_per_yr"] > 0
    assert hit["service_life_yr"] < hit["mission_life_yr"] == base["mission_life_yr"]
    assert hit["debris_years_mission"] == pytest.approx(4e5 * hit["service_life_yr"])
    assert hit["cost_musd"] < base["cost_musd"]  # fewer ops years ...
    assert hit["cost_per_debris_year_usd"] > base["cost_per_debris_year_usd"]  # ... but hardware is sunk
    off = evaluate(load_scenario(overrides={"survival.enabled": False}), row)
    assert np.isnan(off["impact_hazard_per_yr"])
