"""Mass / life / cost model (talos.cost)."""

import numpy as np
import pytest

from talos.config import load_scenario
from talos.cost import evaluate, platform_mass_kg


@pytest.fixture(autouse=True)
def _no_flux_table(tmp_path, monkeypatch):
    """Keep these tests independent of a locally built ORDEM platform flux table."""
    from talos import survival

    monkeypatch.setattr(survival, "DATA_DIR", tmp_path)


def test_mass_breakdown_baseline():
    sc = load_scenario()
    m = platform_mass_kg(sc.resources, sc.laser, sc.cost)
    assert m["radiator"] == pytest.approx(200.0)  # 20 m2 x 10 kg/m2
    assert m["storage"] == pytest.approx(2000.0 / 120.0)  # 2 kWh at 120 Wh/kg
    assert m["laser"] == pytest.approx(25.0 * 20.0)
    assert m["array"] == pytest.approx(sc.resources.array_area_m2 * 250.0 / 100.0)
    payload = sum(v for k, v in m.items() if k not in ("bus", "total"))
    assert m["total"] == pytest.approx(1.3 * payload)


def test_life_limit_and_cost_per_removal():
    sc = load_scenario()
    row = {"n_platforms": 4, "storage_life_yr": 0.5, "laser_shots_per_sat_per_yr": 3e7,
           "removed_credited_per_yr": 2000.0}
    c = evaluate(sc, row)
    assert c["life_limit"] == "storage" and c["mission_life_yr"] == 0.5
    m = c["mass_total_kg"]
    prop = c["mass_propellant_kg"]  # mean over the 4 platforms; cost is linear in it
    total = 4 * ((m + c["mass_tank_kg"]) * (50e3 + 3e3) + prop * (3e3 + 3e3) + 2e6 * 0.5)
    assert c["cost_musd"] == pytest.approx(total / 1e6)
    assert c["cost_per_removal_kusd"] == pytest.approx(total / 1000.0 / 1e3)
    c2 = evaluate(sc, {**row, "storage_life_yr": np.inf, "laser_shots_per_sat_per_yr": 5e9})
    assert c2["life_limit"] == "laser" and c2["mission_life_yr"] == pytest.approx(2.0)
    assert evaluate(sc, {**row, "removed_credited_per_yr": 0.0})["cost_per_removal_kusd"] == np.inf


def test_cost_per_debris_year():
    sc = load_scenario()
    row = {"n_platforms": 4, "storage_life_yr": 2.0, "laser_shots_per_sat_per_yr": 3e7,
           "removed_credited_per_yr": 500.0, "debris_years_removed_per_yr": 4e5,
           "removed": 80, "removed_credited": 20}
    c = evaluate(sc, row)
    assert c["debris_years_mission"] == pytest.approx(8e5)
    assert c["cost_per_debris_year_usd"] == pytest.approx(c["cost_musd"] * 1e6 / 8e5)
    assert c["credited_fraction"] == pytest.approx(0.25)
    assert evaluate(sc, {**row, "debris_years_removed_per_yr": 0.0})["cost_per_debris_year_usd"] == np.inf


def test_storage_life_adds_calendar_ageing():
    from talos.simulation import _storage_life_yr

    assert _storage_life_yr(1 / 2.66, np.inf) == pytest.approx(2.66)
    assert _storage_life_yr(1 / 2.66, 10.0) == pytest.approx(1 / (1 / 2.66 + 0.1))
    assert _storage_life_yr(0.0, 10.0) == pytest.approx(10.0)  # idle storage still ages
    assert _storage_life_yr(0.0, np.inf) == np.inf
    d = load_scenario("config/design.toml").resources
    assert d.storage_calendar_life_yr == 10.0 and d.storage_cycles_100 == 5e5


def test_station_keeping_dv_matches_hand_calc():
    from talos.constants import MU_EARTH, R_EARTH, YEAR
    from talos.cost import drag_area_m2, station_keeping_dv
    from talos.propagator import exponential_density

    sc = load_scenario(overrides={"atmosphere.model": "exponential", "constellation.n_sats": 3})
    area = drag_area_m2(sc.resources, sc.cost)
    assert area == pytest.approx(10.0 + 0.15 * (sc.resources.array_area_m2 + sc.resources.radiator_area_m2))
    dv = station_keeping_dv(sc, 3000.0, area, 2.0)
    r = R_EARTH + np.array([700e3, 800e3, 900e3])
    hand = 0.5 * exponential_density(r - R_EARTH) * MU_EARTH / r * 2.2 * area / 3000.0 * 2 * YEAR
    np.testing.assert_allclose(dv, hand, rtol=1e-9)
    assert dv[0] > dv[1] > dv[2] > 0


def test_station_keeping_propellant_in_cost():
    sc = load_scenario(overrides={"cost.drag_area_frac": 5.0})  # exaggerate to see it
    row = {"n_platforms": 4, "storage_life_yr": 3.0, "laser_shots_per_sat_per_yr": 3e7}
    c = evaluate(sc, row)
    dv = c["sk_dv_mission_m_s"]
    assert dv > 0 and c["sk_dv_max_m_s"] >= dv
    assert c["mass_propellant_kg"] == pytest.approx(c["mass_total_kg"] * np.expm1(dv / (9.80665 * 1500.0)),
                                                    rel=0.05)  # mean of expm1 ~ expm1 of mean at small dv
    assert c["mass_tank_kg"] == pytest.approx(0.1 * c["mass_propellant_kg"])
