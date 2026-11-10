"""Mass / life / cost model (talos.cost)."""

import numpy as np
import pytest

from talos.config import load_scenario
from talos.cost import evaluate, platform_mass_kg


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
    total = 4 * (m * (50e3 + 3e3) + 2e6 * 0.5)
    assert c["cost_musd"] == pytest.approx(total / 1e6)
    assert c["cost_per_removal_kusd"] == pytest.approx(total / 1000.0 / 1e3)
    c2 = evaluate(sc, {**row, "storage_life_yr": np.inf, "laser_shots_per_sat_per_yr": 5e9})
    assert c2["life_limit"] == "laser" and c2["mission_life_yr"] == pytest.approx(2.0)
    assert evaluate(sc, {**row, "removed_credited_per_yr": 0.0})["cost_per_removal_kusd"] == np.inf
