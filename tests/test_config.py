from dataclasses import fields

import numpy as np
import pytest

from talos.atmosphere import AtmosphereConfig
from talos.config import RiskConfig, Scenario, load_scenario, scenario_dict
from talos.cost import CostConfig
from talos.dynamics import DynamicsConfig
from talos.engagement import AblationParams
from talos.platform import ATPConfig, LaserConfig, ResourceConfig
from talos.sun import SunConfig


def test_baseline_toml_matches_dataclass_defaults():
    """config/baseline.toml and the dataclass defaults describe the same platform."""
    sc = load_scenario()
    for got, ref in ((sc.laser, LaserConfig()), (sc.atp, ATPConfig()), (sc.resources, ResourceConfig()),
                     (sc.ablation, AblationParams()), (sc.sun, SunConfig()),
                     (sc.atmosphere, AtmosphereConfig()), (sc.dynamics, DynamicsConfig()),
                     (sc.risk, RiskConfig()), (sc.cost, CostConfig())):
        for f in fields(ref):
            a, b = getattr(got, f.name), getattr(ref, f.name)
            if isinstance(b, float):
                assert a == pytest.approx(b, rel=1e-12), (type(ref).__name__, f.name)
            else:
                assert a == b, (type(ref).__name__, f.name)


def test_unit_conversions():
    sc = load_scenario()
    assert sc.laser.burst_power_w == 25e3
    assert sc.laser.wavelength_m == pytest.approx(355e-9)
    assert sc.atp.slew_rate_limit_rad_s == pytest.approx(np.deg2rad(2.0))
    assert sc.resources.pcm_melt_k == pytest.approx(288.15)
    assert sc.resources.battery_capacity_j == pytest.approx(7.2e6)
    assert sc.ablation.cm == pytest.approx(50e-6)
    assert sc.run.duration_s == 86400.0
    assert sc.planner.horizon_s == 6 * 3600.0
    assert sc.run.epoch.year == 2026 and sc.run.epoch.tzinfo is not None


def test_overrides_accept_unit_or_si_form():
    a = load_scenario(overrides={"laser.burst_power_kw": 10})
    b = load_scenario(overrides={"laser.burst_power_w": 10e3})
    assert a.laser.burst_power_w == b.laser.burst_power_w == 10e3
    c = load_scenario(overrides={"population.min_lifetime_yr": None, "constellation.altitudes_km": [800.0]})
    assert c.population.min_lifetime_yr is None and c.constellation.altitudes_km == (800.0,)


def test_unknown_keys_raise(tmp_path):
    with pytest.raises(KeyError):
        load_scenario(overrides={"laser.bogus": 1})
    with pytest.raises(KeyError):
        load_scenario(overrides={"nosection.x": 1})
    p = tmp_path / "bad.toml"
    p.write_text("[laser]\nburst_power_kw = 1\nburst_power_w = 2\n")
    with pytest.raises(KeyError):
        load_scenario(p)


def test_scenario_dict_is_flat():
    d = scenario_dict(load_scenario())
    assert d["laser.burst_power_w"] == 25e3 and "sun.enabled" in d
    assert isinstance(Scenario(), Scenario)


def test_base_scenario_inheritance(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text('[run]\nname = "b"\n[resources]\nradiator_area_m2 = 20.0\nburst_s = 10.0\n')
    child = tmp_path / "sub" / "child.toml"
    child.parent.mkdir()
    child.write_text('base = "../base.toml"\n[resources]\nradiator_area_m2 = 60.0\n')
    sc = load_scenario(child)
    assert sc.resources.radiator_area_m2 == 60.0  # replaced
    assert sc.resources.burst_s == 10.0  # inherited
    assert sc.run.name == "b"
    # a unit-suffixed spelling in the child replaces the SI field from the base
    child.write_text('base = "../base.toml"\n[resources]\nradiator_area_ft2 = 100.0\n')
    assert load_scenario(child).resources.radiator_area_m2 == pytest.approx(100.0 * 0.09290304)


def test_base_scenario_cycle(tmp_path):
    a, b = tmp_path / "a.toml", tmp_path / "b.toml"
    a.write_text('base = "b.toml"\n')
    b.write_text('base = "a.toml"\n')
    with pytest.raises(ValueError, match="cycle"):
        load_scenario(a)


def test_design_scenario():
    from talos.config import DEFAULT_CONFIG
    from talos.platform import STORAGE_PRESETS

    sc = load_scenario(DEFAULT_CONFIG.parent / "design.toml")
    base = load_scenario()
    assert sc.resources.radiator_area_m2 == 60.0 and sc.resources.burst_s == 20.0
    for k, v in STORAGE_PRESETS["supercap"].items():
        assert getattr(sc.resources, k) == pytest.approx(v), k
    assert sc.laser == base.laser and sc.ablation == base.ablation
