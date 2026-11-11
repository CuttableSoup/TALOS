import numpy as np
import pytest

from talos.campaign import CampaignConfig, focus_need, reference_impulse, select_focus
from talos.config import load_scenario
from talos.engagement import AblationParams
from talos.platform import LaserConfig, baseline_platform
from talos.population import toy_population
from talos.simulation import run_engagements


def _pop(n=400, seed=3):
    pop = toy_population(n, np.random.default_rng(seed))
    r = np.random.default_rng(seed + 1)
    pop.diameter_m = r.uniform(0.01, 0.10, n)
    pop.areal_density = r.lognormal(1.0, 1.0, n)
    return pop


def test_select_focus_filters_and_orders():
    pop = _pop()
    dead = np.zeros(len(pop), bool)
    dead[:50] = True
    cfg = CampaignConfig(enabled=True, min_diameter_m=0.07, count=10, select="heaviest")
    idx = select_focus(cfg, pop, dead)
    elig = np.flatnonzero(~dead & (pop.diameter_m >= 0.07))
    assert len(idx) == 10 and np.all(idx >= 50) and np.all(pop.diameter_m[idx] >= 0.07)
    assert pop.areal_density[idx].min() >= np.sort(pop.areal_density[elig])[-10]
    light = select_focus(CampaignConfig(enabled=True, min_diameter_m=0.07, count=10, select="lightest"), pop, dead)
    assert pop.areal_density[light].max() <= np.sort(pop.areal_density[elig])[9]
    everyone = select_focus(CampaignConfig(enabled=True, min_diameter_m=0.07, count=0), pop, dead)
    assert np.array_equal(everyone, elig)
    assert len(select_focus(CampaignConfig(enabled=False), pop, dead)) == 0
    a = select_focus(CampaignConfig(enabled=True, count=5, seed=1, min_diameter_m=0.0), pop, dead)
    b = select_focus(CampaignConfig(enabled=True, count=5, seed=1, min_diameter_m=0.0), pop, dead)
    assert np.array_equal(a, b)


def test_reference_impulse_matches_definition():
    las, p = LaserConfig(), AblationParams()
    ref = reference_impulse(20.0, las, p)
    assert ref == pytest.approx(20.0 * 33.0 * p.eta1_nominal * np.cos(p.off_axis_mean_rad) * p.cm * p.phi_opt)


def test_exclusive_campaign_fires_only_at_focus():
    pop = _pop(3000, seed=5)
    sat = baseline_platform(800.0)
    cfg = CampaignConfig(enabled=True, min_diameter_m=0.05, count=300, min_impulse_frac=0.0)
    res = run_engagements([sat], pop, 3 * 3600.0, campaign=cfg)
    focus = set(select_focus(cfg, pop, np.zeros(len(pop), bool)).tolist())
    assert res.events, "expected some focus bursts in 3 h"
    assert all(e.focus and e.debris_index in focus for e in res.events)
    s = res.summary()
    assert s["focus_objects"] == 300 and s["focus_bursts"] == len(res.events) and s["focus_burst_frac"] == 1.0
    assert 0.0 <= s["focus_progress_mean"] <= s["focus_progress_max"] <= 1.0
    assert s["focus_dv_total_m_s"] > 0
    # dv still needed only falls for engaged objects
    hit = np.array(sorted({e.debris_index for e in res.events}))
    assert np.all(res.focus_need1 <= res.focus_need0 + 1e-6)
    assert np.all(focus_need(res.population, hit) <= focus_need(pop, hit) + 1e-6)


def test_campaign_off_leaves_run_unchanged():
    pop = _pop(500, seed=7)
    sat = baseline_platform(800.0)
    a = run_engagements([sat], pop, 1800.0)
    b = run_engagements([sat], pop, 1800.0, campaign=CampaignConfig(enabled=False))
    assert a.records() == b.records()
    assert "focus_objects" not in b.summary()


def test_scenario_has_campaign_section():
    sc = load_scenario(None, {"campaign.enabled": True, "campaign.count": 7})
    assert sc.campaign.enabled and sc.campaign.count == 7
