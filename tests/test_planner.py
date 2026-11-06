import numpy as np
import pytest

from talos.debris import DebrisPopulation
from talos.elements import KeplerElements
from talos.engagement import AblationParams
from talos.planner import CommitPlanner, Opportunity, _ObjState
from talos.platform import baseline_platform
from talos.population import toy_population
from talos.resources import ResourceState
from talos.scheduler import Candidate
from talos.simulation import run_engagements


def trailing_pop(sat, dM_deg=-1.5):
    o = sat.orbit
    el = KeplerElements(a=o.a, e=0.0, i=o.i, raan=o.raan, argp=0.0, M=o.M + np.deg2rad(np.atleast_1d(dM_deg)))
    return DebrisPopulation(elements=el, diameter_m=0.05, areal_density=1.0, weight=1.0)


def test_rescreen_finds_opportunity_for_coorbital_object():
    sat = baseline_platform(800.0)
    pop = trailing_pop(sat)
    pl = CommitPlanner([sat], AblationParams(), horizon_s=1800.0)
    pl._rescreen(0, 0.0, pop)
    opps = pl.objs[0].opps
    assert len(opps) >= 1
    o = opps[0]
    assert o.platform == 0 and o.t_open == 0.0
    assert o.latest_select == pytest.approx(o.t_close - sat.resources.min_burst_s - sat.atp.settle_s)
    assert o.dv_pred > 0


def make_blocking_planner(sat, t_open, t_close):
    pl = CommitPlanner([sat], AblationParams())
    s = _ObjState(dv_sum=10.0, hits=1)
    s.opps = [Opportunity(0, t_open, t_close, t_close - sat.resources.min_burst_s - sat.atp.settle_s, 10.0)]
    pl.objs[7] = s
    pl.committed.add(7)
    return pl


def filler(t0=5.0, f=10.0):
    return Candidate(debris_index=1, t_start=t0, t_stop=t0 + f, dv_retrograde=5.0, n_fired=330,
                     optical_energy_j=2.5e5, dv_required=150.0)


def test_filler_blocked_by_imminent_reservation():
    sat = baseline_platform(800.0)
    rs = ResourceState(sat.resources, sat.laser)
    pl = make_blocking_planner(sat, t_open=30.0, t_close=60.0)  # latest select = 50 s
    blocked, resume = pl.filler_blocks(0, rs, filler(), sat)
    assert blocked
    assert resume >= 25.0  # t_open - settle
    assert rs.t == 0.0 and rs.heat_j == 0.0  # real state untouched


def test_filler_allowed_when_reservation_is_far():
    sat = baseline_platform(800.0)
    rs = ResourceState(sat.resources, sat.laser)
    pl = make_blocking_planner(sat, t_open=600.0, t_close=700.0)
    assert pl.filler_blocks(0, rs, filler(), sat) == (False, 0.0)


def test_zero_commitments_reproduces_greedy():
    sat = baseline_platform(800.0)
    pop = toy_population(20000, np.random.default_rng(21))
    ref = run_engagements([sat], pop, 3 * 3600.0)
    pl = CommitPlanner([sat], AblationParams(), max_committed=0)
    got = run_engagements([sat], pop, 3 * 3600.0, planner=pl)
    assert got.records() == [dict(r, committed=False) for r in ref.records()]


def test_planner_run_concentrates_on_committed_objects():
    sat = baseline_platform(800.0)
    pop = toy_population(20000, np.random.default_rng(22))
    pl = CommitPlanner([sat], AblationParams(), max_committed=50, horizon_s=3 * 3600.0)
    res = run_engagements([sat], pop, 12 * 3600.0, planner=pl)
    committed = [e for e in res.events if e.committed]
    assert len(committed) > 0
    assert res.planner_stats["rescreens"] > 0
    hits = np.bincount([e.debris_index for e in committed])
    assert hits.max() >= 2  # committed objects get revisited
