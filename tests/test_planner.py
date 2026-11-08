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
    assert o.t_open <= o.latest_select <= o.t_close - sat.resources.min_burst_s - sat.atp.settle_s
    assert o.dv_pred > 0
    # booked time is the best probed start: no worse than window open
    from talos.scheduler import evaluate_candidates
    from talos.screening import screen_windows
    t = np.arange(0.0, 1800.0, 1.0)
    w = screen_windows(sat, pop, t, candidates=np.array([0]))[0]
    c0 = evaluate_candidates(sat, pop, [w], t, w.start, AblationParams())[0]
    assert o.dv_pred >= c0.dv_retrograde - 1e-9


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
    rs.heat_j = h0 = rs.h_max - 2.6e6  # store has room for about one more 10 s burst
    assert 10.0 <= rs.max_fire_s() < 11.0
    pl = make_blocking_planner(sat, t_open=30.0, t_close=60.0)  # latest select = 50 s
    blocked, resume = pl.filler_blocks(0, rs, filler(), sat)
    assert blocked
    assert resume >= 25.0  # t_open - settle
    assert rs.t == 0.0 and rs.heat_j == h0  # real state untouched


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


def test_accept_gate_declines_weak_committed_burst():
    sat = baseline_platform(800.0)
    pl = make_blocking_planner(sat, t_open=0.0, t_close=100.0)  # booked dv_pred = 10
    pl.objs[7].opps[0] = Opportunity(0, 0.0, 100.0, 40.0, 10.0)
    weak = Candidate(7, 10.0, 20.0, 3.0, 330, 2.5e5, 150.0)
    good = Candidate(7, 10.0, 20.0, 9.0, 330, 2.5e5, 150.0)
    assert not pl.accept(0, weak, None, sat.atp.settle_s)
    assert pl.accept(0, good, None, sat.atp.settle_s)
    strong_filler = filler()
    strong_filler = Candidate(1, 10.0, 20.0, 15.0, 330, 2.5e5, 150.0)
    assert not pl.accept(0, good, strong_filler, sat.atp.settle_s)
    late = Candidate(7, 60.0, 70.0, 3.0, 330, 2.5e5, 150.0)  # past booked time: take what's left
    assert pl.accept(0, late, None, sat.atp.settle_s)
