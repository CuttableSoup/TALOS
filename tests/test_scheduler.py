import numpy as np
import pytest

from talos.constants import R_EARTH
from talos.elements import KeplerElements, elements_to_rv
from talos.engagement import AblationParams, integrate_burst, predict_bursts, pulse_times
from talos.lifetime import apply_dv
from talos.platform import LaserConfig, baseline_platform
from talos.population import toy_population
from talos.scheduler import Candidate, dv_to_remove, evaluate_candidates, pick
from talos.screening import screen_windows


def test_dv_to_remove_circular_800km():
    from talos.constants import MU_EARTH

    r, rp = R_EARTH + 800e3, R_EARTH + 200e3
    expected = np.sqrt(MU_EARTH / r) - np.sqrt(MU_EARTH * (2 / r - 2 / (r + rp)))  # ~164 m/s
    assert dv_to_remove(r, 0.0) == pytest.approx(expected, rel=1e-12)
    assert 160 < expected < 168


def test_dv_to_remove_matches_apply_dv():
    a = R_EARTH + 800e3
    need = float(dv_to_remove(a, 0.0))
    el = KeplerElements(a=a, e=0.0, i=1.7, raan=0, argp=0, M=0)
    _, v = elements_to_rv(el)
    new = apply_dv(el, -need * v / np.linalg.norm(v))
    assert new.rp - R_EARTH == pytest.approx(200e3, abs=1e3)


def test_dv_to_remove_zero_when_already_low():
    a = R_EARTH + 500e3
    e = 1 - (R_EARTH + 150e3) / a
    assert dv_to_remove(a, e) == 0.0


def cand(j, dv, need, energy=7.5e3 * 330):
    return Candidate(j, 0.0, 10.0, dv, 330, energy, need)


def test_removal_progress_prefers_partly_lowered():
    fresh = cand(0, dv=12.0, need=150.0)
    lowered = cand(1, dv=10.0, need=20.0)
    assert pick([fresh, lowered], "removal_progress").debris_index == 1
    assert pick([fresh, lowered], "dv_per_energy").debris_index == 0


def test_removal_progress_completion_tiebreak():
    a = cand(0, dv=30.0, need=20.0)
    b = cand(1, dv=50.0, need=10.0)
    assert pick([a, b]).debris_index == 1
    assert pick([]) is None


def test_predict_bursts_matches_integrate_burst(rng):
    sat = baseline_platform(800.0)
    pop = toy_population(4000, rng)
    t = np.arange(0.0, 600.0)
    wins = screen_windows(sat, pop, t)
    assert len(wins) >= 3
    js = np.array([w.debris_index for w in wins[:8]])
    tp = pulse_times(100.0, 10.0, sat.laser.prf_hz)
    from talos.propagator import states_at

    r_s, _ = states_at(sat.orbit, tp)
    r_d, v_d = states_at(pop.elements[js], tp, pop.ballistic_coeff[js])
    active = np.ones((len(tp), len(js)), bool)
    active[200:, 0] = False  # first candidate gets a truncated burst
    p = AblationParams()
    batch = predict_bursts(r_s, r_d, v_d, active, sat.laser, pop.areal_density[js], p)
    for i in range(len(js)):
        m = active[:, i]
        one = integrate_burst(r_s[m], r_d[m, i], v_d[m, i], sat.laser, pop.areal_density[js[i]], p)
        assert batch.n_fired[i] == one.n_fired
        assert batch.dv_retrograde[i] == pytest.approx(one.dv_retrograde, rel=1e-9, abs=1e-12)
        assert np.allclose(batch.dv_vec[i], one.dv_vec, rtol=1e-9, atol=1e-12)


def test_evaluate_candidates_respects_min_burst_and_settle():
    sat = baseline_platform(800.0)
    pop = toy_population(20000, np.random.default_rng(9))
    t = np.arange(0.0, 600.0)
    wins = screen_windows(sat, pop, t)
    k = 50
    open_w = [w for w in wins if w.start <= k < w.stop]
    cands = evaluate_candidates(sat, pop, open_w, t, k, AblationParams())
    assert cands, "expected at least one viable candidate"
    for c in cands:
        assert c.t_start == pytest.approx(t[k] + sat.atp.settle_s)
        assert sat.resources.min_burst_s - 1e-9 <= c.t_stop - c.t_start <= sat.resources.burst_s + 1e-9
        assert c.dv_retrograde > 0 and c.dv_required > 0
    assert evaluate_candidates(sat, pop, open_w, t, k, AblationParams(), max_fire_s=1.0) == []


def test_closest_to_removal_concentrates():
    fresh_light = cand(0, dv=60.0, need=150.0)   # 90 left
    lowered = cand(1, dv=10.0, need=60.0)        # 50 left
    assert pick([fresh_light, lowered], "closest_to_removal").debris_index == 1
    assert pick([fresh_light, lowered], "removal_progress").debris_index == 0
