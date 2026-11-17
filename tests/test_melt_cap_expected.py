"""Jitter-aware melt cap (``ablation.melt_cap_basis = "expected"``) and the fragment
debris-year charge (``debris_years_net``, ``cost_per_net_debris_year_usd``)."""

import numpy as np
import pytest

from talos import _kernels, cost
from talos.config import load_scenario, run_scenario
from talos.engagement import AblationParams, delivered_fraction, fluence, integrate_burst, predict_bursts
from talos.platform import LaserConfig

from test_melt_gate import TOY, _candidate_sets, _static

LASER = LaserConfig()
R = 300e3
PHI = float(fluence(R, LASER))
JIT = 0.3e-6
CAP = AblationParams(thermal_coupling=0.1, melt_policy="cap", melt_cap_basis="expected")


def test_delivered_fraction_matches_sampled_jitter():
    f = delivered_fraction(LASER, JIT)
    assert 0.15 < f < 0.25  # design jitter: about a fifth of the on-axis fluence
    assert delivered_fraction(LASER, 0.0) == 1.0
    res = integrate_burst(*_static(20000), LASER, 1.0, AblationParams(), jitter_rad=JIT,
                          rng=np.random.default_rng(5))
    assert res.fluence_sum_j_m2 / (20000 * PHI) == pytest.approx(f, rel=0.03)


def test_basis_is_validated_and_commanded_by_default():
    assert AblationParams().melt_cap_basis == "commanded"
    assert load_scenario(overrides=TOY).ablation.melt_cap_basis == "commanded"
    with pytest.raises(ValueError, match="melt_cap_basis"):
        AblationParams(melt_cap_basis="delivered")
    p = AblationParams(thermal_coupling=0.1)
    assert p.melt_cap_sum(0.5, LASER, JIT) == p.melt_fluence_sum(0.5)
    assert CAP.melt_cap_sum(0.5, LASER, JIT) == pytest.approx(
        CAP.melt_fluence_sum(0.5) / delivered_fraction(LASER, JIT))


def test_expected_cap_lands_near_the_melt_limit():
    mu, n = 0.5, 3300  # 100 s at 33 Hz: long enough for the expected cap to bind
    n_cap = int(CAP.melt_cap_sum(mu, LASER, JIT) // PHI)
    assert n_cap < n
    heat = []
    for seed in range(8):
        res = integrate_burst(*_static(n), LASER, mu, CAP, jitter_rad=JIT, rng=np.random.default_rng(seed))
        assert res.capped and res.n_pulses == n_cap
        heat.append(res.heat_frac)
    # delivered heat scatters about the limit (vs <= 0.2 of it on the commanded basis)
    assert np.mean(heat) == pytest.approx(1.0, abs=0.05)
    commanded = AblationParams(thermal_coupling=0.1, melt_policy="cap")
    res = integrate_burst(*_static(n), LASER, mu, commanded, jitter_rad=JIT, rng=np.random.default_rng(0))
    assert res.heat_frac < 0.3


def test_deterministic_burst_uses_the_platform_jitter_for_the_cap():
    mu, n = 0.5, 3300
    sto = integrate_burst(*_static(n), LASER, mu, CAP, jitter_rad=JIT, rng=np.random.default_rng(1))
    det = integrate_burst(*_static(n), LASER, mu, CAP, cap_jitter_rad=JIT)
    assert det.n_pulses == sto.n_pulses
    # no jitter given at all: the cap falls back to the commanded sum
    assert integrate_burst(*_static(n), LASER, mu, CAP).n_pulses == int(CAP.melt_fluence_sum(mu) // PHI)


def test_prediction_matches_the_expected_cap():
    n = 3300
    r_s, r_d, v_d = _static(n)
    mu = np.array([0.2, 0.5, 30.0])
    pred = predict_bursts(r_s, np.repeat(r_d[:, None, :], 3, axis=1), np.repeat(v_d[:, None, :], 3, axis=1),
                          np.ones((n, 3), bool), LASER, mu, CAP, cap_jitter_rad=JIT)
    for i, m in enumerate(mu):
        res = integrate_burst(r_s, r_d, v_d, LASER, float(m), CAP, cap_jitter_rad=JIT)
        assert pred.n_fired[i] == res.n_fired
        assert pred.dv_retrograde[i] == pytest.approx(res.dv_retrograde)
    assert pred.n_fired[0] < pred.n_fired[1] < pred.n_fired[2] == n


@pytest.mark.skipif(not _kernels.HAVE_NUMBA, reason="numba not installed")
def test_kernel_prediction_matches_numpy_with_expected_cap():
    from talos.scheduler import evaluate_candidates

    cap = AblationParams(thermal_coupling=1.0, melt_policy="cap", melt_cap_basis="expected")
    commanded = AblationParams(thermal_coupling=1.0, melt_policy="cap")
    n = looser = 0
    for pop, sat, open_w, t, k in _candidate_sets():
        args = _kernels.element_args(pop.elements, pop.ballistic_coeff, pop.epoch_s)
        ref = evaluate_candidates(sat, pop, open_w, t, k, cap)
        got = evaluate_candidates(sat, pop, open_w, t, k, cap, elem_args=args)
        tight = {c.debris_index: c.n_fired for c in evaluate_candidates(sat, pop, open_w, t, k, commanded)}
        assert [c.debris_index for c in got] == [c.debris_index for c in ref]
        for a, b in zip(got, ref):
            assert a.n_fired == b.n_fired
            assert a.dv_retrograde == pytest.approx(b.dv_retrograde, rel=1e-6)
            assert a.n_fired >= tight.get(a.debris_index, 0)
            looser += a.n_fired > tight.get(a.debris_index, 0)
        n += len(ref)
    assert n > 20 and looser > 5


def test_run_expected_cap_fires_longer_than_commanded():
    ovr = {**TOY, "ablation.thermal_coupling": 1.0, "ablation.melt_policy": "cap"}
    com = run_scenario(load_scenario(overrides=ovr))
    exp = run_scenario(load_scenario(overrides=ovr | {"ablation.melt_cap_basis": "expected"}))

    def mean_fire(r):
        return np.mean([e.t_stop - e.t_start for e in r.events])

    assert mean_fire(exp) > mean_fire(com)
    assert exp.summary()["melt_bursts"] == 0  # the cap policy never sheds fragments


def test_fragment_charge():
    ovr = {**TOY, "ablation.fragment_prob_per_burst": 1.0, "ablation.fragment_count_mean": 3.0}
    sc = load_scenario(overrides=ovr)
    s = run_scenario(sc).summary()
    assert s["ablation_fragments"] > 0 and s["debris_years_fragments"] > 0
    assert s["debris_years_net"] == pytest.approx(s["debris_years_removed"] - s["debris_years_fragments"])
    c = cost.evaluate(sc, s)
    assert c["cost_per_net_debris_year_usd"] > c["cost_per_debris_year_usd"] or (
        c["debris_years_net_mission"] <= 0 and np.isinf(c["cost_per_net_debris_year_usd"]))
    # no fragments: no charge columns, net cost = headline
    sc0 = load_scenario(overrides=TOY)
    s0 = run_scenario(sc0).summary()
    assert "debris_years_fragments" not in s0
    c0 = cost.evaluate(sc0, s0)
    assert c0["cost_per_net_debris_year_usd"] == c0["cost_per_debris_year_usd"]
