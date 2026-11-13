"""Melt gate: absorbed heat per burst against the target's melt enthalpy
(``ablation.thermal_coupling``, ``melt_policy``)."""

from dataclasses import replace

import numpy as np
import pytest

from talos import _kernels
from talos.config import load_scenario, run_scenario
from talos.engagement import AblationParams, fluence, integrate_burst, predict_bursts
from talos.platform import LaserConfig, baseline_platform
from talos.population import toy_population
from talos.scheduler import evaluate_candidates
from talos.screening import screen_windows

LASER = LaserConfig()
R = 300e3
PHI = float(fluence(R, LASER))  # ~11.8 kJ/m^2 per pulse
TOY = {"population.kind": "toy", "population.n_real": 4000, "population.min_lifetime_yr": None,
       "constellation.n_sats": 1, "run.duration_s": 4 * 3600.0}


def _static(n_pulses):
    """Target held at range R, flying at the laser (so the push is retrograde)."""
    r_s = np.zeros((n_pulses, 3))
    r_d = np.tile([R, 0.0, 0.0], (n_pulses, 1))
    v_d = np.tile([-7500.0, 0.0, 0.0], (n_pulses, 1))
    return r_s, r_d, v_d


def test_gate_is_off_by_default():
    p = AblationParams()
    assert not p.melt_cap and not p.melt_fragments and not p.sheds_fragments
    assert np.isinf(p.melt_fluence_sum(1.0))
    res = integrate_burst(*_static(330), LASER, 1.0, p)
    assert res.heat_frac == 0.0 and not res.capped
    assert res.fluence_sum_j_m2 == pytest.approx(330 * PHI)


def test_melt_limit_is_enthalpy_times_areal_density_over_coupling():
    p = AblationParams(thermal_coupling=0.1)
    # 976 kJ/kg x 0.5 kg/m^2 / 0.1 = 4.88 MJ/m^2 of incident fluence
    assert p.melt_fluence_sum(0.5) == pytest.approx(4.88e6)
    np.testing.assert_allclose(p.melt_fluence_sum([1.0, 2.0]), [9.76e6, 19.52e6])
    assert replace(p, melt_margin=0.59).melt_fluence_sum(1.0) == pytest.approx(0.59 * 9.76e6)


def test_readme_melt_bound():
    # Sourcing pass 2: 20 s at 33 Hz and 10 kJ/m^2 with 12.5 % coupling melts mu below ~0.85 kg/m^2
    p = AblationParams(thermal_coupling=0.125)
    phi_sum = 20 * 33 * 10e3
    assert phi_sum / p.melt_fluence_sum(0.85) == pytest.approx(1.0, abs=0.01)
    assert phi_sum < p.melt_fluence_sum(0.9)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="melt_policy"):
        AblationParams(melt_policy="ignore")
    with pytest.raises(ValueError, match="thermal_coupling"):
        AblationParams(thermal_coupling=7.0)  # percent given for a fraction


def test_fragment_policy_fires_the_whole_burst_and_reports_heat():
    p = AblationParams(thermal_coupling=0.1)
    n = 660  # 20 s
    light = integrate_burst(*_static(n), LASER, 0.5, p)
    heavy = integrate_burst(*_static(n), LASER, 5.0, p)
    assert light.n_fired == heavy.n_fired == n and not light.capped
    assert light.heat_frac == pytest.approx(0.1 * n * PHI / (0.5 * 976e3))
    assert light.heat_frac > 1.0 > heavy.heat_frac
    assert heavy.heat_frac == pytest.approx(light.heat_frac / 10)


def test_cap_policy_ends_the_burst_at_the_limit():
    p = AblationParams(thermal_coupling=0.1, melt_policy="cap")
    n, mu = 660, 0.5
    n_max = int(p.melt_fluence_sum(mu) // PHI)
    assert 0 < n_max < n
    free = integrate_burst(*_static(n), LASER, mu, AblationParams())
    res = integrate_burst(*_static(n), LASER, mu, p)
    assert res.capped and res.n_pulses == res.n_fired == n_max
    assert 1.0 - PHI / p.melt_fluence_sum(mu) < res.heat_frac <= 1.0  # within one pulse of the limit
    assert res.dv_retrograde == pytest.approx(free.dv_retrograde * n_max / n)
    assert res.optical_energy_j == pytest.approx(n_max * LASER.pulse_energy_j)
    # a heavy target is not affected
    heavy = integrate_burst(*_static(n), LASER, 5.0, p)
    assert not heavy.capped and heavy.n_fired == n


def test_cap_uses_commanded_fluence_under_jitter():
    # pointing jitter lowers the delivered fluence: the burst still ends where the on-axis
    # sum reaches the limit, with less heat actually absorbed
    p = AblationParams(thermal_coupling=0.1, melt_policy="cap")
    exact = integrate_burst(*_static(660), LASER, 0.5, p)
    jit = integrate_burst(*_static(660), LASER, 0.5, p, jitter_rad=0.3e-6, rng=np.random.default_rng(3))
    assert jit.n_pulses == exact.n_pulses and jit.capped
    assert jit.heat_frac < exact.heat_frac


def test_pulses_that_miss_the_threshold_still_heat():
    # jitter leaves some emitted pulses below the ablation threshold on target: no impulse
    # from them, but their fluence still counts as heat
    p = AblationParams(thermal_coupling=0.1)
    jit = integrate_burst(*_static(660), LASER, 0.5, p, jitter_rad=0.3e-6, rng=np.random.default_rng(3))
    assert jit.n_fired < jit.n_pulses
    assert jit.fluence_sum_j_m2 > jit.n_fired * p.threshold_fluence
    assert jit.heat_frac == pytest.approx(0.1 * jit.fluence_sum_j_m2 / (0.5 * 976e3))


def test_prediction_matches_the_capped_burst():
    p = AblationParams(thermal_coupling=0.1, melt_policy="cap")
    n = 660
    r_s, r_d, v_d = _static(n)
    mu = np.array([0.3, 0.5, 5.0])
    pred = predict_bursts(r_s, np.repeat(r_d[:, None, :], 3, axis=1), np.repeat(v_d[:, None, :], 3, axis=1),
                          np.ones((n, 3), bool), LASER, mu, p)
    for i, m in enumerate(mu):
        res = integrate_burst(r_s, r_d, v_d, LASER, float(m), p)
        assert pred.n_fired[i] == res.n_fired
        assert pred.dv_retrograde[i] == pytest.approx(res.dv_retrograde)
    assert pred.n_fired[0] < pred.n_fired[1] < pred.n_fired[2] == n


def _candidate_sets():
    pop = toy_population(20000, np.random.default_rng(21), areal_density=0.5)
    pop.areal_density[:] = np.random.default_rng(22).uniform(0.05, 5.0, len(pop))
    sat = baseline_platform(800.0)
    t = np.arange(0.0, 600.0)
    wins = screen_windows(sat, pop, t)
    out = []
    for k in sorted({w.start for w in wins})[:40]:
        open_w = [w for w in wins if w.start <= k < w.stop]
        out.append((pop, sat, open_w, t, k))
    return out


def test_scheduler_sees_the_cap():
    cap = AblationParams(thermal_coupling=0.25, melt_policy="cap")
    fewer = same = 0
    for pop, sat, open_w, t, k in _candidate_sets():
        free = {c.debris_index: c for c in evaluate_candidates(sat, pop, open_w, t, k, AblationParams())}
        for c in evaluate_candidates(sat, pop, open_w, t, k, cap):
            f = free[c.debris_index]
            assert c.n_fired <= f.n_fired and c.dv_retrograde <= f.dv_retrograde * (1 + 1e-9)
            assert c.optical_energy_j <= f.optical_energy_j
            fewer += c.n_fired < f.n_fired
            same += c.n_fired == f.n_fired
    assert fewer > 5 and same > 5  # light targets capped, heavy ones untouched


@pytest.mark.skipif(not _kernels.HAVE_NUMBA, reason="numba not installed")
def test_kernel_prediction_matches_numpy_with_cap():
    cap = AblationParams(thermal_coupling=0.25, melt_policy="cap")
    n = capped = 0
    for pop, sat, open_w, t, k in _candidate_sets():
        args = _kernels.element_args(pop.elements, pop.ballistic_coeff, pop.epoch_s)
        ref = evaluate_candidates(sat, pop, open_w, t, k, cap)
        got = evaluate_candidates(sat, pop, open_w, t, k, cap, elem_args=args)
        full = {c.debris_index: c.n_fired for c in evaluate_candidates(sat, pop, open_w, t, k, AblationParams())}
        assert [c.debris_index for c in got] == [c.debris_index for c in ref]
        for a, b in zip(got, ref):
            assert a.n_fired == b.n_fired
            assert a.dv_retrograde == pytest.approx(b.dv_retrograde, rel=1e-6)
            capped += a.n_fired < full[a.debris_index]
        n += len(ref)
    assert n > 20 and capped > 5


def test_run_fragment_policy_turns_molten_bursts_into_fragments():
    # pointing jitter spreads most of the commanded fluence off the target, so delivered heat is
    # far below the on-axis figure: an extreme coupling / light target to get molten bursts
    ovr = {**TOY, "ablation.thermal_coupling": 1.0,
           "ablation.fragment_count_mean": 3.0}
    res = run_scenario(load_scenario(overrides=ovr))
    s = res.summary()
    heat = np.array([e.heat_frac for e in res.events])
    assert s["melt_bursts"] == int((heat >= 1.0).sum()) > 0
    assert s["ablation_fragmentations"] == s["melt_bursts"]  # fragment_prob_per_burst is 0
    assert s["ablation_fragments"] > 0 and s["capped_bursts"] == 0
    # a tenth of the coupling melts fewer targets
    low = run_scenario(load_scenario(overrides={**ovr, "ablation.thermal_coupling": 0.1})).summary()
    assert low["melt_bursts"] < s["melt_bursts"]


def test_run_cap_policy_shortens_bursts_and_sheds_nothing():
    ovr = {**TOY, "ablation.thermal_coupling": 1.0}
    free = run_scenario(load_scenario(overrides=ovr | {"ablation.thermal_coupling": 0.0}))
    res = run_scenario(load_scenario(overrides=ovr | {"ablation.melt_policy": "cap"}))
    s = res.summary()
    assert s["capped_bursts"] > 0 and s["melt_bursts"] == 0 and s["ablation_fragments"] == 0
    assert max(e.heat_frac for e in res.events) <= 1.0 + 1e-9
    assert s["n_samples"] == 4000

    def mean_fire(r):
        return np.mean([e.t_stop - e.t_start for e in r.events])

    assert mean_fire(res) < 0.9 * mean_fire(free)
    assert s["mean_dv_retro_per_engagement"] < free.summary()["mean_dv_retro_per_engagement"]
