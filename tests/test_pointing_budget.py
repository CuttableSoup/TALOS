"""Pointing error budget (talos.knowledge): boresight, centroid, drift, point-ahead."""

import numpy as np
import pytest

from talos.config import load_scenario, run_scenario
from talos.engagement import AblationParams, delivered_fraction, fluence, integrate_burst
from talos.knowledge import C_LIGHT, Knowledge, KnowledgeConfig
from talos.platform import LaserConfig

LASER = LaserConfig()
R = 300e3
PHI = float(fluence(R, LASER))
TOY = {"population.kind": "toy", "population.n_real": 4000, "population.min_lifetime_yr": None,
       "constellation.n_sats": 1, "run.duration_s": 4 * 3600.0}


def _static(n):
    r_s = np.zeros((n, 3))
    r_d = np.tile([R, 0.0, 0.0], (n, 1))
    v_d = np.tile([-7500.0, 0.0, 0.0], (n, 1))
    return r_s, r_d, v_d


def _know(seed=0, **kw):
    return Knowledge(KnowledgeConfig(**kw), np.random.default_rng(seed), seed=seed)


def test_off_by_default():
    k = _know()
    assert not k.cfg.pointing_active and not k.cfg.active
    tp = np.arange(10) / 33.0
    assert k.pointing_offsets("s0", tp, *_static(10)[:2]) is None


def test_bias_only_draws_as_before():
    # runs that use only the pre-budget pointing_bias_rad see the same draws
    a, b = _know(3, pointing_bias_rad=0.2e-6), _know(3, pointing_bias_rad=0.2e-6)
    tp = np.arange(20) / 33.0
    for _ in range(5):
        old = a.pointing_bias()
        new = b.pointing_offsets("s0", tp, *_static(20)[:2])
        assert new.shape == (20, 2)
        np.testing.assert_array_equal(new, np.broadcast_to(old, (20, 2)))


def test_boresight_is_fixed_per_platform():
    k = _know(1, boresight_bias_rad=0.1e-6)
    tp = np.arange(5) / 33.0
    s = _static(5)[:2]
    a1, a2 = k.pointing_offsets("s0", tp, *s), k.pointing_offsets("s0", tp, *s)
    np.testing.assert_array_equal(a1, a2)  # no per-burst draw
    assert not np.allclose(k.boresight("s0"), k.boresight("s1"))
    np.testing.assert_array_equal(_know(1, boresight_bias_rad=0.1e-6).boresight("s0"), k.boresight("s0"))
    assert not np.allclose(_know(2, boresight_bias_rad=0.1e-6).boresight("s0"), k.boresight("s0"))
    sd = np.std([_know(1, boresight_bias_rad=0.1e-6).boresight(f"p{i}") for i in range(4000)])
    assert sd == pytest.approx(0.1e-6, rel=0.05)


def test_centroid_scales_with_photon_noise():
    tp = np.arange(3) / 33.0
    s = _static(3)[:2]
    k = _know(4, centroid_rad=0.1e-6, centroid_ref_mag=14.0)
    ref = np.std([k.pointing_offsets("s", tp, *s, mag=14.0)[0] for _ in range(4000)])
    faint = np.std([k.pointing_offsets("s", tp, *s, mag=19.0)[0] for _ in range(4000)])
    assert ref == pytest.approx(0.1e-6, rel=0.05)
    assert faint / ref == pytest.approx(10.0, rel=0.08)  # 5 mag = 100x fewer photons
    assert np.std([k.pointing_offsets("s", tp, *s)[0] for _ in range(4000)]) == pytest.approx(0.1e-6, rel=0.05)


def test_drift_grows_linearly_over_the_burst():
    k = _know(5, drift_rad_s=0.01e-6)
    tp = 100.0 + np.arange(661) / 33.0  # 20 s
    off = k.pointing_offsets("s", tp, *_static(661)[:2])
    np.testing.assert_allclose(off[0], 0.0)
    np.testing.assert_allclose(off[-1], off[330] * (tp[-1] - tp[0]) / (tp[330] - tp[0]))


def test_point_ahead_error_follows_the_apparent_motion():
    # target crossing at 7.5 km/s transverse, 300 km away: point-ahead 2 v / c = 50 urad
    n = 66
    tp = np.arange(n) / 33.0
    r_s = np.zeros((n, 3))
    r_d = np.column_stack([np.full(n, R), 7500.0 * tp, np.zeros(n)])
    k = _know(6, point_ahead_frac=0.01)
    off = k.pointing_offsets("s", tp, r_s, r_d)
    v_perp = 7500.0 * R / np.linalg.norm(r_d, axis=1)  # velocity across the line of sight
    np.testing.assert_allclose(off[:, 0], 0.01 * 2 * v_perp / C_LIGHT, rtol=1e-3)
    assert off[n // 2, 0] == pytest.approx(0.5e-6, rel=0.01)
    np.testing.assert_array_equal(off[:, 1], 0.0)


def test_gaussian_terms_add_in_quadrature_for_the_mean():
    jit, bor = 0.2e-6, 0.2e-6
    n, sums = 330, []
    rng = np.random.default_rng(7)
    for i in range(600):
        b = bor * rng.standard_normal(2)
        res = integrate_burst(*_static(n), LASER, 1.0, AblationParams(), jitter_rad=jit, rng=rng, pointing_bias=b)
        sums.append(res.fluence_sum_j_m2 / (n * PHI))
    assert np.mean(sums) == pytest.approx(delivered_fraction(LASER, np.hypot(jit, bor)), rel=0.05)


def test_per_pulse_offsets_are_used_and_cut_with_a_capped_burst():
    n = 660
    big = np.zeros((n, 2))
    big[n // 2:, 0] = 5e-6  # second half of the burst misses
    res = integrate_burst(*_static(n), LASER, 1.0, AblationParams(), jitter_rad=1e-12,
                          rng=np.random.default_rng(0), pointing_bias=big)
    assert res.n_fired == n // 2
    cap = AblationParams(thermal_coupling=0.1, melt_policy="cap")
    c = integrate_burst(*_static(n), LASER, 0.3, cap, jitter_rad=1e-12, rng=np.random.default_rng(0),
                        pointing_bias=big)
    assert c.capped and c.n_pulses < n // 2 and c.n_fired == c.n_pulses


def test_run_with_the_budget_on_and_resume(tmp_path):
    import pandas as pd

    ref = run_scenario(load_scenario(overrides=TOY)).summary()
    ov = {**TOY, "knowledge.boresight_bias_urad": 0.2, "knowledge.centroid_urad": 0.1,
          "knowledge.drift_urad_s": 0.01, "knowledge.point_ahead_frac": 0.002,
          "run.duration_s": 3.5 * 3600, "run.checkpoint_s": 3600.0}
    ck = tmp_path / "c.pkl"
    full = run_scenario(load_scenario(overrides=ov), checkpoint_path=ck)
    assert full.summary()["mean_dv_retro_per_engagement"] < ref["mean_dv_retro_per_engagement"]
    again = run_scenario(load_scenario(overrides=ov), resume_path=ck)
    pd.testing.assert_frame_equal(pd.DataFrame(again.records()), pd.DataFrame(full.records()))
