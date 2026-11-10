"""Target response (material Cm, attitude, fragmentation) and orbit knowledge."""

from dataclasses import replace

import numpy as np
import pytest

from talos.config import load_scenario, run_scenario
from talos.engagement import AblationParams, integrate_burst
from talos.knowledge import Knowledge, KnowledgeConfig
from talos.platform import LaserConfig, baseline_platform
from talos.population import toy_population
from talos.simulation import run_engagements

TOY = {"population.kind": "toy", "population.n_real": 4000, "population.min_lifetime_yr": None,
       "constellation.n_sats": 1, "run.duration_s": 4 * 3600.0}


def test_cm_scale_by_material():
    p = AblationParams(cm_scale_nk=0.0, cm_scale_ld=2.0, cm_scale_hd=0.5)
    np.testing.assert_array_equal(p.cm_scale_for([0, 1, 2, 3, 4, -1]), [0.0, 2.0, 1.0, 0.5, 1.0, 1.0])


def test_immune_targets_are_never_engaged():
    base = run_scenario(load_scenario(overrides=TOY)).summary()
    off = run_scenario(load_scenario(overrides={**TOY, "ablation.cm_scale_md": 0.0})).summary()  # toy = unknown -> md
    assert base["useful_engagements"] > 0 and off["useful_engagements"] == 0
    half = run_scenario(load_scenario(overrides={**TOY, "ablation.cm_scale_md": 0.5})).summary()
    assert half["mean_dv_retro_per_engagement"] < 0.75 * base["mean_dv_retro_per_engagement"]


def _burst_states(P=330):
    t = np.arange(P) / 33.0
    r_sat = np.zeros((P, 3))
    r_deb = np.stack([np.full(P, 250e3), 7.5e3 * t - 37e3, np.zeros(P)], 1)
    v_deb = np.tile([0.0, -7.5e3, 0.0], (P, 1))
    return r_sat, r_deb, v_deb


def test_per_burst_attitude_same_mean_more_spread():
    rs, rd, vd = _burst_states()
    las = LaserConfig()
    u_mid = rd[len(rd) // 2] / np.linalg.norm(rd[len(rd) // 2])
    out = {}
    for model in ("per_pulse", "per_burst"):
        p = AblationParams(attitude_model=model)
        rng = np.random.default_rng(3)
        out[model] = np.array([integrate_burst(rs, rd, vd, las, 1.0, p, rng=rng).dv_vec for _ in range(400)])
    a, b = out["per_pulse"], out["per_burst"]
    # along the line of sight: same expectation (E cos theta x sum |dv|)
    assert (b @ u_mid).mean() == pytest.approx((a @ u_mid).mean(), rel=0.05)
    # sideways kicks add up coherently within a burst instead of cancelling pulse to pulse
    assert np.linalg.norm(b, axis=1).mean() > 1.2 * np.linalg.norm(a, axis=1).mean()
    assert np.std(b @ np.array([0.0, 1.0, 0.0])) > 3 * np.std(a @ np.array([0.0, 1.0, 0.0]))


def test_pointing_bias_cuts_fluence():
    rs, rd, vd = _burst_states()
    las = LaserConfig()
    rng = np.random.default_rng(1)
    clean = integrate_burst(rs, rd, vd, las, 1.0, AblationParams(), jitter_rad=0.3e-6, rng=rng)
    off = integrate_burst(rs, rd, vd, las, 1.0, AblationParams(), jitter_rad=0.3e-6, rng=rng,
                          pointing_bias=np.array([3e-6, 0.0]))
    assert off.n_fired < 0.5 * clean.n_fired


def test_fragmentation_adds_objects():
    s = run_scenario(load_scenario(overrides={**TOY, "ablation.fragment_prob_per_burst": 1.0,
                                              "ablation.fragment_count_mean": 3.0,
                                              "population.areal_density_model": "sbm"})).summary()
    assert 0 < s["ablation_fragmentations"] <= s["useful_engagements"] + 5
    assert s["ablation_fragments"] > 0
    # fragments join at the next chunk start; the last chunk's are still pending at the end
    assert 4000 < s["n_samples"] <= 4000 + s["ablation_fragments"]


def test_knowledge_errors_fixed_per_chunk():
    k = Knowledge(KnowledgeConfig(od_sigma_pos_m=100.0, od_sigma_vel_m_s=1.0), np.random.default_rng(0))
    k.new_chunk(0.0)
    a = k.state_error([3, 7])
    b = k.state_error([7, 3])
    np.testing.assert_array_equal(a[0][0], b[0][1])
    k.new_chunk(600.0)
    c = k.state_error([3])
    assert not np.array_equal(a[0][0], c[0][0])
    assert np.std(np.concatenate([k.state_error(np.arange(100, 1100))[1]])) == pytest.approx(1.0, rel=0.1)


def test_knowledge_errors_change_choices_but_zero_is_identity():
    import pandas as pd

    ref = run_scenario(load_scenario(overrides=TOY))
    bad = run_scenario(load_scenario(overrides={**TOY, "knowledge.od_sigma_pos_m": 20e3,
                                                "knowledge.od_sigma_vel_m_s": 20.0}))
    assert [e.debris_index for e in ref.events] != [e.debris_index for e in bad.events]
    zero = run_scenario(load_scenario(overrides={**TOY, "knowledge.od_sigma_pos_m": 0.0}))
    pd.testing.assert_frame_equal(pd.DataFrame(zero.records()), pd.DataFrame(ref.records()))


def test_resume_exact_with_fragments_and_knowledge(tmp_path):
    import pandas as pd

    ov = {**TOY, "run.duration_s": 3.5 * 3600, "run.checkpoint_s": 3600.0, "ablation.fragment_prob_per_burst": 0.5,
          "knowledge.od_sigma_pos_m": 500.0, "knowledge.od_sigma_vel_m_s": 0.5, "knowledge.pointing_bias_rad": 0.2e-6,
          "ablation.attitude_model": "per_burst"}
    ck = tmp_path / "c.pkl"
    full = run_scenario(load_scenario(overrides=ov), checkpoint_path=ck)
    again = run_scenario(load_scenario(overrides=ov), resume_path=ck)
    pd.testing.assert_frame_equal(pd.DataFrame(again.records()), pd.DataFrame(full.records()))
    assert full.summary()["ablation_fragments"] > 0
