"""Full-state checkpoints: a resumed (or extended) run reproduces the uninterrupted one."""

import pandas as pd
import pytest

from talos.config import load_scenario, run_scenario

BASE = {"population.kind": "toy", "population.n_real": 3000, "population.min_lifetime_yr": None,
        "population.areal_density_model": "sbm", "constellation.n_sats": 2,
        "dynamics.update_days": 0.03, "dynamics.breakup_rate_per_yr": 30000.0, "dynamics.breakup_fragments": 50,
        "run.checkpoint_s": 3600.0}


def sc(hours, **kw):
    return load_scenario(overrides={**BASE, "run.duration_s": hours * 3600.0, **kw})


def frame(res):
    return pd.DataFrame(res.records())


def test_resume_reproduces_uninterrupted_run(tmp_path):
    ck = tmp_path / "c.pkl"
    full = run_scenario(sc(3.5), checkpoint_path=ck)  # last checkpoint at 3 h
    assert ck.exists()
    again = run_scenario(sc(3.5), resume_path=ck)
    pd.testing.assert_frame_equal(frame(again), frame(full))
    assert again.summary() == pytest.approx(full.summary(), rel=1e-12, nan_ok=True)
    assert full.summary()["breakups"] > 0  # dynamics state (rng, next update) was exercised


def test_resume_extends_a_shorter_run(tmp_path):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(1.5), checkpoint_path=ck)  # checkpoint at 1 h
    ext = run_scenario(sc(3.0), resume_path=ck)
    straight = run_scenario(sc(3.0))
    pd.testing.assert_frame_equal(frame(ext), frame(straight))
    assert ext.summary() == pytest.approx(straight.summary(), rel=1e-12, nan_ok=True)


def test_resume_rejects_a_different_scenario(tmp_path):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(1.5), checkpoint_path=ck)
    with pytest.raises(ValueError, match="laser.burst_power_w"):
        run_scenario(sc(3.0, **{"laser.burst_power_kw": 10.0}), resume_path=ck)
