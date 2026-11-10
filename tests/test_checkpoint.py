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


def test_fresh_platforms_restart_keeps_the_sky_and_counts_only_new_work(tmp_path):
    import numpy as np

    from talos.simulation import load_checkpoint

    ck = tmp_path / "c.pkl"
    run_scenario(sc(2.5), checkpoint_path=ck)  # checkpoint at 2 h
    st = load_checkpoint(ck)
    c0, done = st["c0"], st["removed"].copy()
    hw = {"laser.burst_power_kw": 50.0, "constellation.n_sats": 3, "resources.radiator_area_m2": 40.0}
    res = run_scenario(sc(4.0, **hw), resume_path=ck, fresh_platforms=True)
    s = res.summary()
    assert res.events and min(e.t_start for e in res.events) >= c0
    assert s["n_platforms"] == 3 and s["sim_hours"] == pytest.approx((4 * 3600.0 - c0) / 3600.0)
    new = res.removed[:len(done)] & ~done
    assert s["removed"] == int(new.sum()) + int(res.removed[len(done):].sum())
    assert s["pop_initial"] == int((~(st["removed"] | st["decayed"])).sum())
    # objects removed before the restart are never engaged again
    assert not any(done[e.debris_index] for e in res.events if e.debris_index < len(done))
    # lifetimes are measured from the restart state, not from t = 0
    j = res.events[0].debris_index
    from talos.lifetime import orbital_lifetime
    from talos.render import _with_atmosphere
    p = st["pop"]
    want = _with_atmosphere(sc(4.0), lambda: float(orbital_lifetime(
        p.elements.a[j:j + 1], p.elements.e[j:j + 1], p.ballistic_coeff[j:j + 1], t0_s=p.epoch_s[j:j + 1])[0]))
    assert res.lifetime_initial_s[j] == pytest.approx(want, rel=1e-9)


def test_fresh_platforms_still_rejects_population_changes(tmp_path):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(1.5), checkpoint_path=ck)
    with pytest.raises(ValueError, match="dynamics.breakup_fragments"):
        run_scenario(sc(3.0, **{"dynamics.breakup_fragments": 10}), resume_path=ck, fresh_platforms=True)
    with pytest.raises(ValueError, match="absolute end"):
        run_scenario(sc(0.5), resume_path=ck, fresh_platforms=True)
