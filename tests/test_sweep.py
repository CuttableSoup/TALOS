import pandas as pd

from talos.sweep import RunConfig, readme_matrix, run_sweep


def tiny(name, **kw):
    return RunConfig(name, overrides={"constellation.n_sats": 1, "population.kind": "toy",
                                      "population.n_real": 3000,
                                      "run.duration_s": 900.0, "population.min_lifetime_yr": None, **kw})


def test_run_sweep_parallel_writes_parquet(tmp_path):
    cfgs = [tiny("a"), tiny("b", **{"run.score": "dv_per_energy"})]
    df = run_sweep(cfgs, tmp_path, n_jobs=2, threads_per_job=2, progress=False)
    assert list(df["name"]) == ["a", "b"]
    assert {"useful_engagements_per_hr_per_sat", "duty_cycle_mean", "wall_s"} <= set(df.columns)
    assert (tmp_path / "summary.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "summary.parquet")) == 2
    assert (tmp_path / "events" / "a.parquet").exists()


def test_parallel_equals_serial(tmp_path):
    cfgs = [tiny("x", **{"run.seed": 3}), tiny("y", **{"run.seed": 4})]
    par = run_sweep(cfgs, n_jobs=2, threads_per_job=1, progress=False)
    ser = run_sweep(cfgs, n_jobs=1, threads_per_job=1, progress=False)
    cols = ["useful_engagements", "mean_dv_retro_per_engagement", "removed", "duty_cycle_mean"]
    pd.testing.assert_frame_equal(par[cols], ser[cols])


def test_readme_matrix_size_and_unique_names():
    m = readme_matrix()
    assert len(m) == 5 * 4 * 2 * 3 * 3
    assert len({c.name for c in m}) == len(m)


def test_readme_matrix_mu_axis_forces_fixed_model():
    m = readme_matrix()
    assert all(c.overrides["population.areal_density_model"] == "fixed" for c in m)
    sc = m[0].scenario()
    assert sc.population.areal_density_model == "fixed" and sc.population.areal_density == 0.2


def test_run_writes_progress_log_and_drops_checkpoint(tmp_path):
    run_sweep([tiny("p")], tmp_path, n_jobs=1, threads_per_job=1, progress=False)
    lines = (tmp_path / "logs" / "p.log").read_text().splitlines()
    assert lines[0].startswith("# p:") and len(lines) == 2  # 900 s run: only the end-of-run line
    assert float(lines[1].split()[0]) == 0.01  # 900 s = 0.0104 d, printed to 2 dp
    assert (tmp_path / "events" / "p.parquet").exists()
    assert not (tmp_path / "events" / "p.partial.parquet").exists()


def test_run_monitor_logs_and_checkpoints(tmp_path):
    import numpy as np

    from talos.simulation import EngagementEvent
    from talos.sweep import RunMonitor

    mon = RunMonitor("m", tmp_path, duration_s=4000.0, log_every_s=1000.0, checkpoint_s=2000.0)
    ev = [EngagementEvent("s0", 1, 10.0, 20.0, 5, 5, 0.1, 0.1, 1.0, 7e5, 6.9e5, False)]
    rem, dec = np.array([False, True]), np.array([False, False])
    for t in (600.0, 1200.0, 1800.0, 2400.0):
        mon(t, ev, rem, dec)
    lines = (tmp_path / "logs" / "m.log").read_text().splitlines()[1:]
    assert len(lines) == 2  # at 1200 and 2400 s
    assert lines[-1].split()[2:6] == ["1", "1", "1", "1"]  # events, useful, removed, live
    part = pd.read_parquet(tmp_path / "events" / "m.partial.parquet")
    assert len(part) == 1 and part["run"].iloc[0] == "m"


def test_lhs_matrix_stratifies_each_axis():
    import numpy as np

    from talos.sweep import lhs_matrix

    m = lhs_matrix(20, {"laser.eta2": (0.3, 0.6), "ablation.cm_n_per_mw": (25.0, 100.0, "log")}, seed=1)
    assert len(m) == 20 and len({c.name for c in m}) == 20 and len({c.overrides["run.seed"] for c in m}) == 20
    eta = np.array([c.overrides["laser.eta2"] for c in m])
    cm = np.array([c.overrides["ablation.cm_n_per_mw"] for c in m])
    assert sorted(np.floor((eta - 0.3) / 0.3 * 20).astype(int)) == list(range(20))  # one per stratum
    assert sorted(np.floor(np.log(cm / 25) / np.log(4) * 20).astype(int)) == list(range(20))
    assert m[0].scenario().laser.eta2 == eta[0]


def test_uncertainty_bands_rank_correlation():
    import numpy as np

    from talos.sweep import uncertainty_bands

    x = np.linspace(0, 1, 50)
    df = pd.DataFrame({"ovr.a": x, "ovr.b": np.random.default_rng(0).permutation(x), "ovr.run.seed": range(50),
                       "m": 3 * x + 1})
    bands, corr = uncertainty_bands(df, ["m"])
    assert bands.loc["m", "q50"] == np.median(3 * x + 1)
    assert corr.loc["a", "m"] == 1.0 and abs(corr.loc["b", "m"]) < 0.5 and "run.seed" not in corr.index
