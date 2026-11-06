import pandas as pd

from talos.sweep import RunConfig, readme_matrix, run_sweep


def tiny(name, **kw):
    return RunConfig(name=name, n_sats=1, n_real=3000, duration_s=900.0, min_lifetime_yr=None, **kw)


def test_run_sweep_parallel_writes_parquet(tmp_path):
    cfgs = [tiny("a"), tiny("b", score="dv_per_energy")]
    df = run_sweep(cfgs, tmp_path, n_jobs=2, threads_per_job=2, progress=False)
    assert list(df["name"]) == ["a", "b"]
    assert {"useful_engagements_per_hr_per_sat", "duty_cycle_mean", "wall_s"} <= set(df.columns)
    assert (tmp_path / "summary.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "summary.parquet")) == 2
    assert (tmp_path / "events" / "a.parquet").exists()


def test_parallel_equals_serial(tmp_path):
    cfgs = [tiny("x", seed=3), tiny("y", seed=4)]
    par = run_sweep(cfgs, n_jobs=2, threads_per_job=1, progress=False)
    ser = run_sweep(cfgs, n_jobs=1, threads_per_job=1, progress=False)
    cols = ["useful_engagements", "mean_dv_retro_per_engagement", "removed", "duty_cycle_mean"]
    pd.testing.assert_frame_equal(par[cols], ser[cols])


def test_readme_matrix_size_and_unique_names():
    m = readme_matrix()
    assert len(m) == 5 * 4 * 2 * 3 * 3
    assert len({c.name for c in m}) == len(m)
