"""Latin-hypercube uncertainty sweep over the README placeholder ranges (roadmap 7).

    python scripts/run_uncertainty.py N DAYS [--out runs/uncertainty] [--jobs 4] [--toy]

Prints 5/50/95 % bands of the headline metrics and the Spearman rank correlation of
each input with each metric. ``--toy`` uses the toy population (no ORDEM table needed).
"""

import argparse

from talos.sweep import RunConfig, lhs_matrix, run_sweep, uncertainty_bands

METRICS = ["useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
           "removed_per_yr", "debris_years_removed_per_yr"]


def main(n: int, days: float, out: str, jobs: int, toy: bool, seed: int) -> None:
    ovr = {"run.duration_days": days}
    if toy:
        ovr |= {"population.kind": "toy", "population.areal_density_model": "sbm"}
    cfgs = lhs_matrix(n, base=RunConfig("lhs", overrides=ovr), seed=seed)
    df = run_sweep(cfgs, out_dir=out, n_jobs=jobs, threads_per_job=max(1, 32 // jobs))
    bands, corr = uncertainty_bands(df, METRICS)
    print(bands.to_string(float_format=lambda x: f"{x:,.3f}"))
    print()
    print(corr.to_string(float_format=lambda x: f"{x:+.2f}"))
    bands.to_csv(f"{out}/bands.csv")
    corr.to_csv(f"{out}/rank_corr.csv")


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=32)
    ap.add_argument("days", type=float, nargs="?", default=1.0)
    ap.add_argument("--out", default="runs/uncertainty")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.days, a.out, a.jobs, a.toy, a.seed)
