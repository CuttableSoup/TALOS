"""30-day ORDEM baseline: fixed mu vs NASA standard-breakup-model mu (and sphere)."""

import argparse

from talos.sweep import RunConfig, run_sweep


def main(days: float, models: list[str], out: str) -> None:
    cfgs = [RunConfig(f"mu_{m}", overrides={"population.areal_density_model": m, "run.duration_days": days})
            for m in models]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    cols = ["name", "n_samples", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement",
            "duty_cycle_mean", "removed", "removed_credited_per_yr", "debris_years_removed_per_yr",
            "natural_decays", "net_change_per_yr", "wall_s"]
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=30.0)
    ap.add_argument("--models", nargs="+", default=["sbm"])
    ap.add_argument("--out", default="runs/compare_mu_model")
    a = ap.parse_args()
    main(a.days, a.models, a.out)
