"""30-day comparison of the receding-horizon commitment planner at several commit-set sizes."""

import argparse

from talos.sweep import RunConfig, run_sweep


def main(days: float, sizes: list[int], horizon_h: float, out: str) -> None:
    cfgs = [
        RunConfig(name=f"planner_M{m}", planner=True, max_committed=m, horizon_s=horizon_h * 3600.0,
                  duration_s=days * 86400.0)
        for m in sizes
    ]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    cols = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "removed",
            "removed_credited_per_yr", "debris_years_removed_per_yr", "committed_engagement_frac",
            "planner_blocked_fillers", "planner_rescreens", "wall_s"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=30.0)
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 200, 800])
    ap.add_argument("--horizon-h", type=float, default=6.0)
    ap.add_argument("--out", default="runs/compare_planner")
    a = ap.parse_args()
    main(a.days, a.sizes, a.horizon_h, a.out)
