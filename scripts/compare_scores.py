"""Scheduler comparison: removal_progress vs dv_per_energy at full density (toy pop)."""

import sys

from talos.sweep import RunConfig, run_sweep


def main(days: float) -> None:
    cfgs = [
        RunConfig(name=f"{score}_mu{mu}", score=score, areal_density=mu, duration_s=days * 86400.0)
        for mu in (0.2, 1.0)
        for score in ("removal_progress", "dv_per_energy")
    ]
    df = run_sweep(cfgs, out_dir="runs/compare_scores", n_jobs=4, threads_per_job=8)
    cols = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
            "removed", "removed_credited", "removed_credited_per_yr", "debris_years_removed_per_yr", "wall_s"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2.0)
