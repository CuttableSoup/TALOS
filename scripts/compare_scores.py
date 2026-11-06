"""Scheduler comparison: removal_progress vs dv_per_energy at full density (toy pop)."""

from talos.sweep import RunConfig, run_sweep


def main(days: float, scores: list[str], mus: list[float], out: str) -> None:
    cfgs = [
        RunConfig(name=f"{score}_mu{mu}", score=score, areal_density=mu, duration_s=days * 86400.0)
        for mu in mus
        for score in scores
    ]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    cols = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
            "removed", "removed_credited", "removed_credited_per_yr", "debris_years_removed_per_yr", "wall_s"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=2.0)
    ap.add_argument("--scores", nargs="+", default=["removal_progress", "closest_to_removal", "dv_per_energy"])
    ap.add_argument("--mus", nargs="+", type=float, default=[1.0])
    ap.add_argument("--out", default="runs/compare_scores")
    a = ap.parse_args()
    main(a.days, a.scores, a.mus, a.out)
