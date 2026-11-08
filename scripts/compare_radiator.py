"""Radiator-area comparison on the TOML baseline (sun + thermal model)."""

import argparse

from talos.sweep import RunConfig, run_sweep


def main(days: float, areas: list[float], config: str | None, out: str) -> None:
    cfgs = [RunConfig(f"rad{a:g}", config=config,
                      overrides={"resources.radiator_area_m2": a, "run.duration_days": days})
            for a in areas]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    cols = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
            "saturation", "removed", "removed_credited_per_yr", "debris_years_removed_per_yr",
            "eclipse_fraction", "heater_kwh_per_day", "brownout_s", "wall_s"]
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=30.0)
    ap.add_argument("--areas", nargs="+", type=float, default=[20.0, 40.0, 60.0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/compare_radiator")
    a = ap.parse_args()
    main(a.days, a.areas, a.config, a.out)
