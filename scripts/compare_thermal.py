"""Thermal design levers on the TOML baseline (ORDEM + SBM, sun + thermal model):
radiator area, PCM mass, burst length, wall-plug efficiency.

    python scripts/compare_thermal.py [DAYS] [--out runs/compare_thermal] [--only base rad40 ...]
"""

import argparse

from talos.sweep import RunConfig, run_sweep

VARIANTS = {
    "base": {},
    "rad40": {"resources.radiator_area_m2": 40.0},
    "rad60": {"resources.radiator_area_m2": 60.0},
    "pcm75": {"resources.pcm_mass_kg": 75.0},
    "burst5": {"resources.burst_s": 5.0},
    "burst20": {"resources.burst_s": 20.0},
    "wallplug15": {"resources.wallplug_eff": 0.15},
    "rad40_pcm75": {"resources.radiator_area_m2": 40.0, "resources.pcm_mass_kg": 75.0},
    "burst30": {"resources.burst_s": 30.0},
    "rad40_burst20": {"resources.radiator_area_m2": 40.0, "resources.burst_s": 20.0},
    "rad60_burst20": {"resources.radiator_area_m2": 60.0, "resources.burst_s": 20.0},
    "burst20_wallplug15": {"resources.burst_s": 20.0, "resources.wallplug_eff": 0.15},
}
COLS = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
        "saturation", "removed", "removed_credited_per_yr", "debris_years_removed_per_yr",
        "heater_kwh_per_day", "brownout_s", "battery_efc_per_sat_per_yr", "battery_dod_mean",
        "risk_starlink_delta_obj_yr_per_yr", "wall_s"]


def main(days: float, only: list[str] | None, out: str) -> None:
    names = only or list(VARIANTS)
    cfgs = [RunConfig(n, overrides={**VARIANTS[n], "run.duration_days": days}) for n in names]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    print(df[[c for c in COLS if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=30.0)
    ap.add_argument("--out", default="runs/compare_thermal")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    main(a.days, a.only, a.out)
