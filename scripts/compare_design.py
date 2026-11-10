"""Burst length x energy storage x thermal store x radiator, with mass / life / cost.

    python scripts/compare_design.py [DAYS] [--out runs/compare_design] [--only base ...]

Removal counts from a short run are still on their ramp: compare variants at
equal run length; cost per removal uses each run's own annualised rate.
"""

import argparse

from talos.platform import STORAGE_PRESETS
from talos.sweep import RunConfig, run_sweep


def storage(kind: str, kwh: float) -> dict:
    return {**{f"resources.{k}": v for k, v in STORAGE_PRESETS[kind].items()},
            "resources.battery_capacity_kwh": kwh}


R60 = {"resources.radiator_area_m2": 60.0}
VARIANTS = {
    "base": {},
    "b20": {"resources.burst_s": 20.0},
    "bmax": {"resources.burst_s": 120.0},  # capped at what the store / battery allow (~25 s)
    "bmax_ready20": {"resources.burst_s": 120.0, "resources.ready_burst_s": 20.0},
    "pcm75_bmax": {"resources.pcm_mass_kg": 75.0, "resources.burst_s": 120.0},
    "pcm75_li4_bmax": {"resources.pcm_mass_kg": 75.0, "resources.burst_s": 120.0, **storage("li_ion", 4.0)},
    "sc2_b20": {"resources.burst_s": 20.0, **storage("supercap", 2.0)},
    "lto4_b20": {"resources.burst_s": 20.0, **storage("lto", 4.0)},
    "r60_b20": {**R60, "resources.burst_s": 20.0},
    "r60_sc2_b20": {**R60, "resources.burst_s": 20.0, **storage("supercap", 2.0)},
    "r60_lto4_b20": {**R60, "resources.burst_s": 20.0, **storage("lto", 4.0)},
    "r60_pcm75_lto4_bmax": {**R60, "resources.pcm_mass_kg": 75.0, "resources.burst_s": 120.0,
                            **storage("lto", 4.0)},
}
COLS = ["name", "ready_burst_s", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement",
        "duty_cycle_mean", "removed", "battery_dod_mean", "storage_life_yr", "mass_total_kg",
        "mission_life_yr", "life_limit", "cost_musd", "cost_per_removal_kusd", "wall_s"]


def main(days: float, only: list[str] | None, out: str) -> None:
    names = only or list(VARIANTS)
    cfgs = [RunConfig(n, overrides={**VARIANTS[n], "run.duration_days": days}) for n in names]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    print(df[[c for c in COLS if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=21.0)
    ap.add_argument("--out", default="runs/compare_design")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    main(a.days, a.only, a.out)
