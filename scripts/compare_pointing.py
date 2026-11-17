"""Pointing error budget on the design scenario: one term at a time (talos.knowledge).

    python scripts/compare_pointing.py [DAYS] [--out runs/compare_pointing] [--only base ...]

Each variant adds one budget term (per axis, 1 sigma) to the design's 0.3 urad jitter, or
changes the jitter itself. The mean delivered fraction under a total Gaussian sigma s is
1 / (1 + 4 s^2 / w^2) (w: 1/e^2 spot radius; s / w ~ 1 at 0.3 urad), so terms that add in
quadrature to ~0.1 urad cost little and terms of ~0.3 urad or more cost most of the removals.
Drift is a rate: 0.01 urad/s is 0.2 urad at the end of a 20 s burst. Point-ahead is a
fraction of 2 v_perp / c (~50-100 urad): 0.002 is ~0.1-0.2 urad.
"""

import argparse

import numpy as np

from talos.config import load_scenario
from talos.engagement import delivered_fraction
from talos.sweep import RunConfig, run_sweep

DESIGN = "config/design.toml"
VARIANTS = {
    "base": {},
    "jit_01": {"atp.jitter_urad": 0.1},
    "jit_02": {"atp.jitter_urad": 0.2},
    "jit_04": {"atp.jitter_urad": 0.4},
    "bore_01": {"knowledge.boresight_bias_urad": 0.1},
    "bore_02": {"knowledge.boresight_bias_urad": 0.2},
    "bore_03": {"knowledge.boresight_bias_urad": 0.3},
    "cent_01": {"knowledge.centroid_urad": 0.1},
    "cent_03": {"knowledge.centroid_urad": 0.3},
    "drift_001": {"knowledge.drift_urad_s": 0.01},
    "drift_003": {"knowledge.drift_urad_s": 0.03},
    "pa_0002": {"knowledge.point_ahead_frac": 0.002},
    "pa_0005": {"knowledge.point_ahead_frac": 0.005},
}
COLS = ["name", "delivered_frac_jitter", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement",
        "removed", "debris_years_removed_per_yr", "capped_bursts", "cost_per_net_debris_year_usd", "wall_s"]


def main(days: float, only: list[str] | None, out: str) -> None:
    names = only or list(VARIANTS)
    cfgs = [RunConfig(n, config=DESIGN, overrides={**VARIANTS[n], "run.duration_days": days}) for n in names]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    df["delivered_frac_jitter"] = [
        delivered_fraction(sc.laser, sc.atp.jitter_rad)
        for sc in (load_scenario(DESIGN, VARIANTS[k]) for k in df["name"])]
    base = df.set_index("name")["debris_years_removed_per_yr"].get("base", np.nan)
    df["dy_vs_base"] = df["debris_years_removed_per_yr"] / base
    print(df[[c for c in COLS + ["dy_vs_base"] if c in df]].to_string(index=False,
                                                                    float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=7.0)
    ap.add_argument("--out", default="runs/compare_pointing")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    main(a.days, a.only, a.out)
