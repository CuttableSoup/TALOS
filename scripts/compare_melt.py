"""Melt gate on the design scenario: thermal coupling x policy (fragment / cap).

    python scripts/compare_melt.py [DAYS] [--out runs/compare_melt] [--only off ...]

``ablation.thermal_coupling`` is the fraction of the incident fluence left in the target as
heat; a burst whose absorbed heat reaches the melt enthalpy either counts as a fragmentation
event (``fragment``) or is cut short there on the commanded fluence (``cap``). Coupling
brackets: 0.07 (Scharring 2018, ultrashort floor), 0.15 / 0.25 (typical measured range).
``onset``: melt_margin 0.59, i.e. the limit is reaching the melting point (579 kJ/kg)
instead of fully molten (976 kJ/kg).
"""

import argparse

from talos.sweep import RunConfig, run_sweep

DESIGN = "config/design.toml"
VARIANTS = {
    "off": {},
    "frag_07": {"ablation.thermal_coupling": 0.07},
    "frag_15": {"ablation.thermal_coupling": 0.15},
    "frag_25": {"ablation.thermal_coupling": 0.25},
    "frag_25_onset": {"ablation.thermal_coupling": 0.25, "ablation.melt_margin": 0.59},
    "cap_07": {"ablation.thermal_coupling": 0.07, "ablation.melt_policy": "cap"},
    "cap_15": {"ablation.thermal_coupling": 0.15, "ablation.melt_policy": "cap"},
    "cap_25": {"ablation.thermal_coupling": 0.25, "ablation.melt_policy": "cap"},
    # the cap counts every pulse as centred; with the design's jitter the target receives at most
    # ~0.3 of that (7-day runs), so a margin of 3 puts the delivered heat at the limit
    "cap_07_x3": {"ablation.thermal_coupling": 0.07, "ablation.melt_policy": "cap", "ablation.melt_margin": 3.0},
    "cap_15_x3": {"ablation.thermal_coupling": 0.15, "ablation.melt_policy": "cap", "ablation.melt_margin": 3.0},
    "cap_25_x3": {"ablation.thermal_coupling": 0.25, "ablation.melt_policy": "cap", "ablation.melt_margin": 3.0},
}
COLS = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "removed",
        "debris_years_removed_per_yr", "melt_bursts", "capped_bursts", "ablation_fragments",
        "net_change_per_yr", "active_collisions_avoided_per_yr", "cost_per_debris_year_usd", "wall_s"]


def main(days: float, only: list[str] | None, out: str) -> None:
    names = only or list(VARIANTS)
    cfgs = [RunConfig(n, config=DESIGN, overrides={**VARIANTS[n], "run.duration_days": days}) for n in names]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    print(df[[c for c in COLS if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=7.0)
    ap.add_argument("--out", default="runs/compare_melt")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    main(a.days, a.only, a.out)
