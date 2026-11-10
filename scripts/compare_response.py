"""Target response and orbit knowledge on the TOML baseline (ORDEM + SBM).

    python scripts/compare_response.py [DAYS] [--out runs/compare_response] [--only base ...]
"""

import argparse

from talos.sweep import RunConfig, run_sweep

VARIANTS = {
    "base": {},
    "nak_immune": {"ablation.cm_scale_nk": 0.0},
    "materials": {"ablation.cm_scale_ld": 1.5, "ablation.cm_scale_md": 0.7, "ablation.cm_scale_hd": 0.4},
    "per_burst": {"ablation.attitude_model": "per_burst"},
    "frag_1pct": {"ablation.fragment_prob_per_burst": 0.01},
    "frag_10pct": {"ablation.fragment_prob_per_burst": 0.10},
    "od_angles_only": {"knowledge.od_sigma_pos_m": 1e3, "knowledge.od_sigma_vel_m_s": 1.0},
    "od_poor": {"knowledge.od_sigma_pos_m": 10e3, "knowledge.od_sigma_vel_m_s": 10.0},
    "bias_0.3urad": {"knowledge.pointing_bias_rad": 0.3e-6},
    "bias_1urad": {"knowledge.pointing_bias_rad": 1.0e-6},
}
COLS = ["name", "useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "removed",
        "removed_credited_per_yr", "ablation_fragments", "net_change_per_yr", "active_collisions_avoided_per_yr",
        "cost_per_removal_kusd", "wall_s"]


def main(days: float, only: list[str] | None, out: str) -> None:
    names = only or list(VARIANTS)
    cfgs = [RunConfig(n, overrides={**VARIANTS[n], "run.duration_days": days}) for n in names]
    n = min(len(cfgs), 4)
    df = run_sweep(cfgs, out_dir=out, n_jobs=n, threads_per_job=32 // n)
    print(df[[c for c in COLS if c in df]].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=14.0)
    ap.add_argument("--out", default="runs/compare_response")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    main(a.days, a.only, a.out)
