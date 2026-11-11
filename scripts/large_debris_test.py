"""Large-debris test: fly a design against the large, heavy objects every mission ends up with.

    python scripts/large_debris_test.py [DAYS] --out runs/large_debris
        [--config config/design.toml config/design_mc.toml ...] [--designs-csv f.csv]
        [--count 0] [--min-diameter-cm 9] [--select random] [--mixed] [--seeds 0 1 2]
        [--min-impulse-frac 0.15]
        [--ckpt runs/design_2yr/checkpoints/design.pkl] [--ckpt-day 720]

Each design restarts with fresh platforms over the depleted day-720 sky of the 2-year design
run (the light tail is gone) and runs a focus campaign (talos.campaign) on the live objects
>= ``--min-diameter-cm`` (all 318 at 9 cm by default, so the set is the same for every design;
``--count N`` draws N of them instead, ``--seeds`` for different draws): exclusive by default (fire only at the
focus set: what the design can do to large debris when it devotes itself to them), or
``--mixed`` (focus first when its burst geometry is >= 0.5 x the best, otherwise normal
targets; rarely picks a focus object, see README).
Wall time: ~5 min per design for 30 days at 8 threads (screening covers the focus set only).

Reports per design: focus objects removed, half-way, mean progress (fraction of the dv
to the 200 km removal perigee delivered), dv per burst and in total, and the
cost per debris-year of the run. Designs come from scenario TOMLs or a CSV of lever
overrides (as written for scripts/mc_design.py: one column per dotted key, optional "tag").
"""

import argparse
from pathlib import Path

import pandas as pd

from talos.sweep import RunConfig, run_sweep

COLS = ["focus_objects", "focus_removed", "focus_half_way", "focus_progress_mean", "focus_progress_max",
        "focus_dv_per_burst", "focus_dv_total_m_s", "focus_bursts", "focus_need_median_m_s", "focus_mu_median",
        "focus_removed_mu_max", "useful_engagements_per_hr_per_sat", "debris_years_removed_per_yr",
        "cost_per_debris_year_usd"]


def designs(a) -> list[tuple[str, str, dict]]:
    out = []
    if a.designs_csv:
        df = pd.read_csv(a.designs_csv)
        for j, r in df.iterrows():
            tag = str(r["tag"]) if "tag" in r else f"csv{j:02d}"
            out.append((tag, "config/design.toml", {k: float(v) for k, v in r.items() if "." in k}))
    for c in a.config or ([] if a.designs_csv else ["config/design.toml"]):
        out.append((Path(c).stem, c, {}))
    return out


if __name__ == "__main__":  # required on Windows (spawned workers re-import this file)
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float, nargs="?", default=30.0)
    ap.add_argument("--out", default="runs/large_debris")
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--designs-csv", default=None)
    ap.add_argument("--count", type=int, default=0, help="focus-set size; 0 = every live object >= min diameter")
    ap.add_argument("--min-diameter-cm", type=float, default=9.0)
    ap.add_argument("--select", default="random", choices=("random", "lightest", "heaviest"))
    ap.add_argument("--mixed", action="store_true", help="focus first, normal targets otherwise")
    ap.add_argument("--min-impulse-frac", type=float, nargs="*", default=[0.15],
                    help="burst-quality gate(s), x a full burst at phi_opt; several = one run each")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--ckpt", default="runs/design_2yr/checkpoints/design.pkl")
    ap.add_argument("--ckpt-day", type=float, default=720.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    camp = {"campaign.enabled": True, "campaign.count": a.count, "campaign.min_diameter_m": a.min_diameter_cm / 100,
            "campaign.select": a.select, "campaign.exclusive": not a.mixed,
            "run.duration_days": a.ckpt_day + a.days, "run.checkpoint_days": 0.0}
    many = len(a.min_impulse_frac) > 1
    cfgs = [RunConfig(f"{tag}_s{s}" + (f"_g{g:g}" if many else ""), cfg,
                      {**ovr, **camp, "campaign.seed": s, "campaign.min_impulse_frac": g},
                      resume=a.ckpt, fresh_platforms=True)
            for tag, cfg, ovr in designs(a) for s in a.seeds for g in a.min_impulse_frac]
    df = run_sweep(cfgs, out_dir=a.out, n_jobs=min(a.jobs, len(cfgs)), threads_per_job=a.threads)
    pd.set_option("display.width", 250)
    print(df.set_index("name")[[c for c in COLS if c in df]].round(3).T.to_string())
