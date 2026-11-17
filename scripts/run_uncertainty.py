"""Latin-hypercube uncertainty sweep over the CLAUDE.md input ranges (roadmap 7).

    python scripts/run_uncertainty.py N DAYS [--out runs/uncertainty] [--jobs 4] [--toy]
        [--response [--melt-policy cap --melt-margin M --melt-cap-basis commanded]] [--config path.toml]

Prints 5/50/95 % bands of the headline metrics and the Spearman rank correlation of
each input with each metric. ``--toy`` uses the toy population (no ORDEM table needed).
``--response`` samples talos.sweep.RESPONSE_INPUTS instead: the same inputs plus a wider Cm
range, the medium-density Cm scale, fragmentation per burst and thermal coupling (melt gate,
with ``--melt-policy`` / ``--melt-margin``). Compare bands only at equal DAYS; a short run
is not an annual rate.
"""

import argparse

from talos.sweep import RESPONSE_INPUTS, RunConfig, lhs_matrix, run_sweep, uncertainty_bands

METRICS = ["useful_engagements_per_hr_per_sat", "mean_dv_retro_per_engagement", "duty_cycle_mean",
           "removed_per_yr", "debris_years_removed_per_yr", "debris_years_net_per_yr", "cost_per_debris_year_usd",
           "cost_per_net_debris_year_usd", "net_change_per_yr"]


def main(n: int, days: float, out: str, jobs: int, toy: bool, seed: int, response: bool = False,
         melt_policy: str | None = None, melt_margin: float | None = None, config: str | None = None,
         melt_cap_basis: str | None = None) -> None:
    ovr = {"run.duration_days": days}
    ranges = None
    if response:
        ranges = RESPONSE_INPUTS
        # melt options not given keep the config's (design.toml: expected cap, margin 1)
        if melt_policy is not None:
            ovr["ablation.melt_policy"] = melt_policy
        if melt_cap_basis is not None:
            ovr["ablation.melt_cap_basis"] = melt_cap_basis
        if melt_margin is None and melt_policy is not None:
            # 3 = jitter allowance of the commanded cap; else the physical limit
            melt_margin = 3.0 if melt_policy == "cap" and (melt_cap_basis or "commanded") == "commanded" else 1.0
        if melt_margin is not None:
            ovr["ablation.melt_margin"] = melt_margin
    if toy:
        ovr |= {"population.kind": "toy", "population.areal_density_model": "sbm"}
    cfgs = lhs_matrix(n, ranges, base=RunConfig("lhs", config=config, overrides=ovr), seed=seed)
    df = run_sweep(cfgs, out_dir=out, n_jobs=jobs, threads_per_job=max(1, 32 // jobs))
    bands, corr = uncertainty_bands(df, [m for m in METRICS if m in df])  # net columns need fragments
    print(bands.to_string(float_format=lambda x: f"{x:,.3f}"))
    print()
    print(corr.to_string(float_format=lambda x: f"{x:+.2f}"))
    bands.to_csv(f"{out}/bands.csv")
    corr.to_csv(f"{out}/rank_corr.csv")


if __name__ == "__main__":  # required: Windows spawns workers by re-importing this file
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=32)
    ap.add_argument("days", type=float, nargs="?", default=1.0)
    ap.add_argument("--out", default="runs/uncertainty")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--response", action="store_true", help="sample RESPONSE_INPUTS (adds fragmentation, "
                    "thermal coupling, wider Cm, cm_scale_md)")
    ap.add_argument("--melt-policy", choices=("fragment", "cap"), default=None, help="default: the config's")
    ap.add_argument("--melt-margin", type=float, default=None,
                    help="with --melt-policy: default 3 for the commanded cap, else 1")
    ap.add_argument("--melt-cap-basis", choices=("commanded", "expected"), default=None,
                    help="expected: cap on the mean delivered fluence under jitter (margin defaults to 1)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    main(a.n, a.days, a.out, a.jobs, a.toy, a.seed, a.response, a.melt_policy, a.melt_margin, a.config,
         a.melt_cap_basis)
