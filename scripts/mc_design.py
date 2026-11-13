"""Monte Carlo design search: Latin-hypercube over the platform design levers, each design
flown over both mission phases, scored on whole-mission cost per debris-year.

    python scripts/mc_design.py N DAYS --out runs/mc_design [--jobs 4] [--threads 8]
        [--seed 0] [--late-ckpt runs/design_2yr/checkpoints/design.pkl] [--late-day 720]
    python scripts/mc_design.py --score runs/mc_design      # re-score finished runs only
    ... --large-test 30   # also fly each design's large-debris test (talos.campaign) for 30 days

Each design d runs twice (common random numbers: same population, same seed):
  * early sky: fresh start from config/design.toml, DAYS days;
  * late sky: --resume the day-720 design checkpoint with --fresh-platforms, DAYS days.
Design 000 is the current config/design.toml (control).

Mission score (README.md in git history, "Monte Carlo design search"): a mission of life T yr sees the early sky for its first
year and the late sky after (pick: the 2-yr design run's removals peak in month 3 and are
down ~5x by month 12), so debris-years/yr = (min(T,1) E + max(T-1,0) L) / T. Life T =
min(design life, laser shot life, storage life) with storage damage rate weighted the same
way, solved by fixed point. Storage cycle rating at 0.5 M (datasheet), 2.3 M and 4.5 M (BOSC)
cycles: with exponent 0 cycle damage scales as 1/rating, so no extra runs are needed.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from talos.sweep import RunConfig, lhs_matrix, run_sweep

DESIGN = "config/design.toml"
# design levers (all have mass/cost in talos.cost); pulse energy = power / 33 Hz
LEVERS = {
    "laser.burst_power_kw": (15.0, 120.0, "log"),
    "resources.radiator_area_m2": (30.0, 150.0),
    "resources.array_area_ft2": (1000.0, 4000.0),
    "resources.battery_capacity_kwh": (1.0, 10.0, "log"),
    "resources.burst_s": (10.0, 60.0),
    "resources.pcm_mass_kg": (25.0, 150.0, "log"),
}
CONTROL = {"laser.burst_power_kw": 25.0, "resources.radiator_area_m2": 60.0, "resources.array_area_ft2": 1600.0,
           "resources.battery_capacity_kwh": 2.0, "resources.burst_s": 20.0, "resources.pcm_mass_kg": 25.0}
CYCLE_RATINGS = (5.0e5, 2.3e6, 4.5e6)


def parse_box(items: list[str]) -> dict:
    """--box resources.radiator_area_m2=70:180 laser.burst_power_kw=15:35:log"""
    out = dict(LEVERS)
    for it in items:
        k, _, spec = it.partition("=")
        if k not in LEVERS:
            raise KeyError(f"--box {k}: not a lever ({sorted(LEVERS)})")
        lo, hi, *kind = spec.split(":")
        out[k] = (float(lo), float(hi), *kind)
    return out


def designs(n: int, seed: int, levers: dict = LEVERS) -> list[dict]:
    lhs = lhs_matrix(n - 1, levers, RunConfig("d"), seed=seed, vary_seed=False)
    return [dict(CONTROL)] + [dict(c.overrides) for c in lhs]


# large-debris test (scripts/large_debris_test.py defaults): day-720 sky, exclusive focus on every
# live object >= 9 cm
LARGE_TEST = {"campaign.enabled": True, "campaign.count": 0, "campaign.min_diameter_m": 0.09,
              "campaign.exclusive": True, "campaign.min_impulse_frac": 0.15}


def configs(ds: list[dict], days: float, late_ckpt: str, late_day: float,
            large_days: float = 0.0) -> list[RunConfig]:
    out = []
    for j, ovr in enumerate(ds):
        common = {**ovr, "run.checkpoint_days": 0.0}
        out.append(RunConfig(f"d{j:03d}_early", DESIGN, {**common, "run.duration_days": days}))
        out.append(RunConfig(f"d{j:03d}_late", DESIGN, {**common, "run.duration_days": late_day + days},
                             resume=late_ckpt, fresh_platforms=True))
        if large_days > 0:
            out.append(RunConfig(f"d{j:03d}_large", DESIGN,
                                 {**common, **LARGE_TEST, "run.duration_days": late_day + large_days},
                                 resume=late_ckpt, fresh_platforms=True))
    return out


def mission_score(early: pd.Series, late: pd.Series, ovr: dict, rating: float) -> dict:
    from talos.config import load_scenario
    from talos.cost import platform_mass_kg

    sc = load_scenario(DESIGN, ovr)
    cfg, r = sc.cost, sc.resources
    n = sc.constellation.n_sats
    mass = platform_mass_kg(r, sc.laser, cfg)["total"]
    scale = r.storage_cycles_100 / rating  # exponent 0: damage per cycle = 1 / rating

    def dmg(s):  # storage damage per yr at this rating (cycling + calendar)
        cyc = s.get("storage_cycle_life_yr", np.inf)
        d = (scale / cyc if np.isfinite(cyc) and cyc > 0 else 0.0)
        cal = r.storage_calendar_life_yr
        return d + (1.0 / cal if cal and np.isfinite(cal) else 0.0)

    shots = 0.5 * (early.get("laser_shots_per_sat_per_yr", 0) + late.get("laser_shots_per_sat_per_yr", 0))
    laser_life = cfg.laser_rated_shots / shots if shots > 0 else np.inf
    E, L = early["debris_years_removed_per_yr"], late["debris_years_removed_per_yr"]
    T = cfg.design_life_yr
    for _ in range(50):
        w = min(T, 1.0) / T
        storage_life = 1.0 / (w * dmg(early) + (1 - w) * dmg(late))
        T_new = min(cfg.design_life_yr, laser_life, storage_life)
        if abs(T_new - T) < 1e-6:
            break
        T = 0.5 * (T + T_new)
    rate = (min(T, 1.0) * E + max(T - 1.0, 0.0) * L) / T
    cost = n * (mass * (cfg.hardware_usd_per_kg + cfg.launch_usd_per_kg) + cfg.ops_usd_per_sat_yr * T)
    dy = rate * T
    return {"mass_kg": mass, "life_yr": T, "storage_life_yr": storage_life, "laser_life_yr": laser_life,
            "cost_musd": cost / 1e6, "debris_years_mission": dy,
            "usd_per_debris_year": cost / dy if dy > 0 else np.inf}


def score(out: Path) -> pd.DataFrame:
    df = pd.read_parquet(out / "summary.parquet").set_index("name")
    ds = pd.read_parquet(out / "designs.parquet")
    rows = []
    for j, ovr in ds.iterrows():
        e, l = f"d{j:03d}_early", f"d{j:03d}_late"
        if e not in df.index or l not in df.index:
            continue
        ovr = {k: float(v) for k, v in ovr.items()}
        row = {"design": f"d{j:03d}", **{k.split(".")[1]: v for k, v in ovr.items()}}
        E, L = df.loc[e], df.loc[l]
        row |= {"pulse_kj": ovr["laser.burst_power_kw"] / 33.0,
                "early_dy_per_yr": E["debris_years_removed_per_yr"], "late_dy_per_yr": L["debris_years_removed_per_yr"],
                "early_removed": E["removed"], "late_removed": L["removed"],
                "early_dv_burst": E.get("mean_dv_retro_per_engagement", np.nan),
                "late_dv_burst": L.get("mean_dv_retro_per_engagement", np.nan),
                "duty": E.get("duty_cycle_mean", np.nan), "late_cycle_life_yr": L.get("storage_cycle_life_yr", np.nan)}
        g = f"d{j:03d}_large"
        if g in df.index:  # large-debris test: per 30 days, so tests of other lengths compare
            G = df.loc[g]
            per30 = 30.0 * 24.0 / G["sim_hours"]
            row |= {"large_removed_per_30d": G["focus_removed"] * per30,
                    "large_dv_per_30d": G["focus_dv_total_m_s"] * per30,
                    "large_dv_burst": G["focus_dv_per_burst"], "large_progress_mean": G["focus_progress_mean"],
                    "large_removed_mu_max": G["focus_removed_mu_max"]}
        for rating in CYCLE_RATINGS:
            tag = f"{rating / 1e6:g}M"
            row |= {f"{k}_{tag}": v for k, v in mission_score(E, L, ovr, rating).items()}
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(f"usd_per_debris_year_{CYCLE_RATINGS[0] / 1e6:g}M")
    res.to_csv(out / "mc_design_scores.csv", index=False)
    return res


if __name__ == "__main__":  # required on Windows (spawned workers re-import this file)
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=0)
    ap.add_argument("days", type=float, nargs="?", default=7.0)
    ap.add_argument("--out", default="runs/mc_design")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--late-ckpt", default="runs/design_2yr/checkpoints/design.pkl")
    ap.add_argument("--late-day", type=float, default=720.0)
    ap.add_argument("--score", action="store_true", help="only re-score finished runs in --out")
    ap.add_argument("--large-test", type=float, default=0.0, metavar="DAYS",
                    help="also run the large-debris test (scripts/large_debris_test.py) for DAYS per design")
    ap.add_argument("--box", nargs="*", default=[], help="narrow lever ranges: key=lo:hi[:log]")
    ap.add_argument("--designs-csv", default=None,
                    help="fly these designs instead of an LHS (CSV, one column per lever key)")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.score:
        if a.designs_csv:
            ds = pd.read_csv(a.designs_csv)[list(LEVERS)].to_dict("records")
        else:
            ds = designs(a.n, a.seed, parse_box(a.box))
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ds).to_parquet(out / "designs.parquet")
        cfgs = configs(ds, a.days, a.late_ckpt, a.late_day, a.large_test)
        done = {p.stem for p in (out / "summaries").glob("*.parquet")} if (out / "summaries").exists() else set()
        todo = [c for c in cfgs if c.name not in done]
        print(f"{len(ds)} designs, {len(todo)} of {len(cfgs)} runs to go", flush=True)
        if todo:
            run_sweep(todo, out_dir=out, n_jobs=a.jobs, threads_per_job=a.threads)
    res = score(out)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
