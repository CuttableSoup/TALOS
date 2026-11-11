"""Gate 5b: TALOS orbital lifetime vs ESA DRAMA/OSCAR and NASA DAS, circular orbits.

Same grid as validate_lifetime_das.py (400-1000 km, A/m 0.05-1 m^2/kg, i = 98 deg,
Cd 2.2). OSCAR runs as an executable, in parallel, one private copy of the tool
directory per worker. Run time scales with min(lifetime, span): ~20 s for 7 yr,
~260 s for a 100-yr span, so --span (default 50 yr) sets the cost; anything past it is inf.

--solmag points at current ESA fap_day.dat / fap_mon.dat
(https://static.sdo.esoc.esa.int/SOLMAG/); the A/m = 0.1 and 1 rows are also run
with the May 2021 files the installer ships, to show how much the forecast vintage matters.

usage: python scripts/validate_lifetime_oscar.py --solmag <dir> [--starts 2026.97 2031.0] [--span 50]
       [--workers 24] [--out runs/oscar_lifetime]
"""

import argparse
import itertools
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from talos import oscar
from talos.atmosphere import Atmosphere, AtmosphereConfig
from talos.constants import R_EARTH, YEAR
from talos.das import lifetime as das_lifetime
from talos.lifetime import orbital_lifetime
from talos.propagator import get_atmosphere, set_atmosphere

ALTS = np.arange(400.0, 1001.0, 50.0)
AOMS = [0.05, 0.1, 0.2, 0.5, 1.0]
INC = 98.0


def main(solmag: str, workers: int, out: str, STARTS: list[float], CAP: float) -> None:
    import pandas as pd

    root = Path(tempfile.mkdtemp(prefix="oscar_"))
    local = threading.local()
    counter = itertools.count()

    def run(case):
        y0, h, aom, data = case
        key = f"wd_{data}"
        if not hasattr(local, key):
            n = next(counter)
            setattr(local, key, oscar.prepare_workdir(root / f"{data}_{n}", Path(solmag) if data == "esa2026" else None))
        return oscar.lifetime(getattr(local, key), y0, h, h, INC, aom, span_yr=CAP)

    cases = [(y0, h, aom, "esa2026") for y0 in STARTS for aom in AOMS for h in ALTS]
    cases += [(STARTS[0], h, aom, "esa2021") for aom in (0.1, 1.0) for h in ALTS]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        osc = list(ex.map(run, cases))
    print(f"{len(cases)} OSCAR cases in {time.perf_counter() - t0:.0f} s ({workers} workers)")

    rows, prev = [], get_atmosphere()
    for y0 in STARTS:
        set_atmosphere(Atmosphere(AtmosphereConfig(), y0))
        for aom in AOMS:
            tal = orbital_lifetime(R_EARTH + ALTS * 1e3, np.zeros_like(ALTS), np.full_like(ALTS, 2.2 * aom),
                                   max_s=CAP * YEAR) / YEAR
            for h, lt in zip(ALTS, tal):
                d, _, msg = das_lifetime(y0, h, h, INC, aom, end_year=y0 + CAP)
                rows.append({"start_year": y0, "alt_km": h, "area_to_mass": aom, "talos_yr": float(lt),
                             "das_yr": d if msg == 2 else np.inf})
    set_atmosphere(prev)
    df = pd.DataFrame(rows)
    o = pd.DataFrame([dict(start_year=c[0], alt_km=c[1], area_to_mass=c[2], data=c[3], oscar_yr=v)
                      for c, v in zip(cases, osc)])
    o = o.pivot_table(index=["start_year", "alt_km", "area_to_mass"], columns="data", values="oscar_yr").reset_index()
    df = df.merge(o, on=["start_year", "alt_km", "area_to_mass"]).rename(
        columns={"esa2026": "oscar_yr", "esa2021": "oscar2021_yr"})
    for c in ("talos_yr", "das_yr", "oscar_yr", "oscar2021_yr"):
        df.loc[df[c] >= CAP * 0.999, c] = np.inf
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out) / "lifetime.parquet")

    def ratio(a, b):
        return np.where(np.isfinite(df[a]) & np.isfinite(df[b]), df[a] / df[b], np.nan)

    df["talos_oscar"], df["das_oscar"] = ratio("talos_yr", "oscar_yr"), ratio("das_yr", "oscar_yr")
    fmt = lambda x: f"{x:6.2f}"
    for y0 in STARTS:
        s = df[df.start_year == y0]
        print(f"\n=== start {y0} ===")
        for col, name in (("oscar_yr", f"OSCAR lifetime [yr] (inf: > {CAP:.0f} yr)"),
                          ("talos_oscar", "TALOS / OSCAR"), ("das_oscar", "DAS / OSCAR")):
            print(name)
            print(s.pivot(index="alt_km", columns="area_to_mass", values=col).to_string(float_format=fmt, na_rep="     -"))
    s = df[(df.start_year == STARTS[0]) & df.oscar2021_yr.notna()]
    print(f"\nForecast vintage, start {STARTS[0]}: OSCAR with 2021 files / with 2026 files")
    print(s.assign(r=s.oscar2021_yr / s.oscar_yr).pivot(index="alt_km", columns="area_to_mass", values="r")
          .to_string(float_format=fmt, na_rep="     -"))
    for col in ("talos_oscar", "das_oscar"):
        r = np.log(df[col].dropna())
        lo = df[df.oscar_yr < 25][col].dropna()
        print(f"{col}: geo-mean {np.exp(r.mean()):.2f}, log-sd {r.std():.2f}; OSCAR < 25 yr: "
              f"geo-mean {np.exp(np.log(lo).mean()):.2f}, range {lo.min():.2f}-{lo.max():.2f} (n={len(lo)})")
    print(f"\nwork dirs left in {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solmag", required=True, help="dir with current ESA fap_day.dat / fap_mon.dat")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--starts", type=float, nargs="+", default=[2026.97])
    ap.add_argument("--span", type=float, default=50.0, help="propagation cap [yr]")
    ap.add_argument("--out", default="runs/oscar_lifetime")
    a = ap.parse_args()
    main(a.solmag, a.workers, a.out, a.starts, a.span)
