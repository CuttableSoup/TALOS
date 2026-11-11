"""Gate 5c: is the TALOS / OSCAR lifetime gap the atmosphere model? Constant solar flux.

With the solar forecast removed (constant F10.7 = F10.7A, Ap 15, as in TALOS's
density table), two checks:

1. density: global-mean NRLMSISE-00 (OSCAR/FOCUS's model) / NRLMSIS 2.1 (TALOS's
   table), both from pymsis on the same grid as scripts/build_density_table.py;
2. lifetime: OSCAR scenario 3 (constant flux) vs TALOS ``activity="constant"``,
   circular, i = 98 deg, Cd 2.2.

If the lifetime ratio tracks 1 / density ratio, the gap is the model; if not,
it is something else in OSCAR (propagation, lifetime definition, inputs).

usage: python scripts/validate_lifetime_oscar_constflux.py [--workers 24] [--span 50]
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
from talos.lifetime import orbital_lifetime
from talos.propagator import get_atmosphere, set_atmosphere

FLUX = [80.0, 140.0, 200.0]
AP = 15.0


def global_mean_density(alts, f107, version):
    import pymsis

    dates = np.array(["2026-03-20T00:00", "2026-06-21T00:00", "2026-09-22T00:00", "2026-12-21T00:00"],
                     dtype="datetime64[m]")
    lons, lats = np.arange(0.0, 360.0, 15.0), np.arange(-87.5, 90.0, 5.0)
    w = np.cos(np.deg2rad(lats))
    w /= w.sum()
    out = pymsis.calculate(dates, lons, lats, alts, f107s=np.full(4, f107), f107as=np.full(4, f107),
                           aps=np.full((4, 7), AP), version=version)
    rho = out[..., pymsis.Variable.MASS_DENSITY]
    return np.einsum("dlah,a->h", rho, w) / (len(dates) * len(lons))


def main(workers: int, span: float, out: str) -> None:
    import pandas as pd

    dalts = np.arange(300.0, 1001.0, 100.0)
    print(f"Global-mean density ratio NRLMSISE-00 / NRLMSIS 2.1 (Ap {AP:.0f})")
    dens = pd.DataFrame({f: global_mean_density(dalts, f, 0) / global_mean_density(dalts, f, 2.1) for f in FLUX},
                        index=pd.Index(dalts, name="alt_km"))
    print(dens.to_string(float_format=lambda x: f"{x:6.2f}"))

    alts, aoms = np.arange(400.0, 801.0, 100.0), [0.1, 1.0]
    cases = [(f, h, a) for f in FLUX for a in aoms for h in alts]
    root = Path(tempfile.mkdtemp(prefix="oscar_cf_"))
    local, counter = threading.local(), itertools.count()

    def run(case):
        f, h, a = case
        if not hasattr(local, "wd"):
            local.wd = oscar.prepare_workdir(root / f"w{next(counter)}")
        return oscar.lifetime(local.wd, 2026.97, h, h, 98.0, a, span_yr=span, f107_constant=f, ap_constant=AP)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        osc = list(ex.map(run, cases))
    print(f"\n{len(cases)} OSCAR cases in {time.perf_counter() - t0:.0f} s")

    prev, tal = get_atmosphere(), {}
    for f in FLUX:
        set_atmosphere(Atmosphere(AtmosphereConfig(activity="constant", f107_constant=f), 2026.97))
        for a in aoms:
            lt = orbital_lifetime(R_EARTH + alts * 1e3, np.zeros_like(alts), np.full_like(alts, 2.2 * a),
                                  max_s=span * YEAR) / YEAR
            tal.update({(f, h, a): float(v) for h, v in zip(alts, lt)})
    set_atmosphere(prev)
    df = pd.DataFrame([dict(f107=c[0], alt_km=c[1], area_to_mass=c[2], oscar_yr=o, talos_yr=tal[c])
                       for c, o in zip(cases, osc)])
    df.loc[df.talos_yr >= span * 0.999, "talos_yr"] = np.inf
    df["talos_oscar"] = np.where(np.isfinite(df.oscar_yr) & np.isfinite(df.talos_yr), df.talos_yr / df.oscar_yr, np.nan)
    # density ratio averaged over the decay (log-mean from 200 km to the start altitude)
    grid = np.arange(200.0, 801.0, 25.0)
    rat = {f: global_mean_density(grid, f, 0) / global_mean_density(grid, f, 2.1) for f in FLUX}
    df["dens_ratio_path"] = [np.exp(np.log(rat[f][grid <= h]).mean()) for f, h in zip(df.f107, df.alt_km)]
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out) / "lifetime_constflux.parquet")
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nwork dirs left in {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--span", type=float, default=50.0)
    ap.add_argument("--out", default="runs/oscar_lifetime")
    a = ap.parse_args()
    main(a.workers, a.span, a.out)
