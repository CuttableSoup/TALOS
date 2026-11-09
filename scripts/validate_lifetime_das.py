"""Gate 5: TALOS orbital lifetime vs NASA DAS 3.2.7 (LIFEDWELL), circular orbits.

Same Cd (2.2) and A/m definition (cross-section / mass) in both; TALOS uses
1 / areal_density. Differences come from the atmosphere (NRLMSIS 2.1 global-mean
table + parametric cycle vs Jacchia-77 + ODPO F10.7 forecast), SRP and
luni-solar terms (DAS only), and TALOS's orbit-averaged decay.

usage: python scripts/validate_lifetime_das.py [--out runs/das_lifetime]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from talos.atmosphere import Atmosphere, AtmosphereConfig
from talos.constants import R_EARTH, YEAR
from talos.das import lifetime as das_lifetime
from talos.lifetime import orbital_lifetime
from talos.propagator import get_atmosphere, set_atmosphere


def main(out: str) -> None:
    import pandas as pd

    alts = np.arange(400.0, 1001.0, 50.0)
    aoms = np.array([0.05, 0.1, 0.2, 0.5, 1.0])  # m^2/kg; SBM 1-10 cm spans ~0.05-1
    starts = [2026.97, 2031.0]
    inc = 98.0
    cap_yr = 100.0
    rows = []
    prev = get_atmosphere()
    t0 = time.perf_counter()
    for y0 in starts:
        set_atmosphere(Atmosphere(AtmosphereConfig(), y0))
        for aom in aoms:
            tal = orbital_lifetime(R_EARTH + alts * 1e3, np.zeros_like(alts), np.full_like(alts, 2.2 * aom),
                                   max_s=cap_yr * YEAR) / YEAR
            for h, lt in zip(alts, tal):
                d, _, msg = das_lifetime(y0, h, h, inc, aom, end_year=y0 + cap_yr)
                rows.append({"start_year": y0, "alt_km": h, "area_to_mass": aom, "talos_yr": float(lt),
                             "das_yr": d, "das_msg": msg})
    set_atmosphere(prev)
    df = pd.DataFrame(rows)
    ok = (df.das_msg == 2) & (df.das_yr < cap_yr) & (df.talos_yr < cap_yr * 0.999)  # DAS ignores end_year
    df["ratio"] = np.where(ok, df.talos_yr / df.das_yr, np.nan)
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out) / "lifetime.parquet")
    print(f"{len(df)} cases in {time.perf_counter() - t0:.0f} s; both re-enter within {cap_yr:.0f} yr: {ok.sum()}")
    for y0 in starts:
        piv = df[df.start_year == y0].pivot(index="alt_km", columns="area_to_mass", values="ratio")
        print(f"\nTALOS / DAS lifetime, start {y0} (blank: one of them > {cap_yr:.0f} yr)")
        print(piv.to_string(float_format=lambda x: f"{x:5.2f}", na_rep="   - "))
        d = df[df.start_year == y0].pivot(index="alt_km", columns="area_to_mass", values="das_yr")
        print(f"DAS lifetime [yr], start {y0}")
        print(d.to_string(float_format=lambda x: f"{x:7.2f}"))
    # lowered orbits, as the laser leaves them: apogee kept, perigee pulled down
    set_atmosphere(Atmosphere(AtmosphereConfig(), starts[0]))
    erows = []
    for ha in (800.0, 950.0):
        hp = np.arange(250.0, ha - 49.0, 50.0)
        a = R_EARTH + 0.5 * (hp + ha) * 1e3
        e = 0.5 * (ha - hp) * 1e3 / a
        for aom in (0.2, 0.5):
            tal = orbital_lifetime(a, e, np.full_like(a, 2.2 * aom), max_s=cap_yr * YEAR) / YEAR
            for p, lt in zip(hp, tal):
                d, _, msg = das_lifetime(starts[0], p, ha, inc, aom, end_year=starts[0] + cap_yr)
                erows.append({"apogee_km": ha, "perigee_km": p, "area_to_mass": aom, "talos_yr": float(lt),
                              "das_yr": d, "das_msg": msg})
    set_atmosphere(prev)
    ed = pd.DataFrame(erows)
    ed["ratio"] = ed.talos_yr / ed.das_yr
    ed.to_parquet(Path(out) / "lifetime_elliptic.parquet")
    print(f"\nLowered orbits, start {starts[0]}: TALOS / DAS lifetime (DAS days in brackets)")
    for (ha, aom), g in ed.groupby(["apogee_km", "area_to_mass"]):
        print(f"  apogee {ha:.0f} km, A/m {aom}: " +
              " ".join(f"{p:.0f}:{r:.2f}[{d * 365.25:.0f}]" for p, r, d in zip(g.perigee_km, g.ratio, g.das_yr)))

    r = np.log(df.ratio.dropna())
    lo = df[ok & (df.das_yr < 25)]
    print(f"\ngeo-mean ratio {np.exp(r.mean()):.2f}, log-sd {r.std():.2f}; "
          f"cases with DAS < 25 yr: geo-mean {np.exp(np.log(lo.ratio).mean()):.2f}, "
          f"range {lo.ratio.min():.2f}-{lo.ratio.max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/das_lifetime")
    main(ap.parse_args().out)
