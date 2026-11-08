"""Build the solar-activity density table used by talos.atmosphere (needs pymsis).

Global-mean NRLMSIS 2.1 mass density vs altitude and F10.7, averaged over
latitude (cos-weighted), local time (longitude at 00 UT) and four seasonal
dates, with F10.7A = F10.7 and a constant Ap. Writes src/talos/data/msis_density.npz.

License note: NRLMSIS 2.x is distributed by NRL for non-commercial use; this
table is derived MSIS output. Regenerate or replace if that matters for you.

usage: python scripts/build_density_table.py [--ap 15]
"""

import argparse
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "talos" / "data" / "msis_density.npz"


def main(ap: float) -> None:
    import pymsis

    alts = np.concatenate([np.arange(100.0, 300.0, 5.0), np.arange(300.0, 1510.0, 10.0)])
    f107 = np.array([65.0, 75.0, 90.0, 110.0, 130.0, 150.0, 175.0, 200.0, 230.0, 260.0])
    dates = np.array(["2026-03-20T00:00", "2026-06-21T00:00", "2026-09-22T00:00", "2026-12-21T00:00"],
                     dtype="datetime64[m]")
    lons = np.arange(0.0, 360.0, 15.0)
    lats = np.arange(-87.5, 90.0, 5.0)
    w_lat = np.cos(np.deg2rad(lats))
    w_lat /= w_lat.sum()
    log_rho = np.empty((len(alts), len(f107)))
    for k, f in enumerate(f107):
        out = pymsis.calculate(dates, lons, lats, alts, f107s=np.full(len(dates), f),
                               f107as=np.full(len(dates), f), aps=np.full((len(dates), 7), ap))
        rho = out[..., pymsis.Variable.MASS_DENSITY]  # (dates, lons, lats, alts)
        mean = np.einsum("dlah,a->h", rho, w_lat) / (len(dates) * len(lons))
        log_rho[:, k] = np.log(mean)
        print(f"F10.7={f:5.0f}: rho(400)={np.exp(np.interp(400, alts, log_rho[:, k])):.3e}  "
              f"rho(800)={np.exp(np.interp(800, alts, log_rho[:, k])):.3e} kg/m^3", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, alt_km=alts, f107=f107, log_rho=log_rho, ap=ap,
                        source=f"NRLMSIS 2.1 via pymsis {pymsis.__version__}, global mean, Ap={ap}")
    print("wrote", OUT)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ap", type=float, default=15.0)
    main(a.parse_args().ap)
