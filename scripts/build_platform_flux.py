"""Build the platform debris-flux table used by talos.survival.

Runs ORDEM 3.2 spacecraft mode for circular sun-synchronous orbits at each altitude
and keeps the cross-section flux and the flux on two-sided plates normal to ram,
orbit normal and zenith, cumulative from 10 um to 1 m (all materials). Writes
src/talos/data/platform_flux_<year>.npz (gitignored: derived from NASA SUA software).

usage: python scripts/build_platform_flux.py [--year 2027] [--alts 600 700 800 900 1000] [--jobs 5]
"""

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from talos import ordem
from talos.constants import R_EARTH
from talos.propagator import sso_inclination
from talos.survival import SIZES_M, PlatformFluxTable, read_igloo_directional, read_sizeflux, reduce_directional


def main(year: int, alts, jobs: int) -> None:
    root = Path(tempfile.mkdtemp(prefix="talos_pflux_"))

    def one(alt):
        inc = float(np.rad2deg(sso_inclination(R_EARTH + alt * 1e3)))
        work = root / f"h{alt:.0f}"
        ordem.run_spacecraft(year, alt, inc, work)
        print(f"  ORDEM {alt:.0f} km, inc {inc:.2f} deg done", flush=True)
        return (reduce_directional(*read_igloo_directional(work / "IGLOOFLUX_SC.OUT")),
                read_sizeflux(work / "SIZEFLUX_SC.OUT"))

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(one, alts))
    flux = {k: np.stack([r[0][k] for r in res]) for k in res[0][0]}
    fine_sizes = res[0][1][0]
    fine = np.stack([np.interp(np.log10(fine_sizes), np.log10(r[1][0]), r[1][1]) for r in res])
    path = PlatformFluxTable(year, np.asarray(alts, float), SIZES_M, flux, fine_sizes, fine).save()
    print("wrote", path)
    for k in ("sphere", "x", "y", "z"):
        print(f"{k:6s} >=1 mm:", np.array2string(flux[k][:, 4], precision=4),
              " >=1 cm:", np.array2string(flux[k][:, 6], precision=6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--alts", type=float, nargs="*", default=[600.0, 700.0, 800.0, 900.0, 1000.0])
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    main(a.year, a.alts, a.jobs)
