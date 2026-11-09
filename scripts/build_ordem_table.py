"""Build the ORDEM 3.2 population table used by population.kind = "ordem".

Runs ORDEM spacecraft mode for circular orbits on an altitude x inclination grid
(see talos.ordem), fits inclination-resolved counts, and writes
src/talos/data/ordem_<year>.npz (plus the raw spectra, ordem_raw_<year>.npz, so
--refit can redo the fit without ORDEM). Both files are gitignored: they are data
derived from NASA SUA software.

usage: python scripts/build_ordem_table.py [--year 2026] [--jobs 12] [--refit]
"""

import argparse
import time

import numpy as np

from talos.ordem import DATA_DIR, MATERIALS, fit_table, run_grid


def main(year: int, jobs: int, refit: bool, alt_step: float) -> None:
    raw_path = DATA_DIR / f"ordem_raw_{year}.npz"
    if refit:
        with np.load(raw_path) as f:
            raw = {k: f[k] for k in f.files}
    else:
        alts = np.arange(600.0, 1000.0 + 1e-9, alt_step)
        incs = np.array([0.0, 30.0, 50.0, 65.0, 80.0, 90.0, 98.6, 110.0, 140.0, 170.0])
        t0 = time.perf_counter()
        raw = run_grid(year, alts, incs, n_jobs=jobs)
        print(f"{len(alts) * len(incs)} ORDEM runs in {time.perf_counter() - t0:.0f} s")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **raw)
    t = fit_table(raw)
    path = t.save()
    print(f"wrote {path}")
    tot = t.total((600.0, 1000.0))
    print(f"1-10 cm objects, 600-1000 km, {year}: {tot:,.0f}")
    for m, name in enumerate(MATERIALS):
        c = t.count[:, m].sum()
        if c > 0:
            r = (t.fit_residual[:, m] * t.count[:, m].sum(-1)).sum() / c
            print(f"  {name}: {c:9,.0f}  ({c / t.count.sum():5.1%})  fit residual {r:.1%}")
    print("by size: 1-3.16 cm {:,.0f}, 3.16-10 cm {:,.0f}".format(*t.count.sum(axis=(0, 1, 3))))
    inc_c = 0.5 * (t.inc_edges_deg[:-1] + t.inc_edges_deg[1:])
    by_inc = t.count.sum(axis=(0, 1, 2))
    top = np.argsort(by_inc)[::-1][:6]
    print("top inclination bins: " + ", ".join(f"{inc_c[j]:.1f} deg {by_inc[j] / by_inc.sum():.1%}" for j in top))
    alt_c = 0.5 * (t.alt_edges_km[:-1] + t.alt_edges_km[1:])
    print("by altitude: " + ", ".join(f"{a:.0f}:{c:,.0f}" for a, c in zip(alt_c, t.count.sum(axis=(1, 2, 3)))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--alt-step", type=float, default=20.0)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    main(a.year, a.jobs, a.refit, a.alt_step)
