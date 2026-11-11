"""Monthly F10.7 series for ``[atmosphere] activity = "table"`` from ESA's fap_mon.dat.

ESA SOLMAG long-term file (https://static.sdo.esoc.esa.int/SOLMAG/fap_mon.dat):
observed smoothed monthly means, then ESA's 13-month-centred forecast (to 2129).
The "Pre" F10.7 column equals the observed value where one exists, so it is the
whole series. Months sit at mid-month decimal years. The Ap forecast is stored for
reference only; the density table holds Ap fixed (scripts/build_density_table.py).

Writes src/talos/data/f107_esa.npz (git-ignored like the ORDEM tables: rebuild after
downloading a new fap_mon.dat).

usage: python scripts/build_f107_table.py --fap path/to/fap_mon.dat [--from-year 1990]
"""

import argparse
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "talos" / "data" / "f107_esa.npz"


def main(fap: str, from_year: float) -> None:
    text = Path(fap).read_text()
    m = re.search(r"Last Update:\s*([0-9]{4}\s+\w+\s+[0-9]+)", text)
    updated = m.group(1) if m else "unknown"
    rows = np.array([[float(v) for v in l.split()[:11]] for l in text.splitlines()
                     if l.strip() and not l.startswith("#")])
    year = rows[:, 1] + (rows[:, 0] - 0.5) / 12.0
    f107, ap = rows[:, 5], rows[:, 9]
    keep = (year >= from_year) & (f107 > 0)
    np.savez_compressed(OUT, year=year[keep], f107=f107[keep], ap=ap[keep],
                        source=f"ESA SOLMAG fap_mon.dat, last update {updated}, F10.7 'Pre' column")
    y, f = year[keep], f107[keep]
    print(f"wrote {OUT}: {keep.sum()} months {y[0]:.2f}-{y[-1]:.2f} (ESA update {updated})")
    for yy in (2025.0, 2027.0, 2029.0, 2031.0, 2035.0):
        print(f"  F10.7({yy:.0f}) = {np.interp(yy, y, f):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fap", required=True)
    ap.add_argument("--from-year", type=float, default=1990.0)
    a = ap.parse_args()
    main(a.fap, a.from_year)
