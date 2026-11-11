"""ESA DRAMA 4.1.4 OSCAR orbital-lifetime reference, via its command-line executable.

OSCAR propagates with FOCUS (geopotential, luni-solar, SRP, NRLMSISE-00 drag) and
reads ESA's solar/geomagnetic activity files (data\\fap_day.dat, fap_mon.dat:
observed values, then ESA's forecast to 2129). The installer ships the May 2021
files; pass ``solmag_dir`` to use newer ones from https://static.sdo.esoc.esa.int/SOLMAG/.

OSCAR reads ``input\\oscar.inp`` relative to its working directory, so each worker
gets a private copy of the tool directory (``workdir``) and rewrites the input there.

DRAMA is ESA software under its own licence; this module only calls a local
install and ships none of it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

R_EARTH_KM = 6378.137


def oscar_home() -> Path:
    env = os.environ.get("TALOS_OSCAR_HOME")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", "")) / "ESA" / "DRAMA-4.1.4" / "TOOLS" / "OSCAR"


def prepare_workdir(workdir: Path, solmag_dir: Path | None = None) -> Path:
    """Copy the OSCAR tool directory to ``workdir`` (once), optionally swapping the solar files."""
    workdir = Path(workdir)
    if not (workdir / "oscar_windows.exe").exists():
        home = oscar_home()
        if not (home / "oscar_windows.exe").exists():
            raise FileNotFoundError(f"oscar_windows.exe not found in {home}; set TALOS_OSCAR_HOME")
        shutil.copytree(home, workdir, dirs_exist_ok=True)
        (workdir / "input" / "oscar.inp.orig").write_bytes((home / "input" / "oscar.inp").read_bytes())
    if solmag_dir is not None:
        for f in ("fap_day.dat", "fap_mon.dat"):
            shutil.copyfile(Path(solmag_dir) / f, workdir / "data" / f)
    return workdir


def _year_to_datetime(y: float) -> datetime:
    y0 = int(y)
    start = datetime(y0, 1, 1)
    return start + (datetime(y0 + 1, 1, 1) - start) * (y - y0)


def _set(lines: list[str], tag: str, value: str) -> None:
    """Replace the leading value on the first line whose comment contains ``tag``."""
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("#") and tag in line:
            lines[i] = re.sub(r"^\s*\S+", "  " + value, line, count=1)
            return
    raise KeyError(tag)


def write_input(workdir: Path, start_year: float, perigee_km: float, apogee_km: float, inc_deg: float,
                area_to_mass: float, cd: float = 2.2, cr: float = 1.3, raan_deg: float = 0.0,
                argp_deg: float = 0.0, span_yr: float = 100.0, solar_scenario: int = 1,
                f107_constant: float | None = None, ap_constant: float = 15.0) -> None:
    """``f107_constant`` switches to scenario 3 (constant flux, user-defined F10.7 and Ap)."""
    if f107_constant is not None:
        solar_scenario = 3
    lines = (Path(workdir) / "input" / "oscar.inp.orig").read_text().split("\n")
    # begin date: first non-comment line
    i = next(k for k, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#"))
    lines[i] = "  " + _year_to_datetime(start_year).strftime("%Y %m %d %H %M %S")
    rp, ra = R_EARTH_KM + perigee_km, R_EARTH_KM + apogee_km
    a, e = 0.5 * (rp + ra), (ra - rp) / (ra + rp)
    _set(lines, "Semi-major axis / km", f"{a:.3f}")          # first match = target orbit
    _set(lines, "Eccentricity / -", f"{e:.7f}")
    _set(lines, "Inclination / deg", f"{inc_deg:.4f}")
    _set(lines, "Right ascension of ascending node (RAAN) / deg", f"{raan_deg:.4f}")
    _set(lines, "Argument of perigee / deg", f"{argp_deg:.4f}")
    _set(lines, "Mean anomaly / deg", "0.0")
    _set(lines, "Cross-sectional area / m**2", f"{area_to_mass:.6g}")  # 1 kg object
    _set(lines, "Spacecraft mass / kg", "1.0")
    _set(lines, "Drag coefficient", f"{cd}")
    _set(lines, "Reflectivity coefficient", f"{cr}")
    # disposal option stays at the installer default, 6 = none (natural decay)
    span_idx = next(k for k, l in enumerate(lines) if "Propagation time span" in l)
    j = next(k for k in range(span_idx + 1, len(lines)) if lines[k].strip() and not lines[k].lstrip().startswith("#"))
    lines[j] = f"  {span_yr:.1f}"
    scen_idx = next(k for k, l in enumerate(lines) if "5 = Monte Carlo sampling" in l)
    j = next(k for k in range(scen_idx + 1, len(lines)) if lines[k].strip() and not lines[k].lstrip().startswith("#"))
    lines[j] = f"  {solar_scenario}"
    if f107_constant is not None:
        _set(lines, "Mode Switch (1 = auto", "2")
        _set(lines, "User-defined constant solar flux level", f"{f107_constant:.1f}")
        _set(lines, "User-defined constant geomagnetic activity Ap", f"{ap_constant:.1f}")
    for tag in ("Altitude vs. Time", "Eccentricity vs. Time", "Inclination vs. Time",
                "Solar and geomagnetic activity vs. time"):
        _set(lines, tag, "0")
    (Path(workdir) / "input" / "oscar.inp").write_text("\n".join(lines))


_LIFE = re.compile(r"#\s*([0-9.]+)\s+-\s+Remaining lifetime of target orbit")


def lifetime(workdir: Path, start_year: float, perigee_km: float, apogee_km: float, inc_deg: float,
             area_to_mass: float, span_yr: float = 100.0, timeout_s: float = 900.0, **kw) -> float:
    """OSCAR natural (disposal option 'none') remaining lifetime [yr]; inf if not found within span."""
    workdir = Path(workdir)
    write_input(workdir, start_year, perigee_km, apogee_km, inc_deg, area_to_mass, span_yr=span_yr, **kw)
    out = workdir / "output" / "oscar.oop"
    out.unlink(missing_ok=True)
    proc = subprocess.run([str(workdir / "oscar_windows.exe")], cwd=workdir, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout_s)
    if not out.exists():
        raise RuntimeError(f"OSCAR produced no output (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
    m = _LIFE.search(out.read_text())
    return float(m.group(1)) if m else float("inf")
