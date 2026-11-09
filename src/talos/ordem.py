"""NASA ORDEM 3.2 interface: run the model, reduce its output, fit a sampler table.

ORDEM gives debris *flux* seen by a spacecraft, not a list of orbits. This module
turns it into what module 1 needs, a count of objects per (altitude, inclination,
material, size) cell, in three steps:

1. ``run_grid`` runs ORDEM spacecraft mode for circular orbits on a grid of
   altitudes x spacecraft inclinations and keeps, per run, the flux in each
   relative-velocity bin for 1-3.16 cm and 3.16-10 cm debris of each material
   class (from IGLOOFLUX_SC.OUT, summed over impact direction).
2. ``inclination_kernel`` is the forward model: the flux-vs-relative-velocity
   that a circular spacecraft sees from circular debris at the same altitude
   with inclinations in one bin (RAAN and phase uniform). It uses the standard
   single-inclination density n(lat) = 1 / (2 pi^2 r^2 dr sqrt(cos^2 lat - cos^2 i)),
   integrated over debris heading so the turning-latitude singularity drops out.
3. ``fit_table`` solves for non-negative inclination-bin counts per
   (altitude, material, size) that reproduce ORDEM's velocity spectra at every
   spacecraft inclination at once (NNLS with a small smoothness penalty).

Approximations: debris is treated as circular at the run altitude (eccentric
objects passing through are counted as residents there, at circular speed);
ORDEM's uncertainty files are not used; the 1-10 cm size split uses ORDEM's
cumulative values at 1, 3.16 and 10 cm with a power law inside each half-decade.

Licence: ORDEM is NASA software under a Software Usage Agreement. Tables built
here are data derived from it; the SUA requires passing its waiver terms on to
anyone you give them to, so they are kept out of git (src/talos/data/ordem_*.npz).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import MU_EARTH, R_EARTH, YEAR

MATERIALS = ("NK", "LD", "MD", "HD", "IN")
# bulk density [kg/m^3]: ORDEM classes (User Guide Table 2-1); NaK-78 eutectic ~0.87 g/cc
MATERIAL_DENSITY = {"NK": 870.0, "LD": 1400.0, "MD": 2800.0, "HD": 7900.0, "IN": float("nan")}
SIZE_CODES = ("40", "45", "50")  # cumulative >= 10^(code/10) um: 1 cm, 3.16 cm, 10 cm
SIZE_EDGES_M = np.array([1e-2, 10**-1.5, 1e-1])
DATA_DIR = Path(__file__).resolve().parent / "data"


def ordem_home() -> Path:
    """Install root (has model/ and data/). TALOS_ORDEM_HOME overrides the default per-user path."""
    env = os.environ.get("TALOS_ORDEM_HOME")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", "")) / "NASA" / "ORDEM 3.2"


def table_path(year: int) -> Path:
    return DATA_DIR / f"ordem_{int(year)}.npz"


# --------------------------------------------------------------------------- running ORDEM

def write_input(project: Path, year: int, alt_km: float, inc_deg: float,
                igloo: str = "IGLOO_10x10x1.SIG") -> None:
    """ORDEM.IN for a circular spacecraft orbit, random RAAN (User Guide Table 3-1; line order is fixed)."""
    lines = [
        "!file=ORDEM.IN file",
        "1                    ! type of assessment (1=spacecraft 2=telescope/radar)",
        f"{int(year)}                 ! year of observation (2016-2050)",
        "!---------- Spacecraft assessment --------------------------------------",
        "1                    ! way to determine orbit (1=apogee/perigee, 2=semi-major axis/eccentricity)",
        f"{alt_km:.3f}              ! perigee altitude (km)",
        f"{alt_km:.3f}              ! apogee altitude (km)",
        "0.0                  ! semi-major axis (km)",
        "0.0                  ! eccentricity (0-1)",
        f"{inc_deg:.3f}               ! inclination (0 to 180 deg)",
        "0.0                  ! argument of perigee (0 to 360 deg, -1=random)",
        "-1                   ! right ascension of the ascending node, RAAN (0 to 360 deg, -1=random)",
        f"{igloo}    ! file defining all 'igloo' element boundaries (az,el,vel)",
        "!---------- Telescope/Radar assessment ---------------------------------",
        "0.000                ! sensor latitude (SP=-90 to NP=90)",
        "0.000                ! azimuth (0 deg=North, 90 deg=East, to 360 deg)",
        "90.000               ! elevation (0 deg=horz to 90 deg=zenith)",
        "ALT_50.TIG           ! file defining all 'igloo' element boundaries (alt)",
    ]
    project.mkdir(parents=True, exist_ok=True)
    (project / "ORDEM.IN").write_text("\n".join(lines) + "\n")


def read_igloo(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """IGLOOFLUX_SC.OUT -> (vel_lo [km/s], vel_hi, column names, flux [#/m^2/yr], shape (elements, cols)).

    Columns are 'XXYY' (material XX, cumulative size code YY), User Guide 3.2.1.5.
    """
    text = path.read_text().splitlines()
    hdr = text[10].split()
    if hdr[:7] != ["Element", "az_low", "az_high", "el_low", "el_high", "vel_low", "vel_high"]:
        raise ValueError(f"unexpected IGLOOFLUX header in {path}: {hdr[:7]}")
    cols = [c.removeprefix("Flux_") for c in hdr[7:]]
    d = np.loadtxt(text[12:], ndmin=2)
    return d[:, 5], d[:, 6], cols, d[:, 7:7 + len(cols)]


def reduce_igloo(vel_lo, vel_hi, cols, flux, v_edges_km_s) -> np.ndarray:
    """Flux per (material, size bin [1-3.16 cm, 3.16-10 cm], velocity bin), summed over direction."""
    k = {c: j for j, c in enumerate(cols)}
    cum = np.stack([[flux[:, k[m + s]] for s in SIZE_CODES] for m in MATERIALS])  # (mat, 3, elements)
    diff = cum[:, :-1] - cum[:, 1:]  # (mat, 2, elements), cumulative -> half-decade bins
    vb = np.clip(np.searchsorted(v_edges_km_s, 0.5 * (vel_lo + vel_hi)) - 1, 0, len(v_edges_km_s) - 2)
    out = np.zeros((len(MATERIALS), 2, len(v_edges_km_s) - 1))
    for b in range(out.shape[-1]):
        out[..., b] = diff[..., vb == b].sum(-1)
    return out


def run_spacecraft(year: int, alt_km: float, inc_deg: float, workdir: Path,
                   home: Path | None = None, timeout_s: float = 1800.0):
    """One ORDEM spacecraft-mode run. Returns read_igloo(...) of its IGLOOFLUX_SC.OUT."""
    home = Path(home or ordem_home())
    exe = home / "model" / "ORDEM32.exe"
    if not exe.exists():
        raise FileNotFoundError(f"ORDEM executable not found at {exe}; set TALOS_ORDEM_HOME")
    write_input(workdir, year, alt_km, inc_deg)
    # ORDEM opens data\... relative to the working directory, so it must run from the install root
    res = subprocess.run([str(exe), str(workdir) + os.sep], cwd=home, capture_output=True, text=True,
                         timeout=timeout_s)
    out = workdir / "IGLOOFLUX_SC.OUT"
    if res.returncode != 0 or not out.exists() or "ERROR" in res.stdout:
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-15:])
        raise RuntimeError(f"ORDEM failed (alt {alt_km} km, inc {inc_deg} deg):\n{tail}")
    return read_igloo(out)


def run_grid(year: int, alts_km, incs_deg, n_jobs: int = 8, home: Path | None = None,
             v_edges_km_s=None, keep_dir: Path | None = None, progress: bool = True) -> dict:
    """ORDEM spacecraft runs on alts x incs. Returns the raw dataset ``fit_table`` consumes."""
    alts = np.asarray(alts_km, float)
    incs = np.asarray(incs_deg, float)
    v_edges = np.arange(0.0, 24.0, 1.0) if v_edges_km_s is None else np.asarray(v_edges_km_s, float)
    flux = np.zeros((len(alts), len(incs), len(MATERIALS), 2, len(v_edges) - 1))
    root = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="talos_ordem_"))
    jobs = [(a, b) for a in range(len(alts)) for b in range(len(incs))]

    def one(job):
        a, b = job
        r = run_spacecraft(year, alts[a], incs[b], root / f"h{alts[a]:.0f}_i{incs[b]:.1f}", home)
        flux[a, b] = reduce_igloo(*r, v_edges)
        return job

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:  # each job is its own ORDEM process
        for k, (a, b) in enumerate(ex.map(one, jobs), 1):
            if progress:
                print(f"  ORDEM {k}/{len(jobs)}: {alts[a]:.0f} km, {incs[b]:.1f} deg", flush=True)
    return {"year": int(year), "alt_km": alts, "sc_inc_deg": incs, "v_edges_km_s": v_edges, "flux": flux}


# --------------------------------------------------------------------------- forward model + fit

def inclination_kernel(alt_km: float, sc_inc_deg: float, inc_edges_deg, v_edges_km_s,
                       n_u: int = 720, n_psi: int = 3000) -> np.ndarray:
    """Flux [#/m^2/yr] per relative-velocity bin seen by a circular spacecraft at ``alt_km``,
    inclination ``sc_inc_deg``, from 1 debris object per metre of altitude whose inclination
    is in bin j, distributed as sin(i) inside the bin (isotropic within the bin), circular,
    same altitude, uniform RAAN and phase. Shape (n_vbins, n_incbins).
    """
    r = R_EARTH + alt_km * 1e3
    v = np.sqrt(MU_EARTH / r)
    c = 1.0 / (2.0 * np.pi**2 * r**2)  # per object per metre of shell
    edges = np.deg2rad(np.asarray(inc_edges_deg, float))
    z = np.cos(edges[:-1]) - np.cos(edges[1:])  # sin(i)-basis normalisation per bin
    v_edges = np.asarray(v_edges_km_s, float) * 1e3
    nv, ni = len(v_edges) - 1, len(edges) - 1

    i_s = np.deg2rad(sc_inc_deg)
    u = (np.arange(n_u) + 0.5) * 2 * np.pi / n_u  # spacecraft argument of latitude (time-uniform)
    lat = np.arcsin(np.sin(i_s) * np.sin(u))
    cl = np.cos(lat)
    se = np.clip(np.cos(i_s) / cl, -1, 1)  # east component of spacecraft heading
    sn = np.sign(np.cos(u)) * np.sqrt(1 - se**2)  # north component: northbound when cos(u) > 0

    dpsi = np.pi / n_psi
    psi = -np.pi / 2 + (np.arange(n_psi) + 0.5) * dpsi  # debris heading, northbound branch
    de, dn = np.sin(psi), np.cos(psi)
    inc_d = np.arccos(np.clip(cl[:, None] * de[None, :], -1, 1))  # (n_u, n_psi)
    jb = np.clip(np.searchsorted(edges, inc_d) - 1, 0, ni - 1)
    k = np.zeros((nv, ni))
    for branch in (1.0, -1.0):  # northbound / southbound halves of the density
        vrel = v * np.hypot(de[None, :] - se[:, None], branch * dn[None, :] - sn[:, None])
        vb = np.searchsorted(v_edges, vrel) - 1
        ok = (vb >= 0) & (vb < nv)
        w = 0.5 * c * vrel * dpsi * YEAR / z[jb]  # density element x |v_rel| -> flux [#/m^2/yr]
        np.add.at(k, (vb[ok], jb[ok]), w[ok])
    return k / n_u


def _nnls(a: np.ndarray, y: np.ndarray, smooth: float) -> np.ndarray:
    from scipy.optimize import nnls  # build-time only

    n = a.shape[1]
    d = np.diff(np.eye(n), 2, axis=0)
    scale = np.linalg.norm(a) / np.sqrt(n)
    return nnls(np.vstack([a, smooth * scale * d]), np.concatenate([y, np.zeros(len(d))]), maxiter=50 * n)[0]


@dataclass
class OrdemTable:
    """Objects per cell for one year: ``count[alt, material, size, inc]`` (whole cell, not per metre)."""

    year: int
    alt_edges_km: np.ndarray
    inc_edges_deg: np.ndarray
    count: np.ndarray  # (n_alt, n_mat, 2, n_inc)
    size_slope: np.ndarray  # (n_alt, n_mat, 2): b in N(>d) ~ d^-b inside each half-decade
    fit_residual: np.ndarray  # (n_alt, n_mat, 2): |model - ORDEM| / ORDEM over all spectra
    materials: tuple[str, ...] = MATERIALS

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or table_path(self.year))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, year=self.year, alt_edges_km=self.alt_edges_km, inc_edges_deg=self.inc_edges_deg,
                            count=self.count, size_slope=self.size_slope, fit_residual=self.fit_residual,
                            materials=np.array(self.materials))
        return path

    @classmethod
    def load(cls, path_or_year) -> "OrdemTable":
        path = table_path(path_or_year) if isinstance(path_or_year, (int, np.integer)) else Path(path_or_year)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; build it with `python scripts/build_ordem_table.py`")
        with np.load(path) as f:
            return cls(int(f["year"]), f["alt_edges_km"], f["inc_edges_deg"], f["count"], f["size_slope"],
                       f["fit_residual"], tuple(str(m) for m in f["materials"]))

    def total(self, alt_km=(600.0, 1000.0), materials=None) -> float:
        m = [self.materials.index(x) for x in (materials or self.materials)]
        lo, hi = self.alt_edges_km[:-1], self.alt_edges_km[1:]
        frac = np.clip((np.minimum(hi, alt_km[1]) - np.maximum(lo, alt_km[0])) / (hi - lo), 0, 1)
        return float(np.einsum("a,amsi->", frac, self.count[:, m]))


def fit_table(raw: dict, inc_edges_deg=None, smooth: float = 0.02) -> OrdemTable:
    """Inclination-resolved counts from ``run_grid`` output. Altitude cells are centred on the run
    altitudes (edges halfway between them)."""
    alts = raw["alt_km"]
    inc_edges = np.arange(0.0, 180.1, 5.0) if inc_edges_deg is None else np.asarray(inc_edges_deg, float)
    da = np.diff(alts)
    alt_edges = np.concatenate([[alts[0] - da[0] / 2], (alts[:-1] + alts[1:]) / 2, [alts[-1] + da[-1] / 2]])
    flux = raw["flux"]  # (alt, sc_inc, mat, size, v)
    na, ns, nm, nz, nv = flux.shape
    count = np.zeros((na, nm, nz, len(inc_edges) - 1))
    resid = np.zeros((na, nm, nz))
    for a in range(na):
        kern = np.stack([inclination_kernel(alts[a], s, inc_edges, raw["v_edges_km_s"]) for s in raw["sc_inc_deg"]])
        for m in range(nm):
            for z in range(nz):
                y = flux[a, :, m, z]  # (sc_inc, v)
                tot = y.sum(1, keepdims=True)
                if tot.sum() <= 0:
                    continue
                w = 1.0 / np.maximum(tot, 1e-300)  # each spacecraft inclination counts equally
                c = _nnls((kern * w[..., None]).reshape(ns * nv, -1), (y * w).ravel(), smooth)
                model = np.einsum("svj,j->sv", kern, c)
                resid[a, m, z] = np.abs(model - y).sum() / y.sum()
                count[a, m, z] = c * (alt_edges[a + 1] - alt_edges[a]) * 1e3
    # size slope from the two half-decade fluxes (orbit-averaged over all spacecraft inclinations)
    f = flux.sum(axis=(1, 4))  # (alt, mat, size)
    cum = np.stack([f[..., 0] + f[..., 1], f[..., 1], np.zeros_like(f[..., 1])], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = np.log(cum[..., 0] / cum[..., 1]) / np.log(10**0.5)
    b1 = np.where(np.isfinite(b1) & (b1 > 0), b1, 2.5)
    slope = np.stack([b1, b1], -1)  # >10 cm is not resolved here; reuse the lower slope for the upper bin
    return OrdemTable(int(raw["year"]), alt_edges, inc_edges, count, slope, resid)


# --------------------------------------------------------------------------- sampling

def sample_cells(table: OrdemTable, n: int, rng: np.random.Generator, alt_km=(600.0, 1000.0),
                 materials=None) -> dict:
    """Draw ``n`` objects from the table: altitude [km], inclination [rad], diameter [m], material index.

    Cells are clipped to ``alt_km``; altitude is uniform in a cell, cos(i) uniform in an inclination
    bin (the fit's basis), and size follows N(>d) ~ d^-b inside its half-decade.
    """
    mats = [table.materials.index(x) for x in (materials or table.materials)]
    lo, hi = np.maximum(table.alt_edges_km[:-1], alt_km[0]), np.minimum(table.alt_edges_km[1:], alt_km[1])
    frac = np.clip((hi - lo) / np.diff(table.alt_edges_km), 0, None)
    w = table.count * frac[:, None, None, None]
    mask = np.zeros(len(table.materials), bool)
    mask[mats] = True
    w = w * mask[None, :, None, None]
    p = w.ravel() / w.sum()
    cell = rng.choice(p.size, size=n, p=p)
    a, m, z, j = np.unravel_index(cell, w.shape)
    alt = rng.uniform(lo[a], hi[a])
    ie = np.deg2rad(table.inc_edges_deg)
    inc = np.arccos(rng.uniform(np.cos(ie[j + 1]), np.cos(ie[j])))
    b = table.size_slope[a, m, z]
    d1, d2 = SIZE_EDGES_M[z], SIZE_EDGES_M[z + 1]
    x = rng.uniform(0, 1, n)  # inverse CDF of the truncated power law N(>d) ~ d^-b on [d1, d2]
    lo_t, hi_t = d1**-b, d2**-b
    diam = (lo_t - x * (lo_t - hi_t)) ** (-1.0 / b)
    return {"alt_km": alt, "inc": inc, "diameter_m": diam, "material": m, "represented": float(w.sum())}
