"""Gate 4 vs ORDEM: encounter rate along a platform orbit.

The rate at which objects enter a sphere of radius R around the platform is
flux x pi R^2 for any direction distribution (uniform density on the scale of R).
So the sim's raw entry rate within R, divided by pi R^2, should reproduce ORDEM's
1-10 cm flux on that orbit. Checks the whole chain: inclination fit, sampling,
propagation, screening geometry.

    python scripts/validate_encounters_ordem.py [--alt 800] [--inc 98.6] [--hours 12]
"""

import argparse
from dataclasses import replace

import numpy as np

from talos.config import build, load_scenario, make_atmosphere
from talos.constants import R_EARTH, YEAR
from talos.elements import KeplerElements
from talos.metrics import raw_encounter_rate
from talos.ordem import DATA_DIR, OrdemTable, inclination_kernel
from talos.platform import baseline_platform
from talos.propagator import set_atmosphere


def ordem_fluxes(alt_km: float, inc_deg: float) -> tuple[float, float]:
    """(ORDEM 1-10 cm flux, flux predicted by the fitted table) [#/m^2/yr] on a circular orbit."""
    raw = dict(np.load(DATA_DIR / "ordem_raw_2026.npz"))
    ia = int(np.argmin(np.abs(raw["alt_km"] - alt_km)))
    js = int(np.argmin(np.abs(raw["sc_inc_deg"] - inc_deg)))
    f_ordem = float(raw["flux"][ia, js].sum())
    tab = OrdemTable.load(2026)
    a = int(np.searchsorted(tab.alt_edges_km, alt_km) - 1)
    per_m = tab.count[a].sum(axis=(0, 1)) / ((tab.alt_edges_km[a + 1] - tab.alt_edges_km[a]) * 1e3)
    k = inclination_kernel(raw["alt_km"][ia], raw["sc_inc_deg"][js], tab.inc_edges_deg, raw["v_edges_km_s"])
    return f_ordem, float((k @ per_m).sum())


def kernel_check(alt: float = 900.0, inc: float = 98.6, hours: float = 6.0, n: int = 120000,
                 shell_km=(700.0, 1100.0), R_km: float = 100.0) -> None:
    """Forward kernel (talos.ordem.inclination_kernel, used by the fit) vs simulated geometry:
    thick circular shells of one inclination bin, so the sphere sees ~uniform density."""
    from talos.debris import DebrisPopulation

    rng = np.random.default_rng(1)
    lo, hi = shell_km
    R = R_km * 1e3
    sat = baseline_platform(alt)
    o = sat.orbit
    sat = replace(sat, orbit=KeplerElements(a=o.a, e=o.e, i=np.deg2rad(inc), raan=o.raan, argp=o.argp, M=o.M))
    edges, vel = np.arange(0, 180.1, 5.0), np.arange(0.0, 24.0, 1.0)
    k = inclination_kernel(alt, inc, edges, vel)
    t = np.arange(0.0, hours * 3600, 1.0)
    zz = np.cos(np.deg2rad(edges[:-1])) - np.cos(np.deg2rad(edges[1:]))
    r = R_EARTH + alt * 1e3
    rr = R_EARTH + np.linspace(lo, hi, 1001) * 1e3
    corr = (1 / r**2) / np.mean(1 / rr**2)  # shell density ~ 1/r^2 across the thick shell
    print(f"kernel check at {alt:.0f} km, i = {inc} deg, R = {R_km:.0f} km, {hours:g} h")
    for ia, ib in [(0, 180), (30, 35), (60, 65), (80, 85), (95, 100)]:
        i = np.arccos(rng.uniform(np.cos(np.deg2rad(ib)), np.cos(np.deg2rad(ia)), n))
        el = KeplerElements(a=R_EARTH + rng.uniform(lo, hi, n) * 1e3, e=np.zeros(n), i=i,
                            raan=rng.uniform(0, 2 * np.pi, n), argp=np.zeros(n), M=rng.uniform(0, 2 * np.pi, n))
        pop = DebrisPopulation(elements=el, diameter_m=0.02, areal_density=1e6, weight=1.0)
        rate = raw_encounter_rate(sat, pop, t, range_m=R)
        sim = rate * 60 * YEAR / 3600 / (np.pi * R**2)
        js = [j for j in range(len(edges) - 1) if edges[j] >= ia and edges[j + 1] <= ib]
        w = np.zeros(len(edges) - 1)
        w[js] = zz[js] / zz[js].sum()
        kern = (k @ (w * n / ((hi - lo) * 1e3))).sum() * corr
        print(f"  inc {ia:3d}-{ib:3d}: sim / kernel = {sim / kern:.3f} +- {1 / np.sqrt(rate * hours * 60):.3f}")


def main(alt: float, inc: float, hours: float, ranges_km: list[float], circular: bool) -> None:
    f_ordem, f_fit = ordem_fluxes(alt, inc)
    sc = load_scenario(overrides={"population.min_lifetime_yr": None, "population.areal_density_model": "fixed"})
    set_atmosphere(make_atmosphere(sc))
    pop = build(sc).population
    if circular:  # drop the sampler's e ~ U(0, 0.01): ORDEM's altitude already folds in eccentricity
        pop.elements.e[:] = 0.0
    sat = baseline_platform(alt)
    o = sat.orbit
    sat = replace(sat, orbit=KeplerElements(a=o.a, e=o.e, i=np.deg2rad(inc), raan=o.raan, argp=o.argp, M=o.M))
    print(f"orbit {alt:.0f} km, i = {np.rad2deg(sat.orbit.i):.1f} deg (ORDEM grid {inc} deg); "
          f"population {pop.represented_count:,.0f} objects (no lifetime filter{', circular' if circular else ''})")
    print(f"ORDEM flux 1-10 cm: {f_ordem:.3e} /m2/yr   fitted table: {f_fit:.3e} ({f_fit / f_ordem:.3f}x)")
    t = np.arange(0.0, hours * 3600.0, 1.0)
    print(" R [km]  entries/min   sim flux [/m2/yr]   sim / ORDEM   +-1 sigma")
    rows = []
    for R in ranges_km:
        rate = raw_encounter_rate(sat, pop, t, range_m=R * 1e3)
        n = rate * (t[-1] - t[0]) / 60.0
        flux = rate * 60.0 * YEAR / 3600.0 / (np.pi * (R * 1e3) ** 2)
        sig = 1 / np.sqrt(max(n, 1))
        rows.append((R, flux / f_ordem, sig))
        print(f"{R:7.0f}  {rate:11.2f}   {flux:17.3e}   {flux / f_ordem:11.3f}   {sig:9.3f}")
    # finite sphere: it averages density over +-R of altitude; fit ratio(R) = r0 (1 + c R^2), R <= 200 km
    use = [(R, y, s) for R, y, s in rows if R <= 200.0]
    if len(use) >= 2:
        R, y, s = (np.array(v) for v in zip(*use))
        A = np.stack([np.ones_like(R), R**2], 1) / (y * s)[:, None]
        (r0, r0c), *_ = np.linalg.lstsq(A, 1.0 / s, rcond=None)
        cov = np.linalg.inv(A.T @ A)
        print(f"R -> 0 extrapolation (r0 (1 + c R^2), R <= 200 km): sim / ORDEM = {r0:.3f} +- {np.sqrt(cov[0, 0]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=800.0)
    ap.add_argument("--inc", type=float, default=98.6)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--ranges", nargs="+", type=float, default=[50.0, 100.0, 200.0, 250.0])
    ap.add_argument("--circular", action="store_true", help="set e = 0 for every sampled object")
    ap.add_argument("--kernel-check", action="store_true", help="forward kernel vs simulated geometry instead")
    a = ap.parse_args()
    if a.kernel_check:
        kernel_check(a.alt, a.inc)
    else:
        main(a.alt, a.inc, a.hours, a.ranges, a.circular)
