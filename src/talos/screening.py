"""Module 4 — engagement screening.

Two stages:
  1. ``apsis_filter``: cheap radial-band test on mean elements.
  2. ``screen_fine``: per-step geometry (range, Earth occultation, field of
     regard, LOS slew rate) on a ~1 s grid, then ``find_windows``.

TODO: coarse time-window prediction (node/phase crossing) so fine stepping
only runs inside candidate windows instead of the full grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _kernels
from .constants import MU_EARTH, R_EARTH
from .debris import DebrisPopulation
from .elements import KeplerElements
from .platform import ATPConfig, Platform
from .propagator import states_at


def apsis_filter(sat: KeplerElements, deb: KeplerElements, max_range_m: float) -> np.ndarray:
    """True where the radial bands of sat and debris come within ``max_range_m``."""
    gap = np.maximum(np.maximum(deb.rp - sat.ra, sat.rp - deb.ra), 0.0)
    return gap <= max_range_m


def los_clear(r_obs, r_tgt, clearance_radius_m: float) -> np.ndarray:
    """True where the segment observer->target stays outside ``clearance_radius_m``."""
    d = r_tgt - r_obs
    s = np.clip(-np.sum(r_obs * d, -1) / np.sum(d * d, -1), 0.0, 1.0)
    closest = r_obs + s[..., None] * d
    return np.linalg.norm(closest, axis=-1) > clearance_radius_m


def los_angular_rate(r_rel, v_rel) -> np.ndarray:
    """Inertial angular rate [rad/s] of the line of sight: |r x v| / |r|^2."""
    return np.linalg.norm(np.cross(r_rel, v_rel), axis=-1) / np.sum(r_rel * r_rel, -1)


def off_boresight_angle(r_rel, boresight) -> np.ndarray:
    u = r_rel / np.linalg.norm(r_rel, axis=-1, keepdims=True)
    b = boresight / np.linalg.norm(boresight, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(u * b, -1), -1.0, 1.0))


@dataclass
class ScreenResult:
    """Per-step (T, N) geometry and gate flags."""

    t: np.ndarray
    range_m: np.ndarray
    slew_rate: np.ndarray
    in_range: np.ndarray
    los_ok: np.ndarray
    in_for: np.ndarray
    slew_ok: np.ndarray
    sun_ok: np.ndarray | bool = True  # target sunlit and LOS outside the sun-exclusion cone
    tracked: np.ndarray | bool = True  # detection/track state machine (passive optical)

    @property
    def mask(self) -> np.ndarray:
        return self.in_range & self.los_ok & self.in_for & self.slew_ok & self.sun_ok & self.tracked


MAG_SUN = -26.74


def apparent_magnitude(r_sat, r_deb, sun_hat, radius, albedo, bg_penalty_mag=0.0) -> np.ndarray:
    """V magnitude of a Lambertian sphere (radius, geometric albedo) seen from r_sat.

    flux / flux_sun = (2 / 3pi) p r^2 / rho^2 [(pi - a) cos a + sin a], a = phase angle.
    Adds ``bg_penalty_mag`` where the line of sight continues onto the sunlit Earth.
    Shapes: r_sat (T, 1, 3), r_deb (T, N, 3), sun_hat (T, 1, 3), radius/albedo (N,).
    """
    d = r_deb - r_sat
    d2 = np.sum(d * d, -1)
    rho = np.sqrt(d2)
    cosa = np.clip(-np.sum(d * sun_hat, -1) / rho, -1.0, 1.0)
    al = np.arccos(cosa)
    phase = (np.pi - al) * cosa + np.sin(al)
    flux = (2.0 / (3.0 * np.pi)) * albedo * radius ** 2 / d2 * phase
    mag = MAG_SUN - 2.5 * np.log10(np.maximum(flux, 1e-300))
    if bg_penalty_mag:
        b = np.sum(r_sat * d, -1)
        c = np.sum(r_sat * r_sat, -1) - R_EARTH ** 2
        disc = b * b - d2 * c
        s1 = (-b - np.sqrt(np.maximum(disc, 0.0))) / d2
        hit = r_sat + s1[..., None] * d
        bg = (disc > 0) & (s1 > 1.0) & (np.sum(hit * sun_hat, -1) > 0)
        mag = mag + bg_penalty_mag * bg
    return mag


def track_mask(visible, mag, atp: ATPConfig, dt: float) -> np.ndarray:
    """(T, N) track state: detect (mag <= detect limit) for acquisition_s consecutive
    seconds to start a track, keep it while mag <= track limit. A target detectable at
    the first step is treated as already tracked. Mirrors fine_mask_kernel."""
    acq = max(1, int(np.ceil(atp.acquisition_s / dt - 1e-9)))
    det = visible & (mag <= atp.detect_mag_limit)
    trk = visible & (mag <= atp.track_mag_limit)
    T, N = det.shape
    out = np.zeros((T, N), dtype=bool)
    tracked = np.zeros(N, dtype=bool)
    run = np.zeros(N, dtype=int)
    for k in range(T):
        lost = tracked & ~trk[k]
        tracked &= ~lost
        run[lost] = 0
        nt = ~tracked
        run[nt] = np.where(det[k, nt], acq if k == 0 else run[nt] + 1, 0)
        tracked |= nt & (run >= acq)
        out[k] = tracked
    return out


def sun_gate(r_rel, rng, r_deb, sun_hat, atp: ATPConfig) -> np.ndarray:
    """Target sunlit (cylindrical shadow, if required) and LOS outside the sun-exclusion cone.
    ``sun_hat``: (T, 3)."""
    from .sun import target_lit

    s = sun_hat[:, None, :]
    ok = np.sum(r_rel * s, -1) / rng <= np.cos(atp.sun_exclusion_rad)
    if atp.require_target_sunlit:
        ok &= target_lit(r_deb, s)
    return ok


def screen_fine(t, r_sat, v_sat, r_deb, v_deb, atp: ATPConfig, sun_hat=None, radius=None,
                albedo=None) -> ScreenResult:
    """Gate checks. ``r_sat``/``v_sat``: (T, 3); ``r_deb``/``v_deb``: (T, N, 3);
    ``sun_hat``: optional (T, 3) sun unit vectors enabling the sun gates and, with
    ``atp.detection_model == "passive_optical"``, the detection/track model (needs
    per-object ``radius`` and ``albedo``)."""
    from .sun import target_lit

    r_sat = r_sat[:, None, :]
    v_sat = v_sat[:, None, :]
    r_rel = r_deb - r_sat
    v_rel = v_deb - v_sat
    rng = np.linalg.norm(r_rel, axis=-1)
    slew = los_angular_rate(r_rel, v_rel)
    los_ok = los_clear(r_sat, r_deb, R_EARTH + atp.earth_clearance_alt_m)
    tracked = True
    if sun_hat is not None and atp.detection_model == "passive_optical":
        if radius is None or albedo is None:
            raise ValueError("passive_optical detection needs radius and albedo")
        s = sun_hat[:, None, :]
        visible = (los_ok & (rng <= atp.detection_range_m) & target_lit(r_deb, s)
                   & (np.sum(r_rel * s, -1) / rng <= np.cos(atp.sun_exclusion_rad)))
        mag = apparent_magnitude(r_sat, r_deb, s, np.asarray(radius), np.asarray(albedo),
                                 atp.earth_background_penalty_mag)
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        tracked = track_mask(visible, mag, atp, dt)
    elif atp.detection_model not in ("passive_optical", "ideal"):
        raise ValueError(f"unknown detection_model {atp.detection_model!r}")
    return ScreenResult(
        sun_ok=True if sun_hat is None else sun_gate(r_rel, rng, r_deb, sun_hat, atp),
        tracked=tracked,
        t=np.asarray(t),
        range_m=rng,
        slew_rate=slew,
        in_range=(rng >= atp.engage_range_min_m) & (rng <= atp.engage_range_max_m),
        los_ok=los_ok,
        in_for=off_boresight_angle(r_rel, v_sat) <= atp.field_of_regard_half_angle_rad,
        slew_ok=slew <= atp.slew_rate_limit_rad_s,
    )


@dataclass(frozen=True)
class Window:
    debris_index: int  # column index into the screened population
    start: int  # first time index (inclusive)
    stop: int  # last time index + 1

    @property
    def n_steps(self) -> int:
        return self.stop - self.start


def find_windows(mask: np.ndarray, min_steps: int = 1) -> list[Window]:
    """Contiguous True runs along axis 0 of a (T, N) mask, sorted by (debris, start)."""
    if _kernels.HAVE_NUMBA:
        col, st, sp = _kernels.mask_runs(np.ascontiguousarray(mask, dtype=np.bool_), int(min_steps))
        return [Window(c, s, e) for c, s, e in zip(col.tolist(), st.tolist(), sp.tolist())]
    T, N = mask.shape
    pad = np.zeros((1, N), dtype=np.int8)
    d = np.diff(np.concatenate([pad, mask.astype(np.int8), pad]), axis=0)
    starts = np.argwhere(d == 1)
    stops = np.argwhere(d == -1)
    starts = starts[np.lexsort((starts[:, 0], starts[:, 1]))]
    stops = stops[np.lexsort((stops[:, 0], stops[:, 1]))]
    return [
        Window(int(c), int(s), int(e))
        for (s, c), (e, _) in zip(starts, stops)
        if e - s >= min_steps
    ]


def max_speed(el: KeplerElements) -> np.ndarray:
    """Upper bound on orbital speed [m/s]: vis-viva at perigee."""
    return np.sqrt(MU_EARTH * (2.0 / el.rp - 1.0 / el.a))


def coarse_sieve(
    sat: KeplerElements,
    deb: KeplerElements,
    ballistic_coeff,
    t: np.ndarray,
    max_range_m: float,
    coarse_step_s: float,
    epoch=None,
) -> np.ndarray:
    """Objects that *may* come within ``max_range_m`` during [t[0], t[-1]].

    Range is Lipschitz with constant v_max = v_sat,max + v_deb,max, so over a
    coarse interval of length h the minimum range is at least
    (rho_k + rho_{k+1} - v_max h) / 2. Conservative: never drops a true window.
    """
    n = max(1, int(np.ceil((t[-1] - t[0]) / coarse_step_s)))
    tc = np.linspace(t[0], t[-1], n + 1)
    h = tc[1] - tc[0] if n > 0 else 0.0
    r_s, _ = states_at(sat, tc)
    r_d, _ = states_at(deb, tc, ballistic_coeff, epoch=epoch)
    rho = np.linalg.norm(r_d - r_s[:, None, :], axis=-1)  # (n+1, N)
    vmax = max_speed(sat) + max_speed(deb)
    lower = 0.5 * (rho[:-1] + rho[1:] - vmax * h) if n > 0 else rho
    return np.any(lower <= max_range_m, axis=0) | np.any(rho <= max_range_m, axis=0)


def screen_windows(
    platform: Platform,
    population: DebrisPopulation,
    t: np.ndarray,
    candidates: np.ndarray | None = None,
    min_steps: int = 1,
    batch: int = 1000,
    coarse_step_s: float | None = 30.0,
    use_kernel: bool | None = None,
    sat_states: tuple[np.ndarray, np.ndarray] | None = None,
    sun_hat: np.ndarray | None = None,
    elem_args: tuple | None = None,
) -> list[Window]:
    """Memory-bounded screening: windows only, objects processed in batches.

    ``candidates`` optionally restricts to these population indices (e.g. not yet removed).
    ``coarse_step_s`` enables the conservative coarse sieve (None = fine-step everything).
    Window ``debris_index`` refers to the original population index.
    ``sun_hat`` (T, 3) enables the sun gates (target sunlit, sun exclusion).
    ``elem_args``: ``_kernels.element_args`` of the WHOLE population at its current state,
    computed once and shared by all platforms in a chunk (kernel path only).
    """
    idx = np.arange(len(population)) if candidates is None else np.asarray(candidates)
    rmax = platform.atp.engage_range_max_m
    idx = idx[apsis_filter(platform.orbit, population.elements[idx], rmax)]
    if use_kernel is None:
        use_kernel = _kernels.HAVE_NUMBA
    if use_kernel and len(idx) and len(t) > 1:
        return _screen_windows_kernel(platform, population, t, idx, min_steps, coarse_step_s, sat_states,
                                      sun_hat, elem_args)
    if coarse_step_s is not None and len(idx) and len(t) > 1:
        keep = np.zeros(len(idx), dtype=bool)
        for k in range(0, len(idx), 4 * batch):
            cols = idx[k : k + 4 * batch]
            keep[k : k + 4 * batch] = coarse_sieve(
                platform.orbit, population.elements[cols], population.ballistic_coeff[cols],
                t, rmax, coarse_step_s, population.epoch_s[cols],
            )
        idx = idx[keep]
    r_s, v_s = sat_states if sat_states is not None else states_at(platform.orbit, t)
    out: list[Window] = []
    for k in range(0, len(idx), batch):
        cols = idx[k : k + batch]
        sub = population.subset(cols)
        r_d, v_d = states_at(sub.elements, t, sub.ballistic_coeff, epoch=sub.epoch_s)
        mask = screen_fine(t, r_s, v_s, r_d, v_d, platform.atp, sun_hat, 0.5 * sub.diameter_m, sub.albedo).mask
        out += [Window(int(cols[w.debris_index]), w.start, w.stop) for w in find_windows(mask, min_steps)]
    return out


def _screen_windows_kernel(platform, population, t, idx, min_steps, coarse_step_s, sat_states=None,
                           sun_hat=None, elem_args=None) -> list[Window]:
    """Numba path: parallel coarse sieve, then parallel fused fine gates.

    ``sat_states`` = (r, v) of the platform on grid ``t`` (precomputed/cached);
    the coarse grid is then a strided subset of ``t`` so no extra propagation is needed.
    """
    atp = platform.atp
    rmax = atp.engage_range_max_m
    sub = population.elements[idx]
    args = (_kernels.element_args(sub, population.ballistic_coeff[idx], population.epoch_s[idx])
            if elem_args is None else tuple(x[idx] for x in elem_args))
    if sat_states is not None:
        r_s, v_s = sat_states
    if coarse_step_s is not None:
        if sat_states is not None:
            dt = t[1] - t[0]
            step = max(1, int(round(coarse_step_s / dt)))
            ci = np.unique(np.append(np.arange(0, len(t), step), len(t) - 1))
            tc, r_sc = np.ascontiguousarray(t[ci]), r_s[ci]
        else:
            n = max(1, int(np.ceil((t[-1] - t[0]) / coarse_step_s)))
            tc = np.linspace(t[0], t[-1], n + 1)
            r_sc, _ = states_at(platform.orbit, tc)
        vmax = np.ascontiguousarray(max_speed(platform.orbit) + max_speed(sub), dtype=float)
        keep = _kernels.sieve_kernel(tc, np.ascontiguousarray(r_sc), vmax, rmax, *args)
        idx = idx[keep]
        args = tuple(x[keep] for x in args)
    if not len(idx):
        return []
    if sat_states is None:
        r_s, v_s = states_at(platform.orbit, t)
    half = atp.field_of_regard_half_angle_rad
    cos_for = -1.0 if half >= np.pi else float(np.cos(half))
    mask = _kernels.fine_mask_kernel(
        np.asarray(t, float), np.ascontiguousarray(r_s), np.ascontiguousarray(v_s),
        atp.engage_range_min_m, rmax, R_EARTH + atp.earth_clearance_alt_m, cos_for,
        atp.slew_rate_limit_rad_s,
        np.ascontiguousarray(sun_hat, dtype=float) if sun_hat is not None else np.zeros((len(t), 3)),
        sun_hat is not None, bool(atp.require_target_sunlit), float(np.cos(atp.sun_exclusion_rad)),
        bool(sun_hat is not None and atp.detection_model == "passive_optical"), float(atp.detection_range_m),
        float(atp.detect_mag_limit), float(atp.track_mag_limit), float(atp.earth_background_penalty_mag),
        max(1, int(np.ceil(atp.acquisition_s / (t[1] - t[0]) - 1e-9))),
        np.ascontiguousarray(0.5 * population.diameter_m[idx], dtype=float),
        np.ascontiguousarray(population.albedo[idx], dtype=float),
        *args,
    )
    return [Window(int(idx[w.debris_index]), w.start, w.stop) for w in find_windows(mask, min_steps)]


def screen(
    platform: Platform,
    population: DebrisPopulation,
    t: np.ndarray,
    min_steps: int = 1,
) -> tuple[np.ndarray, ScreenResult, list[Window]]:
    """Coarse filter -> propagate -> fine gates -> windows.

    Returns (indices into ``population`` that passed the coarse filter,
    ScreenResult over those columns, windows whose ``debris_index`` refers to
    the original population index).
    """
    keep = np.flatnonzero(apsis_filter(platform.orbit, population.elements, platform.atp.engage_range_max_m))
    sub = population.subset(keep)
    r_s, v_s = states_at(platform.orbit, t)
    r_d, v_d = states_at(sub.elements, t, sub.ballistic_coeff, epoch=sub.epoch_s)
    res = screen_fine(t, r_s, v_s, r_d, v_d, platform.atp)
    windows = [Window(int(keep[w.debris_index]), w.start, w.stop) for w in find_windows(res.mask, min_steps)]
    return keep, res, windows
