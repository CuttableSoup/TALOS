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

    @property
    def mask(self) -> np.ndarray:
        return self.in_range & self.los_ok & self.in_for & self.slew_ok


def screen_fine(t, r_sat, v_sat, r_deb, v_deb, atp: ATPConfig) -> ScreenResult:
    """Gate checks. ``r_sat``/``v_sat``: (T, 3); ``r_deb``/``v_deb``: (T, N, 3)."""
    r_sat = r_sat[:, None, :]
    v_sat = v_sat[:, None, :]
    r_rel = r_deb - r_sat
    v_rel = v_deb - v_sat
    rng = np.linalg.norm(r_rel, axis=-1)
    slew = los_angular_rate(r_rel, v_rel)
    return ScreenResult(
        t=np.asarray(t),
        range_m=rng,
        slew_rate=slew,
        in_range=(rng >= atp.engage_range_min_m) & (rng <= atp.engage_range_max_m),
        los_ok=los_clear(r_sat, r_deb, R_EARTH + atp.earth_clearance_alt_m),
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
    r_d, _ = states_at(deb, tc, ballistic_coeff)
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
) -> list[Window]:
    """Memory-bounded screening: windows only, objects processed in batches.

    ``candidates`` optionally restricts to these population indices (e.g. not yet removed).
    ``coarse_step_s`` enables the conservative coarse sieve (None = fine-step everything).
    Window ``debris_index`` refers to the original population index.
    """
    idx = np.arange(len(population)) if candidates is None else np.asarray(candidates)
    rmax = platform.atp.engage_range_max_m
    idx = idx[apsis_filter(platform.orbit, population.elements[idx], rmax)]
    if coarse_step_s is not None and len(idx) and len(t) > 1:
        keep = np.zeros(len(idx), dtype=bool)
        for k in range(0, len(idx), 4 * batch):
            cols = idx[k : k + 4 * batch]
            keep[k : k + 4 * batch] = coarse_sieve(
                platform.orbit, population.elements[cols], population.ballistic_coeff[cols],
                t, rmax, coarse_step_s,
            )
        idx = idx[keep]
    r_s, v_s = states_at(platform.orbit, t)
    out: list[Window] = []
    for k in range(0, len(idx), batch):
        cols = idx[k : k + batch]
        sub = population.subset(cols)
        r_d, v_d = states_at(sub.elements, t, sub.ballistic_coeff)
        mask = screen_fine(t, r_s, v_s, r_d, v_d, platform.atp).mask
        out += [Window(int(cols[w.debris_index]), w.start, w.stop) for w in find_windows(mask, min_steps)]
    return out


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
    r_d, v_d = states_at(sub.elements, t, sub.ballistic_coeff)
    res = screen_fine(t, r_s, v_s, r_d, v_d, platform.atp)
    windows = [Window(int(keep[w.debris_index]), w.start, w.stop) for w in find_windows(res.mask, min_steps)]
    return keep, res, windows
