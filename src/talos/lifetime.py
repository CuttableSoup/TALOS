"""Module 7 — orbit update and lifetime.

* ``apply_dv``: impulsive dv on mean elements (elements are treated as
  osculating for the conversion — consistent inside this mean-element model,
  ignores J2 short-period terms).
* ``orbital_lifetime``: semi-analytic decay. Orbit-averaged Gauss equations for
  tangential drag, averaged over eccentric anomaly by trapezoid quadrature,
  stepped with a per-object adaptive step. Vectorized over objects.

TODO: pymsis density table (solar-cycle dependent) instead of the static
exponential atmosphere; atmospheric co-rotation factor.
"""

from __future__ import annotations

import numpy as np

from .constants import DAY, R_EARTH, YEAR
from .elements import KeplerElements, elements_to_rv, rv_to_elements
from .propagator import _EXP_ATMOS, exponential_density, propagate

REMOVAL_PERIGEE_ALT_M = 200e3
REMOVAL_LIFETIME_S = 30 * DAY
REENTRY_ALT_M = 120e3


def apply_dv(el: KeplerElements, dv_vec: np.ndarray) -> KeplerElements:
    """Add an impulsive ECI dv [m/s] (shape el.shape + (3,)) to elements at the same epoch."""
    r, v = elements_to_rv(el)
    return rv_to_elements(r, v + np.asarray(dv_vec, dtype=float))


def move_epoch(el: KeplerElements, dt) -> KeplerElements:
    """Shift elements by dt seconds under J2 only (exactly reversible in this model)."""
    return propagate(el, dt)


def _scale_height_m(alt_m) -> np.ndarray:
    h_km = np.asarray(alt_m) / 1e3
    idx = np.clip(np.searchsorted(_EXP_ATMOS[:, 0], h_km, side="right") - 1, 0, len(_EXP_ATMOS) - 1)
    return _EXP_ATMOS[idx, 2] * 1e3


def decay_rates(a, e, ballistic_coeff, n_quad: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Orbit-averaged (da/dt, de/dt) [m/s, 1/s] from drag. B = Cd*A/m [m^2/kg]."""
    a = np.asarray(a, float)[..., None]
    e = np.asarray(e, float)[..., None]
    B = np.asarray(ballistic_coeff, float)[..., None]
    E = np.linspace(0.0, 2 * np.pi, n_quad, endpoint=False)
    cE = np.cos(E)
    rho = exponential_density(a * (1 - e * cE) - R_EARTH)
    s = np.sqrt((1 + e * cE) / (1 - e * cE))
    # d/dE forms (King-Hele); per-rev change = mean * 2pi; period = 2pi/n.
    da_dE = -B * rho * a**2 * (1 + e * cE) * s
    de_dE = -B * rho * a * (1 - e**2) * cE * s
    n = np.sqrt(3.986004418e14 / a[..., 0] ** 3)
    return da_dE.mean(-1) * n, de_dE.mean(-1) * n


def orbital_lifetime(
    a,
    e,
    ballistic_coeff,
    max_s: float = 200 * YEAR,
    reentry_alt_m: float = REENTRY_ALT_M,
    step_frac: float = 0.05,
    max_step_s: float = 30 * DAY,
) -> np.ndarray:
    """Time [s] until perigee altitude drops below ``reentry_alt_m``; inf if > ``max_s``.

    Step size: ``step_frac`` scale heights of perigee drop per step (RK2).
    """
    scalar = np.ndim(a) == 0 and np.ndim(e) == 0 and np.ndim(ballistic_coeff) == 0
    a, e, B = (np.atleast_1d(x).astype(float) for x in np.broadcast_arrays(a, e, ballistic_coeff))
    t = np.zeros_like(a)
    life = np.full_like(a, np.inf)
    alive = a * (1 - e) - R_EARTH >= reentry_alt_m
    life[~alive] = 0.0
    while np.any(alive):
        idx = np.flatnonzero(alive)
        ai, ei, Bi = a[idx], e[idx], B[idx]
        da, de = decay_rates(ai, ei, Bi)
        drp = da * (1 - ei) - ai * de
        H = _scale_height_m(ai * (1 - ei) - R_EARTH)
        dt = np.clip(step_frac * H / np.maximum(-drp, 1e-30), 1.0, max_step_s)
        da2, de2 = decay_rates(ai + da * dt, np.maximum(ei + de * dt, 0.0), Bi)
        a[idx] = ai + 0.5 * (da + da2) * dt
        e[idx] = np.maximum(ei + 0.5 * (de + de2) * dt, 0.0)
        t[idx] += dt
        down = a[idx] * (1 - e[idx]) - R_EARTH < reentry_alt_m
        life[idx[down]] = t[idx[down]]
        alive[idx[down]] = False
        alive[idx[t[idx] > max_s]] = False
    return float(life[0]) if scalar else life


def is_removed(el: KeplerElements, ballistic_coeff) -> np.ndarray:
    """Removal criterion: perigee < 200 km, or lifetime < 1 month."""
    low = el.rp - R_EARTH < REMOVAL_PERIGEE_ALT_M
    out = np.array(low, copy=True).reshape(-1)
    check = ~out
    if np.any(check):
        B = np.broadcast_to(ballistic_coeff, el.a.shape).reshape(-1)
        life = orbital_lifetime(
            el.a.reshape(-1)[check], el.e.reshape(-1)[check], B[check], max_s=2 * REMOVAL_LIFETIME_S
        )
        out[check] = life < REMOVAL_LIFETIME_S
    return out.reshape(el.a.shape)
