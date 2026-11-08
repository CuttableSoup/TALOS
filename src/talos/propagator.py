"""Module 2 — mean-element propagator: J2 secular rates + first-order drag.

This is the fast vectorized path used for screening thousands of objects.
TODO:
  * Real TLEs via ``sgp4.api.SatrecArray`` (optional extra ``talos[tle]``).
  * Replace the exponential atmosphere with a pymsis density table (module 7).
  * Proper semi-analytic decay (integrate da/dt, de/dt) for long horizons;
    the first-order drag here is only valid for dt << decay timescale.
"""

from __future__ import annotations

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH, SSO_RAAN_RATE
from .elements import KeplerElements, elements_to_rv

# Vallado exponential atmosphere (Fundamentals of Astrodynamics, Table 8-4).
# Columns: base altitude [km], nominal density [kg/m^3], scale height [km].
# Placeholder until the pymsis table is wired in.
_EXP_ATMOS = np.array(
    [
        [100, 5.297e-7, 5.877],
        [110, 9.661e-8, 7.263],
        [120, 2.438e-8, 9.473],
        [130, 8.484e-9, 12.636],
        [140, 3.845e-9, 16.149],
        [150, 2.070e-9, 22.523],
        [180, 5.464e-10, 29.740],
        [200, 2.789e-10, 37.105],
        [250, 7.248e-11, 45.546],
        [300, 2.418e-11, 53.628],
        [350, 9.518e-12, 53.298],
        [400, 3.725e-12, 58.515],
        [450, 1.585e-12, 60.828],
        [500, 6.967e-13, 63.822],
        [600, 1.454e-13, 71.835],
        [700, 3.614e-14, 88.667],
        [800, 1.170e-14, 124.64],
        [900, 5.245e-15, 181.05],
        [1000, 3.019e-15, 268.00],
    ]
)


def j2_secular_rates(el: KeplerElements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Secular J2 rates (raan_dot, argp_dot, M_dot) in rad/s."""
    n = el.n
    k = 0.75 * n * J2 * (R_EARTH / el.p) ** 2
    ci = np.cos(el.i)
    raan_dot = -2.0 * k * ci
    argp_dot = k * (5.0 * ci**2 - 1.0)
    M_dot = n + k * np.sqrt(1.0 - el.e**2) * (3.0 * ci**2 - 1.0)
    return raan_dot, argp_dot, M_dot


def sso_inclination(a, e=0.0) -> np.ndarray:
    """Inclination [rad] giving sun-synchronous RAAN precession."""
    a = np.asarray(a, dtype=float)
    p = a * (1.0 - np.asarray(e) ** 2)
    n = np.sqrt(MU_EARTH / a**3)
    cos_i = -SSO_RAAN_RATE / (1.5 * n * J2 * (R_EARTH / p) ** 2)
    return np.arccos(cos_i)


def exponential_density(alt_m) -> np.ndarray:
    """Atmospheric density [kg/m^3] at geodetic-ish altitude [m] (spherical Earth)."""
    h_km = np.asarray(alt_m, dtype=float) / 1e3
    idx = np.clip(np.searchsorted(_EXP_ATMOS[:, 0], h_km, side="right") - 1, 0, len(_EXP_ATMOS) - 1)
    h0, rho0, H = _EXP_ATMOS[idx, 0], _EXP_ATMOS[idx, 1], _EXP_ATMOS[idx, 2]
    return rho0 * np.exp(-(h_km - h0) / H)


_ATMOSPHERE = None  # talos.atmosphere.Atmosphere; None -> static exponential table


def set_atmosphere(atm) -> None:
    """Install the density model used by drag and lifetime (None: exponential)."""
    global _ATMOSPHERE
    _ATMOSPHERE = atm


def get_atmosphere():
    return _ATMOSPHERE


def density(alt_m, t_s=0.0) -> np.ndarray:
    """Density [kg/m^3] from the installed atmosphere; ``t_s`` = s from scenario epoch."""
    if _ATMOSPHERE is None:
        return exponential_density(alt_m)
    return _ATMOSPHERE.density(alt_m, t_s)


def drag_da_dt(a, ballistic_coeff, t_s=0.0) -> np.ndarray:
    """Near-circular orbit-averaged decay rate da/dt [m/s] at time ``t_s``.

    ``ballistic_coeff`` is Cd*A/m [m^2/kg] (= Cd / areal density).
    """
    a = np.asarray(a, dtype=float)
    rho = density(a - R_EARTH, t_s)
    return -rho * np.asarray(ballistic_coeff) * np.sqrt(MU_EARTH * a)


def propagate(el: KeplerElements, dt, ballistic_coeff=None, t_epoch=0.0) -> KeplerElements:
    """Advance mean elements by ``dt`` seconds (broadcast against element shape).

    J2 rates are evaluated at the initial elements. Drag, if given, decrements
    ``a`` to first order only, with density at the element epoch ``t_epoch``.
    """
    dt = np.asarray(dt, dtype=float)
    raan_dot, argp_dot, M_dot = j2_secular_rates(el)
    a = el.a
    if ballistic_coeff is not None:
        a = np.maximum(a + drag_da_dt(el.a, ballistic_coeff, t_epoch) * dt, R_EARTH)
    return KeplerElements(
        a=a,
        e=el.e,
        i=el.i,
        raan=np.mod(el.raan + raan_dot * dt, 2 * np.pi),
        argp=np.mod(el.argp + argp_dot * dt, 2 * np.pi),
        M=np.mod(el.M + M_dot * dt, 2 * np.pi),
    )


_KERNEL_MIN_SIZE = 20_000  # T*N above which the numba kernel beats numpy


def states_at(el: KeplerElements, t, ballistic_coeff=None, use_kernel: bool | None = None, epoch=None):
    """ECI states at sim times ``t`` [s], shape ``(len(t),) + el.shape + (3,)``.

    ``epoch`` (scalar or el.shape) is the sim time the elements refer to (default 0).
    Large 1-D element sets use the multithreaded numba kernel when available.
    """
    from . import _kernels

    t = np.asarray(t, dtype=float).reshape(-1)
    if use_kernel is None:
        use_kernel = _kernels.HAVE_NUMBA and len(el.shape) == 1 and t.size * el.shape[0] >= _KERNEL_MIN_SIZE
    if use_kernel:
        return _kernels.states_kernel(t, *_kernels.element_args(el, ballistic_coeff, epoch))
    t = t.reshape((-1,) + (1,) * len(el.shape))
    ep = 0.0 if epoch is None else np.asarray(epoch, float)
    return elements_to_rv(propagate(el, t - ep, ballistic_coeff, ep))
