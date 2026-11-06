"""Numba multithreaded kernels for the hot screening path.

Each kernel fuses J2-secular propagation, the Kepler solve and the geometry
gates into one parallel loop over objects (``prange``), so no (T, N, 3)
temporaries are allocated. Numba is optional: ``HAVE_NUMBA`` is False when it is
missing or when ``TALOS_NUMBA=0``, and callers fall back to the numpy path.

Thread count: numba uses all logical cores by default; set NUMBA_NUM_THREADS
(the sweep driver does this per worker process).
"""

from __future__ import annotations

import math
import os

import numpy as np

from .constants import MU_EARTH, R_EARTH

try:  # pragma: no cover - import guard
    if os.environ.get("TALOS_NUMBA", "1") == "0":
        raise ImportError("disabled by TALOS_NUMBA=0")
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    nb = None
    HAVE_NUMBA = False

TWO_PI = 2.0 * math.pi

if HAVE_NUMBA:

    @nb.njit(cache=True, inline="always")
    def _rv(a, e, inc, raan, argp, M):
        M = M % TWO_PI
        E = M if e < 0.8 else math.pi
        for _ in range(50):
            dE = -(E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E += dE
            if abs(dE) < 1e-12:
                break
        cE, sE = math.cos(E), math.sin(E)
        sq = math.sqrt(1.0 - e * e)
        r = a * (1.0 - e * cE)
        x = a * (cE - e)
        y = a * sq * sE
        k = math.sqrt(MU_EARTH * a) / r
        vx = -k * sE
        vy = k * sq * cE
        cO, sO = math.cos(raan), math.sin(raan)
        cw, sw = math.cos(argp), math.sin(argp)
        ci, si = math.cos(inc), math.sin(inc)
        P0, P1, P2 = cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si
        Q0, Q1, Q2 = -cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si
        return (x * P0 + y * Q0, x * P1 + y * Q1, x * P2 + y * Q2,
                vx * P0 + vy * Q0, vx * P1 + vy * Q1, vx * P2 + vy * Q2)

    @nb.njit(cache=True, inline="always")
    def _elem_at(j, tk, a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        a = a0[j] + adot[j] * tk
        if a < R_EARTH:
            a = R_EARTH
        return a, raan0[j] + raan_dot[j] * tk, argp0[j] + argp_dot[j] * tk, M0[j] + M_dot[j] * tk

    @nb.njit(parallel=True, cache=True)
    def states_kernel(t, a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        T, N = t.shape[0], a0.shape[0]
        r = np.empty((T, N, 3))
        v = np.empty((T, N, 3))
        for j in nb.prange(N):
            for k in range(T):
                a, O, w, M = _elem_at(j, t[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, vx, vy, vz = _rv(a, e[j], inc[j], O, w, M)
                r[k, j, 0], r[k, j, 1], r[k, j, 2] = rx, ry, rz
                v[k, j, 0], v[k, j, 1], v[k, j, 2] = vx, vy, vz
        return r, v

    @nb.njit(parallel=True, cache=True)
    def sieve_kernel(tc, r_sat, vmax, rmax, a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        """Conservative coarse sieve; see screening.coarse_sieve for the bound."""
        Tc, N = tc.shape[0], a0.shape[0]
        h = tc[1] - tc[0] if Tc > 1 else 0.0
        keep = np.zeros(N, dtype=np.bool_)
        for j in nb.prange(N):
            prev = 0.0
            for k in range(Tc):
                a, O, w, M = _elem_at(j, tc[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, _, _, _ = _rv(a, e[j], inc[j], O, w, M)
                dx, dy, dz = rx - r_sat[k, 0], ry - r_sat[k, 1], rz - r_sat[k, 2]
                rho = math.sqrt(dx * dx + dy * dy + dz * dz)
                if rho <= rmax or (k > 0 and 0.5 * (prev + rho - vmax[j] * h) <= rmax):
                    keep[j] = True
                    break
                prev = rho
        return keep

    @nb.njit(parallel=True, cache=True)
    def fine_mask_kernel(t, r_sat, v_sat, rmin, rmax, clear_r, cos_for, slew_max,
                         a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        """(T, n) engagement mask: range, Earth-limb LOS, field of regard, slew rate."""
        T, N = t.shape[0], a0.shape[0]
        mask = np.zeros((T, N), dtype=np.bool_)
        for j in nb.prange(N):
            for k in range(T):
                a, O, w, M = _elem_at(j, t[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, vx, vy, vz = _rv(a, e[j], inc[j], O, w, M)
                sx, sy, sz = r_sat[k, 0], r_sat[k, 1], r_sat[k, 2]
                dx, dy, dz = rx - sx, ry - sy, rz - sz
                d2 = dx * dx + dy * dy + dz * dz
                rho = math.sqrt(d2)
                if rho < rmin or rho > rmax:
                    continue
                ux, uy, uz = v_sat[k, 0], v_sat[k, 1], v_sat[k, 2]
                # field of regard about the ram direction
                if cos_for > -1.0:
                    un = math.sqrt(ux * ux + uy * uy + uz * uz)
                    if (dx * ux + dy * uy + dz * uz) / (rho * un) < cos_for:
                        continue
                # LOS angular rate |r x v| / r^2
                wx, wy, wz = vx - ux, vy - uy, vz - uz
                cx = dy * wz - dz * wy
                cy = dz * wx - dx * wz
                cz = dx * wy - dy * wx
                if math.sqrt(cx * cx + cy * cy + cz * cz) / d2 > slew_max:
                    continue
                # Earth-limb clearance of the segment sat -> target
                s = -(sx * dx + sy * dy + sz * dz) / d2
                s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
                px, py, pz = sx + s * dx, sy + s * dy, sz + s * dz
                if math.sqrt(px * px + py * py + pz * pz) <= clear_r:
                    continue
                mask[k, j] = True
        return mask


def element_args(el, ballistic_coeff=None):
    """Flatten elements + J2/drag rates into the kernel argument tuple."""
    from .propagator import drag_da_dt, j2_secular_rates

    raan_dot, argp_dot, M_dot = j2_secular_rates(el)
    a = np.ascontiguousarray(np.atleast_1d(el.a), dtype=float)
    adot = np.zeros_like(a) if ballistic_coeff is None else np.atleast_1d(drag_da_dt(el.a, ballistic_coeff)) + 0.0 * a
    f = lambda x: np.ascontiguousarray(np.broadcast_to(np.atleast_1d(x), a.shape), dtype=float)
    return (a, f(adot), f(el.e), f(el.i), f(el.raan), f(raan_dot), f(el.argp), f(argp_dot), f(el.M), f(M_dot))
