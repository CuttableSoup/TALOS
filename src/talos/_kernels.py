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
MAG_SUN = -26.74  # apparent V magnitude of the sun at 1 AU

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
        """Conservative coarse sieve; see screening.coarse_sieve for the bound.
        ``tc`` may be non-uniform (the interval length is used per step)."""
        Tc, N = tc.shape[0], a0.shape[0]
        keep = np.zeros(N, dtype=np.bool_)
        for j in nb.prange(N):
            prev = 0.0
            for k in range(Tc):
                a, O, w, M = _elem_at(j, tc[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, _, _, _ = _rv(a, e[j], inc[j], O, w, M)
                dx, dy, dz = rx - r_sat[k, 0], ry - r_sat[k, 1], rz - r_sat[k, 2]
                rho = math.sqrt(dx * dx + dy * dy + dz * dz)
                if rho <= rmax or (k > 0 and 0.5 * (prev + rho - vmax[j] * (tc[k] - tc[k - 1])) <= rmax):
                    keep[j] = True
                    break
                prev = rho
        return keep

    @nb.njit(cache=True, inline="always")
    def _app_mag(sx, sy, sz, dx, dy, dz, d2, rho, ux, uy, uz, radius, albedo, bg_pen):
        """Apparent magnitude of a Lambertian sphere seen from the platform, plus the
        Earth-background penalty when the sunlit Earth lies behind the target."""
        cosa = -(dx * ux + dy * uy + dz * uz) / rho  # phase angle sun-target-observer
        cosa = -1.0 if cosa < -1.0 else (1.0 if cosa > 1.0 else cosa)
        al = math.acos(cosa)
        phase = (math.pi - al) * cosa + math.sin(al)
        flux = (2.0 / (3.0 * math.pi)) * albedo * radius * radius / d2 * phase
        mag = MAG_SUN - 2.5 * math.log10(flux if flux > 1e-300 else 1e-300)
        # ray sat + s*d, s > 1: does it hit Earth, and is that point sunlit?
        b = sx * dx + sy * dy + sz * dz
        c = sx * sx + sy * sy + sz * sz - R_EARTH * R_EARTH
        disc = b * b - d2 * c
        if disc > 0.0:
            s1 = (-b - math.sqrt(disc)) / d2
            if s1 > 1.0:
                if (sx + s1 * dx) * ux + (sy + s1 * dy) * uy + (sz + s1 * dz) * uz > 0.0:
                    mag += bg_pen
        return mag

    @nb.njit(parallel=True, cache=True)
    def fine_mask_kernel(t, r_sat, v_sat, rmin, rmax, clear_r, cos_for, slew_max,
                         sun, use_sun, need_lit, cos_excl,
                         use_det, det_range, det_lim, trk_lim, bg_pen, acq_steps, radius, albedo,
                         a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        """(T, n) engagement mask: range, Earth-limb LOS, field of regard, slew rate and,
        with ``use_sun``, target sunlit (cylindrical shadow) + sun-exclusion cone.
        With ``use_det`` (requires use_sun) a per-object detection/track state machine
        runs along time: detect (mag <= det_lim) for ``acq_steps`` consecutive steps to
        establish a track, keep it while mag <= trk_lim; engagement requires a track.
        A target detectable at the first step is assumed already tracked (chunk start).
        ``sun``: (T, 3) unit vectors (ignored unless use_sun)."""
        T, N = t.shape[0], a0.shape[0]
        mask = np.zeros((T, N), dtype=np.bool_)
        for j in nb.prange(N):
            tracked = False
            run = 0
            for k in range(T):
                a, O, w, M = _elem_at(j, t[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, vx, vy, vz = _rv(a, e[j], inc[j], O, w, M)
                sx, sy, sz = r_sat[k, 0], r_sat[k, 1], r_sat[k, 2]
                dx, dy, dz = rx - sx, ry - sy, rz - sz
                d2 = dx * dx + dy * dy + dz * dz
                rho = math.sqrt(d2)
                far = rho > (det_range if use_det else rmax)
                if far:
                    tracked = False
                    run = 0
                    continue
                # Earth-limb clearance of the segment sat -> target
                s = -(sx * dx + sy * dy + sz * dz) / d2
                s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
                px, py, pz = sx + s * dx, sy + s * dy, sz + s * dz
                los_ok = math.sqrt(px * px + py * py + pz * pz) > clear_r
                excl = False
                lit = True
                qx, qy, qz = 0.0, 0.0, 0.0
                if use_sun:
                    qx, qy, qz = sun[k, 0], sun[k, 1], sun[k, 2]
                    excl = (dx * qx + dy * qy + dz * qz) / rho > cos_excl
                    dd = rx * qx + ry * qy + rz * qz
                    lit = not (dd < 0.0 and rx * rx + ry * ry + rz * rz - dd * dd < R_EARTH * R_EARTH)
                if use_det:
                    det_ok = False
                    trk_ok = False
                    if los_ok and lit and not excl:
                        mag = _app_mag(sx, sy, sz, dx, dy, dz, d2, rho, qx, qy, qz, radius[j], albedo[j], bg_pen)
                        det_ok = mag <= det_lim
                        trk_ok = mag <= trk_lim
                    if tracked and not trk_ok:
                        tracked = False
                        run = 0
                    if not tracked:
                        if det_ok:
                            run = acq_steps if k == 0 else run + 1
                        else:
                            run = 0
                        if run >= acq_steps:
                            tracked = True
                    if not tracked:
                        continue
                if rho < rmin or rho > rmax or not los_ok or excl or (need_lit and not lit):
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
                mask[k, j] = True
        return mask


def element_args(el, ballistic_coeff=None, epoch=None):
    """Flatten elements + J2/drag rates into the kernel argument tuple.

    Kernels evaluate x(t) = x0 + xdot * t at sim time t. Elements referring to sim
    time ``epoch`` are re-based to t = 0 (x0 = x_ep - xdot * epoch), which is exact
    for the linear secular model; drag rates use the density at the epoch."""
    from .propagator import drag_da_dt, j2_secular_rates

    raan_dot, argp_dot, M_dot = j2_secular_rates(el)
    a_ep = np.atleast_1d(np.asarray(el.a, float))
    ep = np.zeros_like(a_ep) if epoch is None else np.broadcast_to(np.asarray(epoch, float), a_ep.shape)
    adot = (np.zeros_like(a_ep) if ballistic_coeff is None
            else np.broadcast_to(drag_da_dt(a_ep, ballistic_coeff, ep), a_ep.shape))
    f = lambda x: np.ascontiguousarray(np.broadcast_to(np.atleast_1d(x), a_ep.shape), dtype=float)
    base = lambda x, xdot: f(np.asarray(x, float) - np.asarray(xdot, float) * ep)
    return (base(a_ep, adot), f(adot), f(el.e), f(el.i), base(el.raan, raan_dot), f(raan_dot),
            base(el.argp, argp_dot), f(argp_dot), base(el.M, M_dot), f(M_dot))
