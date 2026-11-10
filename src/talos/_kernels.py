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

from .constants import MU_EARTH, R_EARTH, YEAR as YEAR_S

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
    def sieve_kernel(tc, r_sat, vsat, vdeb, sel, rmax, a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot,
                     M0, M_dot):
        """Conservative coarse sieve; see screening.coarse_sieve for the bound.
        ``tc`` may be non-uniform (the interval length is used per step). Objects are
        ``sel`` (indices into the element arrays); range-rate bound vsat + vdeb[j]."""
        Tc, N = tc.shape[0], sel.shape[0]
        keep = np.zeros(N, dtype=np.bool_)
        for jj in nb.prange(N):
            j = sel[jj]
            vmax = vsat + vdeb[j]
            prev = 0.0
            for k in range(Tc):
                a, O, w, M = _elem_at(j, tc[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, _, _, _ = _rv(a, e[j], inc[j], O, w, M)
                dx, dy, dz = rx - r_sat[k, 0], ry - r_sat[k, 1], rz - r_sat[k, 2]
                rho = math.sqrt(dx * dx + dy * dy + dz * dz)
                if rho <= rmax or (k > 0 and 0.5 * (prev + rho - vmax * (tc[k] - tc[k - 1])) <= rmax):
                    keep[jj] = True
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
                         use_det, det_range, det_lim, trk_lim, bg_pen, acq_steps, radius, albedo, vsat, vdeb,
                         want_mask, min_steps, sel, a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        """(T, n) engagement mask: range, Earth-limb LOS, field of regard, slew rate and,
        with ``use_sun``, target sunlit (cylindrical shadow) + sun-exclusion cone.
        With ``use_det`` (requires use_sun) a per-object detection/track state machine
        runs along time: detect (mag <= det_lim) for ``acq_steps`` consecutive steps to
        establish a track, keep it while mag <= trk_lim; engagement requires a track.
        A target detectable at the first step is assumed already tracked (chunk start).
        ``sun``: (T, 3) unit vectors (ignored unless use_sun).
        Objects are ``sel`` (indices into the element / radius / albedo / vdeb arrays); mask
        column jj is object sel[jj]. Range-rate bound vsat + vdeb[j]. Out of range,
        steps that cannot be back within range yet are skipped: exact, since every skipped
        step would also be 'far' (mask False, track reset)."""
        T, N = t.shape[0], sel.shape[0]
        # Windows are recorded per object as they are found: up to RUNS per object per call
        # (runs shorter than min_steps dropped); cnt > RUNS flags an overflow (caller falls back
        # to the mask). The (T, N) mask is only filled with ``want_mask``.
        RUNS = 64
        mask = np.zeros((T, N) if want_mask else (1, 1), dtype=np.bool_)
        cnt = np.zeros(N, np.int64)
        r0 = np.empty((N, RUNS), np.int64)
        r1 = np.empty((N, RUNS), np.int64)
        for jj in nb.prange(N):
            j = sel[jj]
            vmax = vsat + vdeb[j]
            last_true = -2
            start = 0
            tracked = False
            run = 0
            nxt = 0
            while nxt < T:
                k = nxt
                nxt = k + 1
                a, O, w, M = _elem_at(j, t[k], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, vx, vy, vz = _rv(a, e[j], inc[j], O, w, M)
                sx, sy, sz = r_sat[k, 0], r_sat[k, 1], r_sat[k, 2]
                dx, dy, dz = rx - sx, ry - sy, rz - sz
                d2 = dx * dx + dy * dy + dz * dz
                rho = math.sqrt(d2)
                r_far = det_range if use_det else rmax
                if rho > r_far:
                    tracked = False
                    run = 0
                    gap = (rho - r_far) / vmax * (1.0 - 1e-9)  # earliest possible re-entry
                    tk = t[k]
                    while nxt < T and t[nxt] - tk < gap:
                        nxt += 1
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
                if want_mask:
                    mask[k, jj] = True
                if last_true != k - 1:  # a new run starts: close the previous one
                    if last_true >= 0 and last_true + 1 - start >= min_steps:
                        if cnt[jj] < RUNS:
                            r0[jj, cnt[jj]] = start
                            r1[jj, cnt[jj]] = last_true + 1
                        cnt[jj] += 1
                    start = k
                last_true = k
            if last_true >= 0 and last_true + 1 - start >= min_steps:
                if cnt[jj] < RUNS:
                    r0[jj, cnt[jj]] = start
                    r1[jj, cnt[jj]] = last_true + 1
                cnt[jj] += 1
        return mask, cnt, r0, r1


if HAVE_NUMBA:

    @nb.njit(cache=True, inline="always")
    def _log_rho_table(h_km, f107, A, F, L):
        """Log-bilinear MSIS table lookup; same rules as Atmosphere.density."""
        nA, nF = A.shape[0], F.shape[0]
        i = min(max(np.searchsorted(A, h_km) - 1, 0), nA - 2)
        wa = max((h_km - A[i]) / (A[i + 1] - A[i]), 0.0)  # log-linear extrapolation above the top
        fc = min(max(f107, F[0]), F[nF - 1])
        j = min(max(np.searchsorted(F, fc) - 1, 0), nF - 2)
        wf = (fc - F[j]) / (F[j + 1] - F[j])
        return ((1 - wa) * (1 - wf) * L[i, j] + wa * (1 - wf) * L[i + 1, j]
                + (1 - wa) * wf * L[i, j + 1] + wa * wf * L[i + 1, j + 1])

    @nb.njit(parallel=True, cache=True)
    def density_table(h_km, f107, A, F, L):
        """Vectorised Atmosphere.density for the MSIS table (flat inputs)."""
        out = np.empty(h_km.shape[0])
        for i in nb.prange(h_km.shape[0]):
            out[i] = math.exp(_log_rho_table(h_km[i], f107[i], A, F, L))
        return out

    @nb.njit(cache=True, inline="always")
    def _rates1(a, e, B, f107, A, F, L, n_quad):
        sa = 0.0
        se = 0.0
        for q in range(n_quad):
            cE = math.cos(TWO_PI * q / n_quad)
            rho = math.exp(_log_rho_table((a * (1 - e * cE) - R_EARTH) / 1e3, f107, A, F, L))
            s = math.sqrt((1 + e * cE) / (1 - e * cE))
            sa += rho * (1 + e * cE) * s
            se += rho * cE * s
        n = math.sqrt(MU_EARTH / a ** 3)
        return -B * a * a * sa / n_quad * n, -B * a * (1 - e * e) * se / n_quad * n

    @nb.njit(parallel=True, cache=True)
    def decay_rates_table(a, e, B, f107, A, F, L, n_quad):
        """Orbit-averaged drag (da/dt, de/dt) per object with the MSIS table density;
        fused version of lifetime.decay_rates (same quadrature)."""
        N = a.shape[0]
        da = np.empty(N)
        de = np.empty(N)
        for k in nb.prange(N):
            da[k], de[k] = _rates1(a[k], e[k], B[k], f107[k], A, F, L, n_quad)
        return da, de


if HAVE_NUMBA:

    @nb.njit(cache=True)
    def mask_runs(mask, min_steps):
        """Contiguous True runs along axis 0 of a (T, N) mask -> (col, start, stop) arrays,
        sorted by (col, start); runs shorter than ``min_steps`` are dropped. Scans rows
        (contiguous in memory) with an open-run start per column."""
        T, N = mask.shape
        open_at = np.full(N, -1, np.int64)
        cap = 64
        col = np.empty(cap, np.int64)
        st = np.empty(cap, np.int64)
        sp = np.empty(cap, np.int64)
        n = 0
        for k in range(T + 1):
            for j in range(N):
                on = k < T and mask[k, j]
                if on:
                    if open_at[j] < 0:
                        open_at[j] = k
                elif open_at[j] >= 0:
                    if k - open_at[j] >= min_steps:
                        if n == cap:
                            cap *= 2
                            col2, st2, sp2 = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64)
                            col2[:n], st2[:n], sp2[:n] = col[:n], st[:n], sp[:n]
                            col, st, sp = col2, st2, sp2
                        col[n], st[n], sp[n] = j, open_at[j], k
                        n += 1
                    open_at[j] = -1
        order = np.argsort(col[:n] * (T + 1) + st[:n])
        return col[:n][order], st[:n][order], sp[:n][order]


if HAVE_NUMBA:

    @nb.njit(cache=True, inline="always")
    def _f107(t_s, act_const, f_const, epoch_year, cmin, cmax, period, fmin, fmax):
        """Atmosphere.f107 (constant or cosine solar cycle)."""
        if act_const:
            return f_const
        year = epoch_year + t_s / YEAR_S
        x = ((year - cmin) / period) % 1.0
        xr = (cmax - cmin) / period
        if x < xr:
            g = 0.5 * (1.0 - math.cos(math.pi * x / xr))
        else:
            g = 0.5 * (1.0 + math.cos(math.pi * (x - xr) / (1.0 - xr)))
        return fmin + (fmax - fmin) * g

    @nb.njit(cache=True, inline="always")
    def _shell_frac(a, e, lo, hi, n_quad, out):
        """Time fraction of the orbit inside each [lo, hi) altitude band, added into ``out``."""
        wsum = 0.0
        for q in range(n_quad):
            wsum += 1.0 - e * math.cos(TWO_PI * q / n_quad)
        for q in range(n_quad):
            cE = math.cos(TWO_PI * q / n_quad)
            alt = a * (1 - e * cE) - R_EARTH
            w = (1.0 - e * cE) / wsum
            for k in range(lo.shape[0]):
                if lo[k] <= alt < hi[k]:
                    out[k] += w

    @nb.njit(parallel=True, cache=True)
    def lifetime_kernel(a0, e0, B, t0, max_s, reentry_alt, step_frac, max_step, A, F, L,
                        act_const, f_const, epoch_year, cmin, cmax, period, fmin, fmax, lo, hi):
        """lifetime.orbital_lifetime with the MSIS table atmosphere, one object per thread:
        same RK2 step rule (``step_frac`` scale heights of perigee drop, clipped [1 s, max_step])
        and the same shell-residence rule (8 sub-step midpoints). Returns (life, residence)."""
        N = a0.shape[0]
        K = lo.shape[0]
        life = np.full(N, np.inf)
        res = np.zeros((N, K))
        dh = 5e3
        hmin = 1e300
        hmax = -1e300
        for k in range(K):
            hmin = min(hmin, lo[k])
            hmax = max(hmax, hi[k])
        for j in nb.prange(N):
            a, e, Bj = a0[j], e0[j], B[j]
            if a * (1 - e) - R_EARTH < reentry_alt:
                life[j] = 0.0
                continue
            t = 0.0
            frac = np.zeros(K)
            while True:
                ti = t0[j] + t
                f = _f107(ti, act_const, f_const, epoch_year, cmin, cmax, period, fmin, fmax)
                da, de = _rates1(a, e, Bj, f, A, F, L, 64)
                drp = da * (1 - e) - a * de
                hp = (a * (1 - e) - R_EARTH) / 1e3
                H = 2 * dh / max(_log_rho_table(hp - dh / 1e3, f, A, F, L)
                                 - _log_rho_table(hp + dh / 1e3, f, A, F, L), 1e-9)
                dt = min(max(step_frac * H / max(-drp, 1e-30), 1.0), max_step)
                if K > 0:
                    a1, e1 = a + da * dt, max(e + de * dt, 0.0)
                    # skip the quadrature when the swept altitude range misses every shell
                    if min(a * (1 - e), a1 * (1 - e1)) - R_EARTH < hmax and                             max(a * (1 + e), a1 * (1 + e1)) - R_EARTH >= hmin:
                        for k in range(K):
                            frac[k] = 0.0
                        for u in range(8):
                            w = (u + 0.5) / 8
                            _shell_frac(a + da * dt * w, max(e + de * dt * w, 0.0), lo, hi, 64, frac)
                        span = min(dt, max(max_s - t, 0.0))
                        for k in range(K):
                            res[j, k] += frac[k] / 8 * span
                f2 = _f107(ti + dt, act_const, f_const, epoch_year, cmin, cmax, period, fmin, fmax)
                da2, de2 = _rates1(a + da * dt, max(e + de * dt, 0.0), Bj, f2, A, F, L, 64)
                a = a + 0.5 * (da + da2) * dt
                e = max(e + 0.5 * (de + de2) * dt, 0.0)
                t += dt
                if a * (1 - e) - R_EARTH < reentry_alt:
                    life[j] = t
                    break
                if t > max_s:
                    break
        return life, res


if HAVE_NUMBA:

    @nb.njit(cache=True)
    def predict_kernel(tp, r_sat, t1, js, phi_k, th, opt, cm0, plasma, pexp, eta1_cos, mu,
                       a0, adot, e, inc, raan0, raan_dot, argp0, argp_dot, M0, M_dot):
        """Batched deterministic burst prediction (engagement.predict_bursts) for candidates
        ``js`` (indices into the element arrays), pulses ``tp`` with sat positions ``r_sat``
        (P, 3); pulse p is active for candidate c while tp[p] < t1[c]. Fluence phi = phi_k / R^2.
        Returns (dv_vec (n, 3), dv_retrograde (n,), n_fired (n,))."""
        P, n = tp.shape[0], js.shape[0]
        dv = np.zeros((n, 3))
        retro = np.zeros(n)
        nf = np.zeros(n, np.int64)
        lr = math.log(opt / th)
        for c in range(n):
            j = js[c]
            for p in range(P):
                if not tp[p] < t1[c] - 1e-9:
                    break
                a, O, w, M = _elem_at(j, tp[p], a0, adot, raan0, raan_dot, argp0, argp_dot, M0, M_dot)
                rx, ry, rz, vx, vy, vz = _rv(a, e[j], inc[j], O, w, M)
                dx, dy, dz = rx - r_sat[p, 0], ry - r_sat[p, 1], rz - r_sat[p, 2]
                R = math.sqrt(dx * dx + dy * dy + dz * dz)
                phi = phi_k / (R * R)
                if phi < th:
                    continue
                if not plasma:
                    cm = cm0
                elif phi <= opt:
                    cm = cm0 * math.log(phi / th) / lr
                else:
                    cm = cm0 * (opt / phi) ** pexp
                mag = eta1_cos * cm * phi / mu[c] / R
                ddx, ddy, ddz = mag * dx, mag * dy, mag * dz
                vn = math.sqrt(vx * vx + vy * vy + vz * vz)
                dv[c, 0] += ddx
                dv[c, 1] += ddy
                dv[c, 2] += ddz
                retro[c] -= (ddx * vx + ddy * vy + ddz * vz) / vn
                nf[c] += 1
        return dv, retro, nf


if HAVE_NUMBA:
    # --- resource state machine (resources.ResourceState), same operation order ---
    # K: float64 constants, see resources.ResourceState._kernel_consts for the layout.
    _SIGMA = 5.670374419e-8
    _H_STEP = 5.0
    _PROBE_S = 10.0

    @nb.njit(cache=True, inline="always")
    def _r_temp(H, K):
        if H <= K[0]:
            return K[3] + H / K[2]
        if H <= K[1]:
            return K[4]
        return K[4] + (H - K[1]) / K[2]

    @nb.njit(cache=True, inline="always")
    def _r_prad(temp, alb, K):
        t_rad = temp - K[7]
        per_side = K[8] * _SIGMA * t_rad ** 4.0 - K[8] * K[9] * K[10] - K[11] * alb - K[12]
        return max(0.0, K[6] * (K[13] * K[14] * per_side))

    @nb.njit(cache=True)
    def r_evolve(H, p_in, dt, alb, goal, has_goal, K):
        """ResourceState._evolve -> (H, elapsed, time held at the floor)."""
        H1, H2 = K[0], K[1]
        t = 0.0
        if has_goal and H == goal:
            return H, 0.0, 0.0
        while t < dt - 1e-12:
            rem = dt - t
            r = p_in - _r_prad(_r_temp(H, K), alb, K)
            if abs(r) < 1e-9:
                return H, dt, 0.0
            if H <= 0.0 and r < 0.0:
                return 0.0, dt, rem
            if H2 > H1 and ((H1 < H < H2) or (H == H1 and r > 0) or (H == H2 and r < 0)):
                edge = H2 if r > 0 else H1
                h = min(rem, (edge - H) / r)
                Hn = edge if h < rem else H + r * h
            else:
                h = min(rem, _H_STEP)
                Hm = max(0.0, H + 0.5 * h * r)
                Hn = H + h * (p_in - _r_prad(_r_temp(Hm, K), alb, K))
                if (H - H1) * (Hn - H1) < 0:
                    h *= (H1 - H) / (Hn - H)
                    Hn = H1
                elif (H - H2) * (Hn - H2) < 0:
                    h *= (H2 - H) / (Hn - H)
                    Hn = H2
                if Hn < 0.0:
                    h *= H / (H - Hn)
                    Hn = 0.0
            if has_goal and (H - goal) * (Hn - goal) <= 0 and Hn != H:
                return goal, t + h * (goal - H) / (Hn - H), 0.0
            H, t = Hn, t + h
        return H, t, 0.0

    @nb.njit(cache=True, inline="always")
    def _r_seg(t, K, has_env, lit, af, albw, step):
        """(lit, p_charge, albedo, segment end) at time t."""
        if not has_env:
            return True, K[20] - K[17], 0.0, np.inf
        n = lit.shape[0]
        k = int(min(max(t // step, 0.0), n - 1))
        end = np.inf if k == n - 1 else (k + 1) * step
        return lit[k], K[20] * af[k] - K[17], albw[k], end

    @nb.njit(cache=True, inline="always")
    def _r_batt_loss(pc, K):
        return (1.0 - K[16]) * max(K[18] - pc, 0.0)

    @nb.njit(cache=True)
    def r_advance(t, heat, soc, heater_j, brown, disch, t_new, K, has_env, lit, af, albw, step):
        """ResourceState.advance -> (t, heat_j, soc_j, heater_j, brownout_s, discharge_j)."""
        cap = K[15]
        while t < t_new - 1e-12:
            lt, pc, alb, end = _r_seg(t, K, has_env, lit, af, albw, step)
            dt = min(t_new, end) - t
            heat, _, t_floor = r_evolve(heat, K[17], dt, alb, 0.0, False, K)
            heater = max(0.0, K[21] * _r_prad(K[3], alb, K) - K[17]) * t_floor
            heater_j += heater
            s = soc + pc * dt - heater
            if s < 0.0:
                rate = (heater / t_floor if t_floor > 0 else 0.0) - pc
                brown += -s / rate if rate > 0 else dt
                s = 0.0
            disch += max(0.0, soc - s)
            soc = min(cap, s)
            t = min(t_new, end) if end < t_new else t_new
        return t, heat, soc, heater_j, brown, disch

    @nb.njit(cache=True)
    def r_max_fire(t, heat, soc, cap, K, has_env, lit, af, albw, step, fire_in_eclipse):
        """ResourceState.max_fire_s."""
        h_max = K[5]
        if heat >= h_max:
            return 0.0
        H, f = heat, 0.0
        while f < cap:
            lt, pc, alb, end = _r_seg(t, K, has_env, lit, af, albw, step)
            if not lt and not fire_in_eclipse:
                return f
            dt = min(end - t, cap - f)
            de = K[18] - pc
            t_soc = soc / de if de > 0 else np.inf
            p_in = K[19] + _r_batt_loss(pc, K) + K[17]
            H, el, _ = r_evolve(H, p_in, min(dt, t_soc), alb, h_max, True, K)
            if H >= h_max - 1e-6:
                return f + el
            if t_soc <= dt:
                return f + t_soc
            f += dt
            t += dt
            soc -= de * dt
        return np.inf if not has_env else f

    @nb.njit(cache=True)
    def r_wait_for(t, heat, soc, heater_j, brown, disch, f, cap, K, has_env, lit, af, albw, step,
                   fire_in_eclipse):
        """ResourceState.wait_for (idle seconds until a burst of f fits; inf if not within cap)."""
        if r_max_fire(t, heat, soc, 600.0, K, has_env, lit, af, albw, step, fire_in_eclipse) >= f - 1e-9:
            return 0.0
        # never fits: a fresh state (cold store, full battery, no environment)
        if r_max_fire(0.0, 0.0, K[15], 600.0, K, False, lit, af, albw, step, fire_in_eclipse) < f - 1e-9:
            return np.inf
        t0 = t
        s = (t, heat, soc, heater_j, brown, disch)
        while s[0] - t0 < cap:
            prev = s
            s = r_advance(s[0], s[1], s[2], s[3], s[4], s[5], s[0] + _PROBE_S, K, has_env, lit, af, albw, step)
            if r_max_fire(s[0], s[1], s[2], 600.0, K, has_env, lit, af, albw, step, fire_in_eclipse) >= f - 1e-9:
                lo, hi = 0.0, _PROBE_S
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    c = r_advance(prev[0], prev[1], prev[2], prev[3], prev[4], prev[5], prev[0] + mid, K,
                                  has_env, lit, af, albw, step)
                    if r_max_fire(c[0], c[1], c[2], 600.0, K, has_env, lit, af, albw, step,
                                  fire_in_eclipse) >= f - 1e-9:
                        hi = mid
                    else:
                        lo = mid
                return prev[0] + hi - t0
        return np.inf


def element_args(el, ballistic_coeff=None, epoch=None, adot=None):
    """Flatten elements + J2/drag rates into the kernel argument tuple.

    Kernels evaluate x(t) = x0 + xdot * t at sim time t. Elements referring to sim
    time ``epoch`` are re-based to t = 0 (x0 = x_ep - xdot * epoch), which is exact
    for the linear secular model. Drag: ``adot`` [m/s] if given (e.g. the population's
    cached DebrisPopulation.drag_adot), else the orbit-averaged rate at the epoch."""
    from .propagator import drag_da_dt, j2_secular_rates

    raan_dot, argp_dot, M_dot = j2_secular_rates(el)
    a_ep = np.atleast_1d(np.asarray(el.a, float))
    ep = np.zeros_like(a_ep) if epoch is None else np.broadcast_to(np.asarray(epoch, float), a_ep.shape)
    if adot is not None:
        adot = np.broadcast_to(np.asarray(adot, float), a_ep.shape)
    elif ballistic_coeff is None:
        adot = np.zeros_like(a_ep)
    else:
        e_ep = np.broadcast_to(np.asarray(el.e, float), a_ep.shape)
        adot = np.broadcast_to(drag_da_dt(a_ep, ballistic_coeff, ep, e_ep), a_ep.shape)
    f = lambda x: np.ascontiguousarray(np.broadcast_to(np.atleast_1d(x), a_ep.shape), dtype=float)
    base = lambda x, xdot: f(np.asarray(x, float) - np.asarray(xdot, float) * ep)
    return (base(a_ep, adot), f(adot), f(el.e), f(el.i), base(el.raan, raan_dot), f(raan_dot),
            base(el.argp, argp_dot), f(argp_dot), base(el.M, M_dot), f(M_dot))
