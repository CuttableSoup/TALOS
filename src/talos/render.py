"""3D engagement playback: export a run window to a self-contained CesiumJS viewer.

    meta = render_window(sc, events, t0_s, t1_s, "runs/x/render/day0.html")

The viewer propagates every object in the browser (J2 secular, no drag; the mean-element
model of talos.propagator) and replays, in time order, (a) each burst's post-burst
elements (``EngagementEvent.a_after_m`` ...) and (b) the population-dynamics steps (daily
drag decay and re-epoch of every live object, natural decays, new objects).

Both come from an exact replay of the run (``Replay``): the scenario's initial population
(or a full-state checkpoint) is stepped through the same ``PopulationDynamics.step`` calls
the sim made, with every logged burst's elements written in. The dynamics draw from their
own generator, so the replay reproduces the sim's population, including objects added
during the run. Not supported: ablation fragmentation (fragments come from the engagement
generator, which the log does not capture).

Fidelity check: a burst's post-burst elements fix where the sim fired at mid-burst, so
``meta["miss_km"]`` is the distance between that and the viewer's drawn target then.
"""

from __future__ import annotations

import base64
import copy
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from .config import Scenario, build, make_atmosphere
from .constants import R_EARTH
from .elements import KeplerElements, elements_to_rv
from .propagator import get_atmosphere, propagate, set_atmosphere

_VIEWER_DIR = Path(__file__).parent / "viewer"
_EL = ("a", "e", "i", "raan", "argp", "M")
_EV_EL = ("a_after_m", "e_after", "i_after_rad", "raan_after_rad", "argp_after_rad", "M_after_rad")
MATERIAL_NAMES = ("NaK", "low density", "medium density (Al)", "high density", "intact")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _events_frame(events):
    import pandas as pd

    ev = pd.DataFrame(events)
    missing = [c for c in ("t_elements_s", *_EV_EL) if c not in ev.columns]
    if missing or (len(ev) and ev["a_after_m"].isna().any()):
        raise ValueError("events lack post-burst elements (runs before 2026-11-09); re-run, "
                         "or render with scripts/render_run.py without --events")
    # the sim writes a burst's elements while processing the chunk it starts in
    return ev.sort_values(["t_start", "t_stop"], kind="stable").reset_index(drop=True)


class Replay:
    """The run's population history, rebuilt from the scenario + event log."""

    def __init__(self, sc: Scenario, ev, checkpoint=None):
        if sc.ablation.fragment_prob_per_burst > 0:
            raise NotImplementedError("playback of runs with ablation fragmentation is not supported")
        self.sc, self.ev = sc, ev
        b = build(sc)
        self.platforms, self.dyn = b.platforms, b.dynamics
        if checkpoint is None:
            self.pop = copy.deepcopy(b.population)
            self.pop0 = copy.deepcopy(self.pop)
            n = len(self.pop)
            self.removed, self.decayed, self.touched = (np.zeros(n, bool) for _ in range(3))
            self.t = 0.0
        else:
            ck = checkpoint
            self.pop, self.pop0 = copy.deepcopy(ck["pop"]), copy.deepcopy(ck["pop0"])
            self.removed, self.decayed, self.touched = (ck[k].copy() for k in ("removed", "decayed", "touched"))
            self.t = float(ck["c0"])
            if self.dyn is not None and ck.get("dynamics") is not None:
                self.dyn.next_update = ck["dynamics"]["next_update"]
                self.dyn.stats = dict(ck["dynamics"]["stats"])
                self.dyn.rng.bit_generator.state = ck["dynamics"]["rng"]
        self.t_base = self.t
        self.k = int(np.searchsorted(ev["t_start"].to_numpy(float), self.t, side="left"))
        self.after = ev[list(_EV_EL)].to_numpy(float)
        self.added_at = np.full(len(self.pop), -np.inf)  # sim time each object entered (-inf: before)
        self.decayed_at = np.full(len(self.pop), np.inf)

    def _apply_events_before(self, t: float) -> None:
        ev, ts = self.ev, self.ev["t_start"].to_numpy(float)
        while self.k < len(ev) and ts[self.k] < t:
            j = int(ev.debris_index.iat[self.k])
            for name, v in zip(_EL, self.after[self.k]):
                getattr(self.pop.elements, name)[j] = v
            self.pop.epoch_s[j] = float(ev.t_elements_s.iat[self.k])
            self.removed[j] |= bool(ev.removed.iat[self.k])
            self.touched[j] = True
            self.k += 1

    def _step(self, u: float) -> None:
        n_old = len(self.pop)
        self.pop, dmask = self.dyn.step(self.pop, self.removed | self.decayed, u, initial=self.pop0,
                                        engaged=self.touched)
        grow = np.zeros(len(self.pop) - n_old, bool)
        self.removed = np.concatenate([self.removed, grow])
        self.decayed = np.concatenate([self.decayed, grow]) | dmask
        self.touched = np.concatenate([self.touched, grow])
        self.added_at = np.concatenate([self.added_at, np.full(len(grow), u)])
        self.decayed_at = np.concatenate([self.decayed_at, np.full(len(grow), np.inf)])
        self.decayed_at[dmask & ~np.isfinite(self.decayed_at)] = u
        if len(self.pop) > n_old:
            self.pop0 = self.pop0.append(self.pop.subset(np.arange(n_old, len(self.pop))))

    def next_step(self) -> float:
        """Sim time of the next dynamics step: the first chunk start at or after the due time."""
        if self.dyn is None or not self.dyn.cfg.enabled:
            return np.inf
        return self._on_chunk_grid(self.dyn.next_update)

    def _on_chunk_grid(self, due: float) -> float:
        c = self.sc.run.chunk_s
        return float(np.ceil((due - 1e-9) / c) * c)

    def advance(self, t: float, include_step_at_t: bool) -> list[float]:
        """Replay to ``t``; returns the dynamics step times taken. Steps happen at chunk
        starts before that chunk's bursts, so a step at exactly ``t`` is taken (and bursts
        before it applied) only with ``include_step_at_t``."""
        done = []
        while True:
            u = self.next_step()
            if not (u < t - 1e-6 or (include_step_at_t and abs(u - t) <= 1e-6)):
                break
            self._apply_events_before(u)
            self._step(u)
            done.append(u)
        self._apply_events_before(t)
        self.t = t
        return done


def _with_atmosphere(sc, fn):
    prev = get_atmosphere()
    set_atmosphere(make_atmosphere(sc))  # build()'s lifetime filter and the decay steps use it
    try:
        return fn()
    finally:
        set_atmosphere(prev)


def export_playback(sc: Scenario, events, t0_s: float, t1_s: float, checkpoint=None,
                    max_background: int | None = None, seed: int = 0) -> dict:
    """Arrays + metadata for the viewer, window [t0_s, t1_s) in sim seconds.

    ``events``: engagement events (``SimResult.records()`` or an events parquet frame) with
    the post-burst element columns. ``checkpoint``: a ``load_checkpoint`` dict taken at or
    before t0, to start the replay there instead of at t = 0. ``max_background``: draw at
    most this many objects that are never engaged (random subset).
    """
    return _with_atmosphere(sc, lambda: _export(sc, events, t0_s, t1_s, checkpoint, max_background, seed))


def _export(sc, events, t0_s, t1_s, checkpoint, max_background, seed):
    ev = _events_frame(events)
    rp = Replay(sc, ev, checkpoint)
    if t0_s < rp.t_base - 1e-6:
        raise ValueError(f"window starts at {t0_s} s, before the checkpoint ({rp.t_base} s)")
    rp.advance(t0_s, include_step_at_t=True)
    engaged_before = rp.touched.copy()
    removed_before = rp.removed.copy()
    base_el = copy.deepcopy(rp.pop.elements)
    base_ep = rp.pop.epoch_s.copy()
    n_t0 = len(rp.pop)
    win = ev[(ev.t_start >= t0_s) & (ev.t_start < t1_s)].reset_index(drop=True)

    # dynamics steps inside the window: snapshot every object's elements after each
    snaps = []
    for u in _step_times(rp, t1_s):
        rp.advance(u, include_step_at_t=True)
        snaps.append((u, copy.deepcopy(rp.pop.elements), rp.pop.epoch_s.copy(), (rp.removed | rp.decayed).copy()))
    rp.advance(t1_s, include_step_at_t=False)
    pop = rp.pop
    n_all = len(pop)

    # objects to draw
    keep = np.ones(n_all, bool)
    engaged_any = np.zeros(n_all, bool)
    engaged_any[:len(engaged_before)] |= engaged_before
    engaged_any[win.debris_index.to_numpy(int)] = True
    if max_background is not None and max_background < n_all:
        keep[:] = False
        pool = np.flatnonzero(~engaged_any)
        keep[np.random.default_rng(seed).choice(pool, min(max_background, len(pool)), replace=False)] = True
        keep |= engaged_any
    idx = np.flatnonzero(keep)
    remap = np.full(n_all, -1, np.int64)
    remap[idx] = np.arange(len(idx))
    N = len(idx)

    # state at t0 (viewer clock: seconds since t0); objects added later start at their entry state
    el0 = np.zeros((N, 6))
    in0 = idx < n_t0
    e0 = propagate(base_el[idx[in0]], t0_s - base_ep[idx[in0]])
    el0[in0] = np.stack([getattr(e0, k) for k in _EL], axis=1)
    late = idx[~in0]
    if len(late):
        e_late = propagate(rp.pop0.elements[late], t0_s - rp.pop0.epoch_s[late])
        el0[~in0] = np.stack([getattr(e_late, k) for k in _EL], axis=1)
    t_add = np.where(np.isfinite(rp.added_at[idx]), rp.added_at[idx] - t0_s, -1.0)
    t_add = np.maximum(t_add, -1.0)
    gone_at = rp.decayed_at[idx] - t0_s
    gone_at[rp.decayed_at[idx] <= t0_s] = -1.0  # decayed before the window: never shown

    # snapshots, per drawn object; NaN where the sim does not track the object then
    K = len(snaps)
    snap_el = np.full((K, N, 6), np.nan, np.float32)
    for s, (u, el_s, ep_s, gone_s) in enumerate(snaps):
        exists = np.flatnonzero(idx < len(ep_s))
        live = exists[~gone_s[idx[exists]]]
        sub = idx[live]
        e_u = propagate(el_s[sub], u - ep_s[sub])  # live objects were re-epoched to u
        snap_el[s, live] = np.stack([getattr(e_u, k) for k in _EL], axis=1)
    snap_t = np.array([u - t0_s for u, *_ in snaps], float)

    # a burst spanning a step was decayed by that step in the sim: keep the snapshot's elements
    ts, te = win.t_start.to_numpy(float) - t0_s, win.t_stop.to_numpy(float) - t0_s
    keep_el = np.ones(len(win), np.uint8)
    for u in snap_t:
        keep_el[(ts < u) & (te > u)] = 0

    miss = _viewer_miss(el0, snap_t, snap_el, win, remap, t0_s, keep_el)
    platforms = rp.platforms
    pnames = [p.name for p in platforms]
    pel = np.array([[float(getattr(propagate(p.orbit, t0_s), k)) for k in _EL] for p in platforms])
    finite = miss[np.isfinite(miss)]
    meta = {
        "title": sc.run.name,
        "epoch_iso": (sc.run.epoch + timedelta(seconds=float(t0_s))).isoformat().replace("+00:00", "Z"),
        "t0_day": t0_s / 86400.0,
        "duration_s": float(t1_s - t0_s),
        "n_obj": int(N),
        "n_population": int(n_t0),
        "n_events": int(len(win)),
        "n_snapshots": int(K),
        "platforms": [{"name": n, "alt_km": float(p.orbit.a - R_EARTH) / 1e3} for n, p in zip(pnames, platforms)],
        "materials": list(MATERIAL_NAMES),
        "removed_before": int(removed_before.sum()),
        "engaged_before": int(engaged_before.sum()),
        "miss_km": {"median": float(np.median(finite)) if len(finite) else None,
                    "p95": float(np.percentile(finite, 95)) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None},
    }
    eb = np.zeros(n_all, np.uint8)
    eb[:len(engaged_before)] = engaged_before
    rb = np.zeros(n_all, np.uint8)
    rb[:len(removed_before)] = removed_before
    mat = np.asarray(pop.material, int)[idx] if np.ndim(pop.material) else np.full(N, -1)
    arrays = {
        **{f"obj_{k}": el0[:, j].astype(np.float64) for j, k in enumerate(_EL)},
        "obj_id": idx.astype(np.int32),
        "obj_t_add": t_add.astype(np.float32),
        "obj_t_gone": gone_at.astype(np.float32),
        "obj_diam": pop.diameter_m[idx].astype(np.float32),
        "obj_mu": pop.areal_density[idx].astype(np.float32),
        "obj_mat": mat.astype(np.int8),
        "obj_engaged_before": eb[idx], "obj_removed_before": rb[idx],
        "plat_el": pel.ravel().astype(np.float64),
        "snap_t": snap_t, "snap_el": snap_el.ravel(),
        "ev_plat": np.array([pnames.index(p) for p in win.platform], np.uint8),
        "ev_obj": remap[win.debris_index.to_numpy(int)].astype(np.int32),
        "ev_t0": ts, "ev_t1": te, "ev_tm": win.t_elements_s.to_numpy(float) - t0_s,
        "ev_dv": win.dv_retrograde.to_numpy(np.float32),
        "ev_rp0": (win.perigee_alt_before_m.to_numpy(float) / 1e3).astype(np.float32),
        "ev_rp1": (win.perigee_alt_after_m.to_numpy(float) / 1e3).astype(np.float32),
        "ev_removed": win.removed.to_numpy(np.uint8),
        "ev_keep_el": keep_el,
        "ev_el": win[list(_EV_EL)].to_numpy(np.float64).ravel(),
        "ev_miss_km": np.nan_to_num(miss, nan=-1.0).astype(np.float32),
    }
    return {"meta": meta, "arrays": arrays}


def _step_times(rp: Replay, t1_s: float) -> list[float]:
    if not np.isfinite(rp.next_step()):
        return []
    out, due = [], rp.dyn.next_update
    while (u := rp._on_chunk_grid(due)) < t1_s - 1e-6:
        out.append(u)
        due += rp.dyn.cfg.update_s
    return out


def _viewer_miss(el0, snap_t, snap_el, win, remap, t0_s, keep_el) -> np.ndarray:
    """Mirror the viewer's state machine and measure |drawn target - sim target| at mid-burst."""
    el, ep = el0.copy(), np.zeros(len(el0))
    t1 = win.t_stop.to_numpy(float) - t0_s
    tm = win.t_elements_s.to_numpy(float) - t0_s
    obj = remap[win.debris_index.to_numpy(int)]
    after = win[list(_EV_EL)].to_numpy(float)
    # timeline at equal times: snapshots, then applies, then measures
    items = [(u, 0, s) for s, u in enumerate(snap_t)]
    items += [(t1[k], 1, k) for k in range(len(win))]
    items += [(tm[k], 2, k) for k in range(len(win))]
    items.sort(key=lambda x: (x[0], x[1]))
    miss = np.full(len(win), np.nan)
    for t, kind, k in items:
        if kind == 0:
            ok = np.isfinite(snap_el[k, :, 0])
            el[ok] = snap_el[k, ok]
            ep[ok] = t
        elif kind == 1:
            if keep_el[k]:
                el[obj[k]] = after[k]
                ep[obj[k]] = tm[k]
        else:
            o = obj[k]
            r_view, _ = elements_to_rv(propagate(KeplerElements(*el[o]), tm[k] - ep[o]))
            r_sim, _ = elements_to_rv(KeplerElements(*after[k]))
            miss[k] = np.linalg.norm(r_view - r_sim) / 1e3
    return miss


def write_viewer(data: dict, path) -> Path:
    """Write the self-contained viewer HTML (Cesium from jsDelivr; data inline)."""
    payload = {"meta": data["meta"],
               "arrays": {k: {"dtype": v.dtype.name, "b64": _b64(v)} for k, v in data["arrays"].items()}}
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (_VIEWER_DIR / "template.html").read_text(encoding="utf-8")
    for key, file in (("/*__ORBIT_JS__*/", "orbit.js"), ("/*__VIEWER_JS__*/", "viewer.js")):
        html = html.replace(key, (_VIEWER_DIR / file).read_text(encoding="utf-8"))
    html = html.replace("__TALOS_TITLE__", f"TALOS playback: {data['meta']['title']}")
    html = html.replace("/*__DATA__*/", blob)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def render_window(sc: Scenario, events, t0_s: float, t1_s: float, out_html, checkpoint=None,
                  max_background: int | None = None) -> dict:
    """Export + write one window; returns the metadata. ``checkpoint`` (path or dict) only
    speeds up the replay: one taken after t0 is ignored (checkpoint files hold only the
    latest state), and a missing file is fine."""
    if checkpoint is not None:
        from .simulation import load_checkpoint

        ck = checkpoint if isinstance(checkpoint, dict) else (
            load_checkpoint(checkpoint) if Path(checkpoint).exists() else None)
        checkpoint = ck if ck is not None and float(ck["c0"]) <= t0_s else None
    data = export_playback(sc, events, t0_s, t1_s, checkpoint=checkpoint, max_background=max_background)
    data["meta"]["path"] = str(write_viewer(data, out_html))
    data["meta"]["from_checkpoint"] = checkpoint is not None
    return data["meta"]


def describe(meta: dict) -> str:
    """One line for logs: where the page is and how faithful the drawn targets are."""
    m = meta["miss_km"]
    gap = (f"median {m['median']:.3g} km, p95 {m['p95']:.3g} km, max {m['max']:.3g} km"
           if m["median"] is not None else "n/a")
    size = Path(meta["path"]).stat().st_size / 1e6
    return (f"{meta['path']} ({size:.1f} MB): {meta['n_obj']:,} objects, {meta['n_events']:,} bursts, "
            f"{meta['n_snapshots']} decay steps; target vs sim at mid-burst {gap}")
