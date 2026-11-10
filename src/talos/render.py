"""3D engagement playback: export a run window to a self-contained CesiumJS viewer.

    data = export_playback(sc, events, t0_s, t1_s)          # arrays + metadata
    write_viewer(data, "runs/x/playback.html")

The viewer propagates every object in the browser (J2 secular, no drag; the same
mean-element model as talos.propagator) and replays each engagement's post-burst
elements (``EngagementEvent.a_after_m`` ...) as playback passes it. Base state at t0:
the scenario's initial population, or a full-state checkpoint (``checkpoint=``) for
windows late in a long run, plus every logged event before t0.

Fidelity check: an event's post-burst elements fix where the sim fired at mid-burst, so
``data["meta"]["miss_km"]`` is the distance between that and the viewer's reconstructed
target at the same instant (drag and the daily decay step are not reconstructed).
"""

from __future__ import annotations

import base64
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


def _base_population(sc: Scenario, checkpoint):
    """(DebrisPopulation, platforms, t_base) at the start of what we can replay from."""
    prev = get_atmosphere()
    set_atmosphere(make_atmosphere(sc))  # the lifetime filter in build() uses it
    try:
        b = build(sc)
    finally:
        set_atmosphere(prev)
    if checkpoint is None:
        return b.population, b.platforms, 0.0
    from .simulation import load_checkpoint

    ck = load_checkpoint(checkpoint) if not isinstance(checkpoint, dict) else checkpoint
    return ck["pop"], b.platforms, float(ck["c0"])


def _miss_km(el_cur: dict, ep_cur: dict, base_el: KeplerElements, base_ep: np.ndarray, ev) -> np.ndarray:
    """Per window event: |viewer target position - sim target position| at mid-burst [km]."""
    out = np.full(len(ev), np.nan)
    cur, cur_ep = dict(el_cur), dict(ep_cur)
    after = ev[list(_EV_EL)].to_numpy(float)
    tm = ev["t_elements_s"].to_numpy(float)
    js = ev["debris_index"].to_numpy(int)
    for k in range(len(ev)):
        j = js[k]
        if j in cur:
            el, ep = cur[j], cur_ep[j]
        elif j < len(base_ep):
            el, ep = base_el[j], float(base_ep[j])
        else:  # object added during the run: nothing to compare before its first event
            el = None
        if el is not None:
            r_view, _ = elements_to_rv(propagate(el, tm[k] - ep))
            r_sim, _ = elements_to_rv(KeplerElements(*after[k]))
            out[k] = np.linalg.norm(r_view - r_sim) / 1e3
        cur[j], cur_ep[j] = KeplerElements(*after[k]), tm[k]
    return out


def export_playback(sc: Scenario, events, t0_s: float, t1_s: float, checkpoint=None,
                    max_background: int | None = None, seed: int = 0) -> dict:
    """Arrays + metadata for the viewer, window [t0_s, t1_s] in sim seconds.

    ``events``: DataFrame of engagement events (``SimResult.records()`` / events parquet)
    with the post-burst element columns. ``max_background``: draw at most this many
    never-engaged objects (random subset); engaged ones are always drawn.
    """
    import pandas as pd

    ev = pd.DataFrame(events)
    missing = [c for c in ("t_elements_s", *_EV_EL) if c not in ev.columns]
    if missing or (len(ev) and ev["a_after_m"].isna().any()):
        raise ValueError("events lack post-burst elements (runs before 2026-11-09); re-run, "
                         "or render with scripts/render_run.py without --events")
    ev = ev.sort_values("t_stop", kind="stable").reset_index(drop=True)
    pop, platforms, t_base = _base_population(sc, checkpoint)
    if t0_s < t_base:
        raise ValueError(f"window starts at {t0_s} s, before the checkpoint ({t_base} s)")
    n_base = len(pop)
    base_el, base_ep = pop.elements, np.asarray(pop.epoch_s, float)

    # replay events between the base and t0
    pre = ev[(ev.t_stop >= t_base) & (ev.t_stop < t0_s)]
    el_cur, ep_cur = {}, {}
    for r in pre.itertuples(index=False):
        el_cur[r.debris_index] = KeplerElements(*(getattr(r, c) for c in _EV_EL))
        ep_cur[r.debris_index] = r.t_elements_s
    win = ev[(ev.t_stop >= t0_s) & (ev.t_start <= t1_s)].reset_index(drop=True)
    miss = _miss_km(el_cur, ep_cur, base_el, base_ep, win)

    # which objects to draw: all (or a subset of) the base population + anything engaged
    engaged = set(win.debris_index.tolist()) | set(el_cur)
    n_obj = max([n_base, *[j + 1 for j in engaged]])
    keep = np.zeros(n_obj, bool)
    keep[:n_base] = True
    if max_background is not None and max_background < n_base:
        keep[:n_base] = False
        keep[np.random.default_rng(seed).choice(n_base, max_background, replace=False)] = True
    keep[list(engaged)] = True
    idx = np.flatnonzero(keep)
    remap = np.full(n_obj, -1, np.int64)
    remap[idx] = np.arange(len(idx))

    # elements at t0 (relative clock: viewer t = sim t - t0)
    el = {k: np.zeros(len(idx)) for k in _EL}
    hidden = np.zeros(len(idx), np.uint8)
    diam = np.full(len(idx), np.nan, np.float32)
    mu = np.full(len(idx), np.nan, np.float32)
    mat = np.full(len(idx), -1, np.int8)
    inb = idx < n_base
    bi = idx[inb]
    e0 = propagate(base_el[bi], t0_s - base_ep[bi])
    for k in _EL:
        el[k][inb] = getattr(e0, k)
    diam[inb] = pop.diameter_m[bi]
    mu[inb] = pop.areal_density[bi]
    mat[inb] = np.asarray(pop.material, int)[bi] if np.ndim(pop.material) else -1
    first = win.groupby("debris_index").first()
    for j, e_j in el_cur.items():  # engaged before t0: their latest elements
        e_t0 = propagate(e_j, t0_s - ep_cur[j])
        for k in _EL:
            el[k][remap[j]] = float(getattr(e_t0, k))
    for j in idx[~inb]:  # added during the run: appear at their first engagement
        if j in el_cur:
            continue
        hidden[remap[j]] = 1
        r = first.loc[j]
        e_t0 = propagate(KeplerElements(*(r[c] for c in _EV_EL)), t0_s - r.t_elements_s)
        for k in _EL:
            el[k][remap[j]] = float(getattr(e_t0, k))

    pnames = [p.name for p in platforms]
    pel = np.array([[float(getattr(propagate(p.orbit, t0_s), k)) for k in _EL] for p in platforms])
    rel = lambda c: (win[c].to_numpy(float) - t0_s)  # noqa: E731
    finite = miss[np.isfinite(miss)]
    meta = {
        "title": sc.run.name,
        "epoch_iso": (sc.run.epoch + timedelta(seconds=float(t0_s))).isoformat().replace("+00:00", "Z"),
        "t0_day": t0_s / 86400.0,
        "duration_s": float(t1_s - t0_s),
        "n_obj": int(len(idx)),
        "n_population": int(n_base),
        "n_events": int(len(win)),
        "platforms": [{"name": n, "alt_km": float(p.orbit.a - R_EARTH) / 1e3} for n, p in zip(pnames, platforms)],
        "materials": list(MATERIAL_NAMES),
        "removed_before": int(pre.removed.sum()),
        "engaged_before": int(len(el_cur)),
        "miss_km": {"median": float(np.median(finite)) if len(finite) else None,
                    "p95": float(np.percentile(finite, 95)) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None},
    }
    arrays = {
        **{f"obj_{k}": el[k].astype(np.float64) for k in _EL},
        "obj_id": idx.astype(np.int32), "obj_hidden": hidden, "obj_diam": diam, "obj_mu": mu, "obj_mat": mat,
        "obj_engaged_before": np.isin(idx, list(el_cur)).astype(np.uint8),
        "obj_removed_before": np.isin(idx, pre.debris_index[pre.removed].to_numpy()).astype(np.uint8),
        "plat_el": pel.ravel().astype(np.float64),
        "ev_plat": np.array([pnames.index(p) for p in win.platform], np.uint8),
        "ev_obj": remap[win.debris_index.to_numpy(int)].astype(np.int32),
        "ev_t0": rel("t_start"), "ev_t1": rel("t_stop"), "ev_tm": rel("t_elements_s"),
        "ev_dv": win.dv_retrograde.to_numpy(np.float32),
        "ev_rp0": (win.perigee_alt_before_m.to_numpy(float) / 1e3).astype(np.float32),
        "ev_rp1": (win.perigee_alt_after_m.to_numpy(float) / 1e3).astype(np.float32),
        "ev_removed": win.removed.to_numpy(np.uint8),
        "ev_el": win[list(_EV_EL)].to_numpy(np.float64).ravel(),
        "ev_miss_km": np.nan_to_num(miss, nan=-1.0).astype(np.float32),
    }
    return {"meta": meta, "arrays": arrays}


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
