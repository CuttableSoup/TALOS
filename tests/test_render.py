"""Playback export: browser propagator matches talos.propagator; export reproduces the sim."""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from talos.config import load_scenario, run_scenario
from talos.elements import KeplerElements, elements_to_rv
from talos.propagator import propagate
from talos.render import _VIEWER_DIR, export_playback, write_viewer

BASE = {"population.kind": "toy", "population.n_real": 3000, "population.min_lifetime_yr": None,
        "population.areal_density_model": "sbm", "constellation.n_sats": 2, "dynamics.enabled": False}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_propagator_matches_python(tmp_path):
    rng = np.random.default_rng(4)
    n = 200
    el = KeplerElements(a=6.378e6 + rng.uniform(2e5, 1.2e6, n), e=rng.uniform(0, 0.08, n),
                        i=rng.uniform(0, np.pi, n), raan=rng.uniform(0, 2 * np.pi, n),
                        argp=rng.uniform(0, 2 * np.pi, n), M=rng.uniform(0, 2 * np.pi, n))
    ep = rng.uniform(-1e5, 1e5, n)
    t = rng.uniform(0, 7 * 86400, n)
    adot = -rng.uniform(0, 0.06, n)  # up to ~5 km/day, the fastest-decaying fragments
    adot[::2] = 0.0
    want, _ = elements_to_rv(propagate(el, t - ep, adot=adot))
    js = tmp_path / "check.js"
    js.write_text(
        f"const O = require({json.dumps(str(_VIEWER_DIR / 'orbit.js'))});\n"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "const n = d.t.length, el = {}; for (const k of ['a','e','i','raan','argp','M','ep']) el[k] = Float64Array.from(d[k]);\n"
        "const rates = new Float64Array(3 * n), out = new Float64Array(3 * n), adot = Float64Array.from(d.adot);\n"
        "for (let k = 0; k < n; k++) { O.j2Rates(el.a[k], el.e[k], el.i[k], rates, k); O.positionAt(el, rates, k, d.t[k], out, 3 * k, adot); }\n"
        "process.stdout.write(JSON.stringify(Array.from(out)));\n")
    inp = {k: getattr(el, k).tolist() for k in ("a", "e", "i", "raan", "argp", "M")} | {"ep": ep.tolist(), "t": t.tolist(), "adot": adot.tolist()}
    out = subprocess.run(["node", str(js)], input=json.dumps(inp), capture_output=True, text=True, check=True)
    got = np.array(json.loads(out.stdout)).reshape(n, 3)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 1e-3  # metres


@pytest.fixture(scope="module")
def short_run():
    sc = load_scenario(overrides={**BASE, "run.duration_s": 3 * 3600.0})
    return sc, run_scenario(sc).records()


def test_events_carry_post_burst_elements(short_run):
    _, ev = short_run
    assert ev and all(np.isfinite(e["a_after_m"]) and e["t_start"] <= e["t_elements_s"] <= e["t_stop"] for e in ev)


def test_export_reproduces_sim_targets(short_run):
    sc, ev = short_run
    data = export_playback(sc, ev, 0.0, 3 * 3600.0)
    m, a = data["meta"], data["arrays"]
    assert m["n_events"] == len(ev) and m["n_obj"] == m["n_population"]
    assert len(a["ev_el"]) == 6 * len(ev) and a["ev_obj"].min() >= 0
    # only drag (not drawn) separates the viewer's target from the sim's at mid-burst
    assert m["miss_km"]["max"] < 1.0


def test_export_window_and_subset(short_run):
    sc, ev = short_run
    full = export_playback(sc, ev, 0.0, 3 * 3600.0)
    late = export_playback(sc, ev, 1.5 * 3600.0, 3 * 3600.0, max_background=200)
    assert 0 < late["meta"]["n_events"] < full["meta"]["n_events"]
    assert late["meta"]["engaged_before"] > 0
    assert late["meta"]["miss_km"]["max"] < 1.0  # objects engaged before t0 start from their logged elements
    engaged = set(late["arrays"]["obj_id"][late["arrays"]["ev_obj"]])
    assert late["meta"]["n_obj"] <= 200 + late["meta"]["engaged_before"] + len(engaged)


def test_old_events_rejected(short_run):
    sc, ev = short_run
    old = [{k: v for k, v in e.items() if not k.endswith("_after_m")} for e in ev]
    with pytest.raises(ValueError, match="post-burst"):
        export_playback(sc, old, 0.0, 3600.0)


def test_write_viewer_inlines_everything(short_run, tmp_path):
    sc, ev = short_run
    data = export_playback(sc, ev, 0.0, 3600.0)
    path = write_viewer(data, tmp_path / "v.html")
    html = path.read_text(encoding="utf-8")
    assert "/*__" not in html and "__TALOS_TITLE__" not in html
    blob = html.split('id="talos-data">', 1)[1].split("</script>", 1)[0]
    payload = json.loads(blob)
    a = payload["arrays"]["obj_a"]
    vals = np.frombuffer(base64.b64decode(a["b64"]), dtype=a["dtype"])
    assert len(vals) == data["meta"]["n_obj"] and np.all(vals > 6.3e6)


def test_render_window_checkpoint(tmp_path):
    from talos.render import render_window

    sc = load_scenario(overrides={**BASE, "dynamics.enabled": True, "dynamics.update_days": 0.03,
                                  "run.duration_s": 3 * 3600.0, "run.checkpoint_s": 3600.0})
    ck = tmp_path / "ck.pkl"
    ev = run_scenario(sc, checkpoint_path=ck).records()  # file holds the last checkpoint, c0 = 2 h
    late = render_window(sc, ev, 2.0 * 3600, 3 * 3600.0, tmp_path / "late.html", checkpoint=ck)
    early = render_window(sc, ev, 0.0, 3600.0, tmp_path / "early.html", checkpoint=ck)
    missing = render_window(sc, ev, 0.0, 3600.0, tmp_path / "m.html", checkpoint=tmp_path / "none.pkl")
    assert late["from_checkpoint"] and not early["from_checkpoint"] and not missing["from_checkpoint"]
    assert (tmp_path / "late.html").exists() and late["n_events"] > 0
    # with daily-decay updates the checkpoint keeps the reconstruction on the sim's targets
    assert late["miss_km"]["max"] < 1.0 and early["miss_km"]["max"] < 1.0


DYN = {**BASE, "dynamics.enabled": True, "dynamics.update_days": 0.03, "dynamics.breakup_rate_per_yr": 30000.0,
       "dynamics.breakup_fragments": 50, "run.checkpoint_s": 3600.0}


def test_replay_reproduces_sim_population(tmp_path):
    from talos.render import Replay, _events_frame, _with_atmosphere
    from talos.simulation import load_checkpoint

    sc = load_scenario(overrides={**DYN, "run.duration_s": 2.5 * 3600.0})
    ck = tmp_path / "ck.pkl"
    ev = _events_frame(run_scenario(sc, checkpoint_path=ck).records())
    want = load_checkpoint(ck)  # the sim's own state at c0 = 2 h
    rp = _with_atmosphere(sc, lambda: Replay(sc, ev))
    _with_atmosphere(sc, lambda: rp.advance(want["c0"], include_step_at_t=False))
    assert len(rp.pop) == len(want["pop"]) > len(ev)  # breakups and decays were replayed too
    for k in ("a", "e", "i", "raan", "argp", "M"):
        np.testing.assert_array_equal(getattr(rp.pop.elements, k), getattr(want["pop"].elements, k))
    np.testing.assert_array_equal(rp.pop.epoch_s, want["pop"].epoch_s)
    np.testing.assert_array_equal(rp.removed | rp.decayed, want["removed"] | want["decayed"])


def test_late_window_without_checkpoint_stays_on_target(tmp_path):
    from talos.render import render_window

    sc = load_scenario(overrides={**DYN, "run.duration_s": 3 * 3600.0})
    ev = run_scenario(sc).records()
    m = render_window(sc, ev, 2.0 * 3600, 3 * 3600.0, tmp_path / "late.html")
    assert not m["from_checkpoint"] and m["n_snapshots"] >= 1 and m["n_events"] > 0
    assert m["miss_km"]["max"] < 1.0  # decay steps before and inside the window are replayed


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_decay_step_matches_python_mirror(tmp_path):
    from talos.render import _local_step

    rng = np.random.default_rng(9)
    n = 100
    el = np.stack([6.378e6 + rng.uniform(3e5, 1.1e6, n), rng.uniform(0, 0.05, n), rng.uniform(0, np.pi, n),
                   rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)], 1)
    ep = rng.uniform(0, 4e4, n)
    rate = np.stack([-rng.uniform(0, 0.05, n), -rng.uniform(0, 1e-9, n)], 1).astype(np.float32).astype(float)
    py = el.copy(), ep.copy()
    for u in (86400.0, 172800.0):
        _local_step(py[0], py[1], rate, u, np.ones(n, bool))
    t = 200000.0
    want, _ = elements_to_rv(propagate(KeplerElements(*py[0].T), t - py[1]))
    js = tmp_path / "step.js"
    js.write_text(
        f"const O = require({json.dumps(str(_VIEWER_DIR / 'orbit.js'))});\n"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8')); const n = d.ep.length;\n"
        "const el = {}; ['a','e','i','raan','argp','M'].forEach((k, j) => el[k] = Float64Array.from(d.el.map(r => r[j])));\n"
        "el.ep = Float64Array.from(d.ep); const ad = Float64Array.from(d.rate.map(r => r[0])), ed = Float64Array.from(d.rate.map(r => r[1]));\n"
        "const rates = new Float64Array(3 * n), out = new Float64Array(3 * n);\n"
        "for (let k = 0; k < n; k++) O.j2Rates(el.a[k], el.e[k], el.i[k], rates, k);\n"
        "for (const u of [86400, 172800]) for (let k = 0; k < n; k++) O.decayStep(el, rates, ad, ed, k, u);\n"
        "for (let k = 0; k < n; k++) O.positionAt(el, rates, k, d.t, out, 3 * k);\n"
        "process.stdout.write(JSON.stringify(Array.from(out)));\n")
    inp = {"el": el.tolist(), "ep": ep.tolist(), "rate": rate.tolist(), "t": t}
    out = subprocess.run(["node", str(js)], input=json.dumps(inp), capture_output=True, text=True, check=True)
    got = np.array(json.loads(out.stdout)).reshape(n, 3)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 1e-2  # metres
