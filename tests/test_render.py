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
    want, _ = elements_to_rv(propagate(el, t - ep))
    js = tmp_path / "check.js"
    js.write_text(
        f"const O = require({json.dumps(str(_VIEWER_DIR / 'orbit.js'))});\n"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "const n = d.t.length, el = {}; for (const k of ['a','e','i','raan','argp','M','ep']) el[k] = Float64Array.from(d[k]);\n"
        "const rates = new Float64Array(3 * n), out = new Float64Array(3 * n);\n"
        "for (let k = 0; k < n; k++) { O.j2Rates(el.a[k], el.e[k], el.i[k], rates, k); O.positionAt(el, rates, k, d.t[k], out, 3 * k); }\n"
        "process.stdout.write(JSON.stringify(Array.from(out)));\n")
    inp = {k: getattr(el, k).tolist() for k in ("a", "e", "i", "raan", "argp", "M")} | {"ep": ep.tolist(), "t": t.tolist()}
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
