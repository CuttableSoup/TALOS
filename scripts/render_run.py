"""Render a 3D engagement playback (CesiumJS) for a window of a run.

Fresh run of a scenario (simulates up to the window end, then renders):
    python scripts/render_run.py --config config/design.toml --days 1 --out runs/render/design_d0.html

Existing run (events parquet written 2026-11-09 or later; same --config/--set as that run):
    python scripts/render_run.py --events runs/x/events/design.parquet --config config/design.toml \\
        --from-day 200 --days 1 --checkpoint runs/x/checkpoints/design.pkl --out runs/x/playback.html

Open it with --serve (or serve the folder: python -m http.server 8765 -d runs/render); CesiumJS
loads from jsDelivr, and its globe workers do not run from file://.
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talos.config import load_scenario, run_scenario  # noqa: E402
from talos.render import describe, render_window  # noqa: E402


def parse_set(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            out[k] = tomllib.loads(f"x = {v}")["x"]
        except tomllib.TOMLDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="scenario TOML (default config/baseline.toml)")
    ap.add_argument("--set", nargs="*", default=[], dest="sets", help="section.key=value overrides")
    ap.add_argument("--events", default=None, help="events parquet of an existing run (else simulate)")
    ap.add_argument("--checkpoint", default=None, help="checkpoint .pkl at or before the window (long runs)")
    ap.add_argument("--from-day", type=float, default=0.0)
    ap.add_argument("--days", type=float, default=1.0, help="window length")
    ap.add_argument("--max-background", type=int, default=None, help="draw at most this many untouched objects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--serve", action="store_true", help="serve the output folder on localhost and open a browser")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    t0, t1 = a.from_day * 86400.0, (a.from_day + a.days) * 86400.0
    over = parse_set(a.sets)
    if a.events is None:
        sc = load_scenario(a.config, {**over, "run.duration_days": a.from_day + a.days})
        w = time.perf_counter()
        print(f"simulating {sc.run.name} for {a.from_day + a.days:g} days ...", flush=True)
        res = run_scenario(sc)
        events = res.records()
        print(f"  {len(events)} bursts in {time.perf_counter() - w:.0f} s", flush=True)
    else:
        import pandas as pd

        sc = load_scenario(a.config, over)
        events = pd.read_parquet(a.events)
    meta = render_window(sc, events, t0, t1, a.out, checkpoint=a.checkpoint, max_background=a.max_background)
    path = Path(meta["path"])
    print("wrote", describe(meta), flush=True)
    if a.serve:
        serve(path, a.port)


def serve(path: Path, port: int):
    """Cesium's web workers (globe tiles) need an http origin, not file://."""
    import functools
    import http.server
    import webbrowser

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.parent.resolve()))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{path.name}"
        print(f"serving {url}  (Ctrl+C to stop)", flush=True)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
