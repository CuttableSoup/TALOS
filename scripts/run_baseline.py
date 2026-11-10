"""One scenario run with progress log and event checkpoints.

    python scripts/run_baseline.py DAYS --out runs/x [--name base] [--threads 16]
        [--config path.toml] [--set section.key=value ...]

Values in --set are parsed as TOML (numbers, true/false, "strings", [lists]).
Progress: runs/x/logs/<name>.log; events: runs/x/events/<name>.parquet;
full-state checkpoints every run.checkpoint_days (30) in runs/x/checkpoints/<name>.pkl.
Continue / extend: run_baseline.py 730 --out runs/y --resume runs/x/checkpoints/base.pkl
(same --config/--set as the original run; only the duration may change).
"""

import argparse
import tomllib

from talos.sweep import RunConfig, run_sweep


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            out[k] = tomllib.loads(f"x = {v}")["x"]
        except tomllib.TOMLDecodeError:
            out[k] = v
    return out


if __name__ == "__main__":  # required on Windows (spawned workers re-import this file)
    ap = argparse.ArgumentParser()
    ap.add_argument("days", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="base")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--resume", default=None, help="checkpoint .pkl to continue (or extend) from")
    a = ap.parse_args()
    cfg = RunConfig(a.name, config=a.config, overrides={"run.duration_days": a.days, **parse_set(a.sets)},
                    resume=a.resume)
    df = run_sweep([cfg], out_dir=a.out, n_jobs=1, threads_per_job=a.threads)
    print(df.T.to_string())
