"""Module 9 — parallel sweep driver.

Runs are independent, so the sweep fans out over worker processes; inside each
worker the numba kernels use ``threads_per_job`` threads. The default split uses
every logical core: n_jobs * threads_per_job = os.cpu_count().

Outputs (in ``out_dir``):
  * summary.parquet: one row per run (config + SimResult.summary()).
  * events/<name>.parquet: per-engagement event log for playback/analysis.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import Scenario, load_scenario, run_scenario


@dataclass(frozen=True)
class RunConfig:
    """One sweep point: a scenario TOML plus dotted overrides, e.g.
    ``RunConfig("hi_power", overrides={"laser.burst_power_kw": 50})``."""

    name: str
    config: str | None = None  # TOML path; None -> config/baseline.toml
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def scenario(self) -> Scenario:
        return load_scenario(self.config, {"run.name": self.name, **self.overrides})


def run_one(cfg: RunConfig) -> tuple[dict, list[dict]]:
    """Build and run ``cfg``'s scenario; return (summary row, events)."""
    t0 = time.perf_counter()
    sc = cfg.scenario()
    res = run_scenario(sc)
    ev_c = [e for e in res.events if e.committed]
    row = {"name": cfg.name, "config": cfg.config or "baseline",
           **{f"ovr.{k}": (str(v) if isinstance(v, (tuple, list)) else v) for k, v in cfg.overrides.items()},
           **res.summary(), **{f"planner_{k}": v for k, v in res.planner_stats.items()},
           "committed_engagement_frac": len(ev_c) / max(len(res.events), 1),
           "wall_s": time.perf_counter() - t0}
    events = [{"run": cfg.name, **e} for e in res.records()]
    return row, events


def _init_worker(threads: int) -> None:
    from . import _kernels

    if _kernels.HAVE_NUMBA:
        _kernels.nb.set_num_threads(max(1, threads))


def _warm_jit() -> None:
    """Compile/load the numba kernels once in the parent so workers hit the cache."""
    run_one(RunConfig("_warm", overrides={"constellation.n_sats": 1, "population.kind": "toy",
                                          "population.n_real": 2000,
                                          "run.duration_s": 120.0, "population.min_lifetime_yr": None}))


def run_sweep(
    configs: list[RunConfig],
    out_dir: str | os.PathLike | None = None,
    n_jobs: int | None = None,
    threads_per_job: int | None = None,
    progress: bool = True,
):
    """Run ``configs`` across processes; returns a pandas DataFrame of summaries.

    On Windows (spawn start method) call this from inside an
    ``if __name__ == "__main__":`` block, or worker processes crash on start.
    """
    import pandas as pd

    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("RunConfig names must be unique")
    cpus = os.cpu_count() or 1
    n_jobs = n_jobs or min(len(configs), max(1, cpus // 4))
    threads_per_job = threads_per_job or max(1, cpus // n_jobs)
    _warm_jit()

    rows, events = [], []
    if n_jobs == 1:
        _init_worker(threads_per_job)
        for c in configs:
            r, ev = run_one(c)
            rows.append(r)
            events.append((c.name, ev))
            if progress:
                print(f"[{len(rows)}/{len(configs)}] {c.name} {r['wall_s']:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(threads_per_job,)) as ex:
            futs = {ex.submit(run_one, c): c for c in configs}
            for f in as_completed(futs):
                r, ev = f.result()
                rows.append(r)
                events.append((futs[f].name, ev))
                if progress:
                    print(f"[{len(rows)}/{len(configs)}] {futs[f].name} {r['wall_s']:.0f}s", flush=True)

    df = pd.DataFrame(rows).set_index("name").loc[names].reset_index()
    if out_dir is not None:
        out = Path(out_dir)
        (out / "events").mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "summary.parquet", index=False)
        for name, ev in events:
            pd.DataFrame(ev).to_parquet(out / "events" / f"{name}.parquet", index=False)
    return df


def readme_matrix(base: RunConfig = RunConfig(name="base"), **axes_override) -> list[RunConfig]:
    """The README sweep: power x sats x orbit config x Cm x mu. Axis keys are dotted
    scenario keys; pass ``axes_override={...}`` entries to replace or add axes."""
    axes = {
        "laser.burst_power_kw": [5.0, 10.0, 20.0, 30.0, 50.0],
        "constellation.n_sats": [1, 2, 4, 8],
        "constellation.altitudes_km": [(800.0,), (700.0, 800.0, 900.0)],
        "ablation.cm_n_per_mw": [25.0, 50.0, 100.0],
        "population.areal_density": [0.2, 1.0, 10.0],
    }
    axes.update(axes_override)
    keys = list(axes)
    out = []
    for vals in itertools.product(*(axes[k] for k in keys)):
        kw = dict(zip(keys, vals))
        tag = "_".join(f"{k.split('.')[-1]}={'-'.join(map(str, v)) if isinstance(v, tuple) else v}"
                       for k, v in kw.items())
        if "population.areal_density" in kw:  # a mu axis only means something with the fixed-mu model
            kw.setdefault("population.areal_density_model", "fixed")
        out.append(RunConfig(name=tag, config=base.config, overrides={**base.overrides, **kw}))
    return out
