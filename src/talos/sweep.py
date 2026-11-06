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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .engagement import AblationParams
from .platform import LaserConfig, baseline_constellation
from .population import toy_population
from .simulation import run_engagements


@dataclass(frozen=True)
class RunConfig:
    name: str
    n_sats: int = 4
    altitudes_km: tuple[float, ...] = (700.0, 800.0, 900.0)
    burst_power_w: float = 25e3
    eta2: float = 0.40
    cm: float = 50e-6
    areal_density: float = 1.0
    n_real: float = 1.0e5
    min_lifetime_yr: float | None = 1.0
    duration_s: float = 86400.0
    score: str = "removal_progress"
    seed: int = 0
    stochastic: bool = True


def run_one(cfg: RunConfig) -> tuple[dict, list[dict]]:
    """Build population + constellation from ``cfg``, run, return (summary row, events)."""
    t0 = time.perf_counter()
    pop = toy_population(int(cfg.n_real), np.random.default_rng(cfg.seed), n_real=cfg.n_real,
                         areal_density=cfg.areal_density, min_lifetime_yr=cfg.min_lifetime_yr)
    plats = baseline_constellation(cfg.n_sats, cfg.altitudes_km,
                                   laser=LaserConfig(burst_power_w=cfg.burst_power_w, eta2=cfg.eta2))
    res = run_engagements(plats, pop, cfg.duration_s, params=AblationParams(cm=cfg.cm),
                          rng=np.random.default_rng(cfg.seed + 1) if cfg.stochastic else None,
                          score=cfg.score)
    row = {**{k: (str(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
           **res.summary(), "wall_s": time.perf_counter() - t0}
    events = [{"run": cfg.name, **e} for e in res.records()]
    return row, events


def _init_worker(threads: int) -> None:
    from . import _kernels

    if _kernels.HAVE_NUMBA:
        _kernels.nb.set_num_threads(max(1, threads))


def _warm_jit() -> None:
    """Compile/load the numba kernels once in the parent so workers hit the cache."""
    run_one(RunConfig(name="_warm", n_sats=1, n_real=2000, duration_s=120.0, min_lifetime_yr=None))


def run_sweep(
    configs: list[RunConfig],
    out_dir: str | os.PathLike | None = None,
    n_jobs: int | None = None,
    threads_per_job: int | None = None,
    progress: bool = True,
):
    """Run ``configs`` across processes; returns a pandas DataFrame of summaries."""
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


def readme_matrix(base: RunConfig = RunConfig(name="base"), **overrides) -> list[RunConfig]:
    """The README sweep: power x sats x orbit config x Cm x mu (x eta2)."""
    axes = {
        "burst_power_w": [5e3, 10e3, 20e3, 30e3, 50e3],
        "n_sats": [1, 2, 4, 8],
        "altitudes_km": [(800.0,), (700.0, 800.0, 900.0)],
        "cm": [25e-6, 50e-6, 100e-6],
        "areal_density": [0.2, 1.0, 10.0],
    }
    axes.update(overrides)
    keys = list(axes)
    out = []
    for vals in itertools.product(*(axes[k] for k in keys)):
        kw = dict(zip(keys, vals))
        tag = "_".join(f"{k}={'-'.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in kw.items())
        out.append(replace(base, name=tag, **kw))
    return out
