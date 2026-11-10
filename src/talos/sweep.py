"""Module 9 — parallel sweep driver.

Runs are independent, so the sweep fans out over worker processes; inside each
worker the numba kernels use ``threads_per_job`` threads. The default split uses
every logical core: n_jobs * threads_per_job = os.cpu_count().

Outputs (in ``out_dir``):
  * summary.parquet: one row per run (config + SimResult.summary()).
  * events/<name>.parquet: per-engagement event log for playback/analysis.
  * logs/<name>.log: one progress line per ``log_every_s`` of sim time (sim day,
    wall time, engagements, removals, live population), written while the run goes.
  * events/<name>.partial.parquet: event checkpoint every ``checkpoint_s`` of sim
    time, so a long run can be inspected (or salvaged) before it finishes; removed
    once the final events file is written.
  * checkpoints/<name>.pkl: full sim state every ``run.checkpoint_s`` (default 30 d);
    ``RunConfig(..., resume=path)`` continues or extends the run from it.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import Scenario, load_scenario, run_scenario
from .constants import DAY


@dataclass(frozen=True)
class RunConfig:
    """One sweep point: a scenario TOML plus dotted overrides, e.g.
    ``RunConfig("hi_power", overrides={"laser.burst_power_kw": 50})``."""

    name: str
    config: str | None = None  # TOML path; None -> config/baseline.toml
    overrides: Mapping[str, Any] = field(default_factory=dict)
    resume: str | None = None  # checkpoint to continue from (see run_scenario)

    def scenario(self) -> Scenario:
        return load_scenario(self.config, {"run.name": self.name, **self.overrides})


class RunMonitor:
    """``on_chunk`` hook: progress log lines and periodic event checkpoints for one run."""

    def __init__(self, name: str, out_dir: str | os.PathLike, duration_s: float,
                 log_every_s: float = DAY, checkpoint_s: float = 7 * DAY):
        out = Path(out_dir)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "events").mkdir(parents=True, exist_ok=True)
        self.name, self.duration_s = name, duration_s
        self.log_path = out / "logs" / f"{name}.log"
        self.partial_path = out / "events" / f"{name}.partial.parquet"
        self.log_every_s, self.checkpoint_s = log_every_s, checkpoint_s
        self.next_log, self.next_ckpt = log_every_s, checkpoint_s
        self.t0 = time.perf_counter()
        self.t_first = None  # sim time of the first chunk end seen (a resumed run starts late)
        self.log_path.write_text(f"# {name}: sim_day wall_h engagements useful removed live_pop eta_h\n")

    def __call__(self, t_s, events, removed, decayed) -> None:
        end = t_s >= self.duration_s
        if self.t_first is None:
            self.t_first = t_s
            self.next_log = max(self.next_log, (t_s // self.log_every_s + 1) * self.log_every_s)
            self.next_ckpt = max(self.next_ckpt, (t_s // self.checkpoint_s + 1) * self.checkpoint_s)
        if t_s >= self.next_log or end:
            wall = time.perf_counter() - self.t0
            useful = sum(1 for e in events if e.n_fired > 0 and e.dv_retrograde > 0)
            done = t_s - self.t_first
            eta = wall * (self.duration_s - t_s) / done if done > 0 else float("nan")
            with self.log_path.open("a") as f:
                f.write(f"{t_s / DAY:8.2f} {wall / 3600:7.3f} {len(events):8d} {useful:8d} "
                        f"{int(removed.sum()):6d} {int((~(removed | decayed)).sum()):7d} {eta / 3600:7.2f}\n")
            self.next_log = (t_s // self.log_every_s + 1) * self.log_every_s
        if t_s >= self.next_ckpt and not end:
            import pandas as pd

            tmp = self.partial_path.with_suffix(".tmp")
            pd.DataFrame([{"run": self.name, **asdict(e)} for e in events]).to_parquet(tmp, index=False)
            os.replace(tmp, self.partial_path)
            self.next_ckpt = (t_s // self.checkpoint_s + 1) * self.checkpoint_s


def run_one(cfg: RunConfig, out_dir: str | os.PathLike | None = None) -> tuple[dict, list[dict]]:
    """Build and run ``cfg``'s scenario; return (summary row, events). With ``out_dir``,
    progress goes to ``out_dir/logs/<name>.log`` and events are checkpointed (RunMonitor)."""
    t0 = time.perf_counter()
    sc = cfg.scenario()
    mon = RunMonitor(cfg.name, out_dir, sc.run.duration_s) if out_dir is not None else None
    ckpt = None
    if out_dir is not None and sc.run.checkpoint_s:
        (Path(out_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        ckpt = Path(out_dir) / "checkpoints" / f"{cfg.name}.pkl"
    res = run_scenario(sc, on_chunk=mon, checkpoint_path=ckpt, resume_path=cfg.resume)
    ev_c = [e for e in res.events if e.committed]
    row = {"name": cfg.name, "config": cfg.config or "baseline",
           **{f"ovr.{k}": (str(v) if isinstance(v, (tuple, list)) else v) for k, v in cfg.overrides.items()},
           **res.summary(), **{f"planner_{k}": v for k, v in res.planner_stats.items()},
           "committed_engagement_frac": len(ev_c) / max(len(res.events), 1),
           "wall_s": time.perf_counter() - t0}
    from .cost import evaluate

    row |= evaluate(sc, row)
    events = [{"run": cfg.name, **e} for e in res.records()]
    return row, events


def _save_events(out_dir, name: str, ev: list[dict]) -> None:
    """Final events for one run, written as soon as it finishes (replaces the checkpoint)."""
    if out_dir is None:
        return
    import pandas as pd

    d = Path(out_dir) / "events"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev).to_parquet(d / f"{name}.parquet", index=False)
    (d / f"{name}.partial.parquet").unlink(missing_ok=True)


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

    rows = []
    if n_jobs == 1:
        _init_worker(threads_per_job)
        for c in configs:
            r, ev = run_one(c, out_dir)
            rows.append(r)
            _save_events(out_dir, c.name, ev)
            if progress:
                print(f"[{len(rows)}/{len(configs)}] {c.name} {r['wall_s']:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(threads_per_job,)) as ex:
            futs = {ex.submit(run_one, c, out_dir): c for c in configs}
            for f in as_completed(futs):
                r, ev = f.result()
                rows.append(r)
                _save_events(out_dir, futs[f].name, ev)
                if progress:
                    print(f"[{len(rows)}/{len(configs)}] {futs[f].name} {r['wall_s']:.0f}s", flush=True)

    df = pd.DataFrame(rows).set_index("name").loc[names].reset_index()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "summary.parquet", index=False)
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


# Uncertain inputs (README Placeholders): key -> (lo, hi[, "log"]). Ranges are the
# sourced ranges where the table gives one, otherwise picks around the baseline.
UNCERTAIN_INPUTS: dict[str, tuple] = {
    "laser.eta2": (0.30, 0.67),
    "ablation.cm_n_per_mw": (25.0, 100.0, "log"),
    "ablation.eta1_mean": (0.2, 0.4),  # pick around the sourced 0.3
    "ablation.off_axis_mean_deg": (30.0, 60.0),  # pick around the sourced 45
    "resources.wallplug_eff": (0.04, 0.17, "log"),
    "atp.sun_exclusion_deg": (30.0, 45.0),
    "atp.settle_s": (2.0, 10.0),  # guess
}


def lhs_matrix(n: int, ranges: Mapping[str, tuple] | None = None, base: RunConfig = RunConfig(name="lhs"),
               seed: int = 0, vary_seed: bool = True) -> list[RunConfig]:
    """Latin-hypercube sample of ``ranges`` (default UNCERTAIN_INPUTS; roadmap 7): n runs,
    each axis cut into n equal-probability strata (log-uniform for "log" axes), one sample
    per stratum, strata shuffled independently per axis. ``vary_seed`` also gives every run
    its own ``run.seed`` so the bands include Monte Carlo noise."""
    import numpy as np

    ranges = dict(UNCERTAIN_INPUTS if ranges is None else ranges)
    rng = np.random.default_rng(seed)
    cols = {}
    for key, spec in ranges.items():
        lo, hi, *kind = spec
        u = (rng.permutation(n) + rng.uniform(size=n)) / n
        cols[key] = (lo * (hi / lo) ** u) if kind == ["log"] else lo + (hi - lo) * u
    out = []
    for j in range(n):
        ovr = {**base.overrides, **{k: float(v[j]) for k, v in cols.items()}}
        if vary_seed:
            ovr["run.seed"] = seed * 100003 + j
        out.append(RunConfig(name=f"{base.name}_{j:03d}", config=base.config, overrides=ovr))
    return out


def uncertainty_bands(df, metrics: list[str], q=(0.05, 0.5, 0.95)):
    """Quantiles of ``metrics`` over an LHS sweep, plus each input's Spearman rank
    correlation with each metric (which inputs drive the spread)."""
    import pandas as pd

    inputs = [c for c in df.columns if c.startswith("ovr.") and c != "ovr.run.seed"
              and df[c].nunique() > 1 and pd.api.types.is_numeric_dtype(df[c])]
    bands = df[metrics].quantile(list(q)).T
    bands.columns = [f"q{int(round(x * 100)):02d}" for x in q]
    rank = df[inputs + metrics].rank()
    corr = rank.corr().loc[[c for c in inputs], metrics]
    corr.index = [c[4:] for c in corr.index]
    return bands, corr
