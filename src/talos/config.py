"""Scenario configuration: TOML file -> typed dataclasses -> runnable objects.

    sc = load_scenario()                                   # config/baseline.toml
    sc = load_scenario("my.toml", {"laser.burst_power_kw": 10, "constellation.n_sats": 8})
    res = run_scenario(sc)

Keys map to dataclass fields. A key with a unit suffix is converted to the
SI field (``burst_power_kw`` -> ``burst_power_w``); see ``_SUFFIXES`` and
``_ALIASES``. Unknown keys raise. Overrides use dotted ``section.key`` names
and go through the same conversion, so either the unit or the SI form works.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from .constants import FT2_TO_M2
from .engagement import AblationParams
from .platform import ATPConfig, LaserConfig, Platform, ResourceConfig, baseline_constellation
from .atmosphere import Atmosphere, AtmosphereConfig, decimal_year
from .dynamics import DynamicsConfig, PopulationDynamics
from .sun import SunConfig, SunEnvironment

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "baseline.toml"


@dataclass(frozen=True)
class RunSettings:
    name: str = "baseline"
    epoch: datetime = datetime(2026, 12, 21, tzinfo=timezone.utc)
    duration_s: float = 86400.0
    seed: int = 0
    stochastic: bool = True
    score: str = "removal_progress"
    dt_s: float = 1.0
    chunk_s: float = 600.0


@dataclass(frozen=True)
class ConstellationConfig:
    n_sats: int = 4
    altitudes_km: tuple[float, ...] = (700.0, 800.0, 900.0)
    ltan_hours: float = 6.0


@dataclass(frozen=True)
class PopulationConfig:
    kind: str = "toy"  # toy | ordem
    n_real: float = 1.0e5  # toy only; for "ordem" the count comes from the table
    n_samples: int | None = None  # None -> full density (one sample per real object)
    ordem_year: int | None = None  # None -> the epoch's year
    materials: tuple[str, ...] | None = None  # ORDEM classes (NK LD MD HD IN); None -> all
    areal_density_model: str = "fixed"  # fixed | sphere (ORDEM material density; ordem only)
    alt_km: tuple[float, float] = (600.0, 1000.0)
    areal_density: float = 1.0
    min_lifetime_yr: float | None = 1.0
    albedo_median: float = 0.1
    albedo_log_sigma: float = 0.5


@dataclass(frozen=True)
class PlannerConfig:
    enabled: bool = False
    max_committed: int = 200
    horizon_s: float = 6 * 3600.0
    min_dv_frac: float = 0.25
    quality: float = 0.8
    probe_s: float = 10.0


@dataclass(frozen=True)
class RiskConfig:
    """Risk transfer (roadmap 6): time engaged objects spend inside protected altitude
    shells after the laser vs on their natural decay. Shell bounds are PICKS."""

    enabled: bool = True
    shell_names: tuple[str, ...] = ("crewed", "starlink")
    shell_lo_m: tuple[float, ...] = (370e3, 530e3)
    shell_hi_m: tuple[float, ...] = (430e3, 575e3)

    def shells(self) -> dict[str, tuple[float, float]] | None:
        if not self.enabled or not self.shell_names:
            return None
        if not len(self.shell_names) == len(self.shell_lo_m) == len(self.shell_hi_m):
            raise ValueError("[risk] shell_names / shell_lo / shell_hi lengths differ")
        return {n: (lo, hi) for n, lo, hi in zip(self.shell_names, self.shell_lo_m, self.shell_hi_m)}


@dataclass(frozen=True)
class Scenario:
    run: RunSettings = field(default_factory=RunSettings)
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    atp: ATPConfig = field(default_factory=ATPConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    ablation: AblationParams = field(default_factory=AblationParams)
    sun: SunConfig = field(default_factory=SunConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


# suffix -> (SI suffix, converter); longest suffix wins
_SUFFIXES: dict[str, tuple[str, Any]] = {
    "_deg_s": ("_rad_s", np.deg2rad),
    "_deg": ("_rad", np.deg2rad),
    "_km": ("_m", lambda x: x * 1e3),
    "_kw": ("_w", lambda x: x * 1e3),
    "_kwh": ("_j", lambda x: x * 3.6e6),
    "_ft2": ("_m2", lambda x: x * FT2_TO_M2),
    "_hours": ("_s", lambda x: x * 3600.0),
    "_days": ("_s", lambda x: x * 86400.0),
    "_c": ("_k", lambda x: x + 273.15),
    "_nm": ("_m", lambda x: x * 1e-9),
    "_ps": ("_s", lambda x: x * 1e-12),
    "_urad": ("_rad", lambda x: x * 1e-6),
}
# keys whose SI field has no unit suffix
_ALIASES: dict[str, tuple[str, Any]] = {
    "cm_n_per_mw": ("cm", lambda x: x * 1e-6),
    "threshold_fluence_kj_m2": ("threshold_fluence", lambda x: x * 1e3),
    "phi_opt_kj_m2": ("phi_opt", lambda x: x * 1e3),
    "heat_capacity_mj_k": ("heat_capacity_j_k", lambda x: x * 1e6),
    "pcm_specific_kj_kg": ("pcm_specific_j_kg", lambda x: x * 1e3),
}


def _resolve(cls, key: str, value):
    """Map (key, value) onto a field of dataclass ``cls``."""
    names = {f.name for f in fields(cls)}
    if key in names:
        return key, value
    if key in _ALIASES:
        k, conv = _ALIASES[key]
        if k in names:
            return k, (None if value is None else conv(value))
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if key.endswith(suf):
            si, conv = _SUFFIXES[suf]
            k = key[: -len(suf)] + si
            if k in names:
                if isinstance(value, (list, tuple)):
                    return k, tuple(float(x) for x in conv(np.asarray(value, float)))
                return k, (None if value is None else conv(value))
    raise KeyError(f"unknown key {key!r} for [{cls.__name__}]; fields: {sorted(names)}")


def _coerce(cls, values: dict) -> dict:
    hints = typing.get_type_hints(cls)
    out = {}
    for k, v in values.items():
        h = hints.get(k)
        if isinstance(v, list):
            v = tuple(v)
        if h is datetime and isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if h is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[k] = v
    return out


def _section(cls, raw: Mapping[str, Any]) -> dict:
    vals = {}
    for k, v in raw.items():
        fk, fv = _resolve(cls, k, v)
        if fk in vals:
            raise KeyError(f"[{cls.__name__}] {fk!r} given twice (as {k!r})")
        vals[fk] = fv
    return vals


def load_scenario(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Scenario:
    """Read a scenario TOML (default: config/baseline.toml) and apply dotted overrides."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    types = {f.name: f.type for f in fields(Scenario)}
    hints = typing.get_type_hints(Scenario)
    unknown = set(raw) - set(types)
    if unknown:
        raise KeyError(f"unknown section(s) {sorted(unknown)} in {path}")
    vals = {sec: _section(hints[sec], raw.get(sec, {})) for sec in types}
    for dotted, v in (overrides or {}).items():
        sec, _, key = dotted.partition(".")
        if sec not in types or not key:
            raise KeyError(f"override {dotted!r}: expected 'section.key' with section in {sorted(types)}")
        fk, fv = _resolve(hints[sec], key, v)
        for other in [k for k in vals[sec] if k == fk]:
            del vals[sec][other]
        vals[sec][fk] = fv
    return Scenario(**{sec: hints[sec](**_coerce(hints[sec], vals[sec])) for sec in types})


def scenario_dict(sc: Scenario) -> dict:
    """Flat {section.field: value} view (for logging / parquet rows)."""
    out = {}
    for sec in fields(sc):
        for f in fields(getattr(sc, sec.name)):
            v = getattr(getattr(sc, sec.name), f.name)
            out[f"{sec.name}.{f.name}"] = str(v) if isinstance(v, (tuple, datetime)) else v
    return out


class Built(NamedTuple):
    platforms: list[Platform]
    population: Any
    params: AblationParams
    env: SunEnvironment | None
    planner: Any
    rng: np.random.Generator | None
    atmosphere: Atmosphere
    dynamics: PopulationDynamics | None


def make_atmosphere(sc: Scenario) -> Atmosphere:
    return Atmosphere(sc.atmosphere, decimal_year(sc.run.epoch))


def build(sc: Scenario) -> Built:
    """Objects for one run. The population's lifetime filter uses whatever atmosphere
    is installed; ``run_scenario`` installs the scenario's atmosphere first."""
    from .planner import CommitPlanner
    from .population import toy_population

    env = SunEnvironment(sc.sun, sc.run.epoch) if sc.sun.enabled else None
    raan = env.raan_for_ltan(sc.constellation.ltan_hours) if env is not None else 0.0
    plats = baseline_constellation(sc.constellation.n_sats, sc.constellation.altitudes_km, raan_rad=raan,
                                   laser=sc.laser, atp=sc.atp, resources=sc.resources)
    p = sc.population
    pop_rng = np.random.default_rng(sc.run.seed)
    source = None
    if p.kind == "toy":
        n = int(p.n_samples or p.n_real)
        if p.areal_density_model not in ("fixed", "sbm"):
            raise ValueError(f"toy population supports areal_density_model fixed|sbm, not {p.areal_density_model!r}")
        mu_fn = None
        if p.areal_density_model == "sbm":  # separate stream so the toy's own draws are unchanged
            from .population import sbm_area_to_mass

            mu_rng = np.random.default_rng(sc.run.seed + 3)
            mu_fn = lambda d: 1.0 / sbm_area_to_mass(d, mu_rng)  # noqa: E731
        pop = toy_population(n, pop_rng, alt_km=p.alt_km, n_real=p.n_real,
                             areal_density=p.areal_density, min_lifetime_yr=p.min_lifetime_yr,
                             albedo_median=p.albedo_median, albedo_log_sigma=p.albedo_log_sigma, mu_fn=mu_fn)
    elif p.kind == "ordem":
        from .ordem import OrdemTable
        from .population import ordem_population

        table = OrdemTable.load(int(p.ordem_year or sc.run.epoch.year))
        p = replace(p, n_real=table.total(p.alt_km, p.materials))
        n = int(p.n_samples or round(p.n_real))
        kw = dict(alt_km=p.alt_km, areal_density=p.areal_density, areal_density_model=p.areal_density_model,
                  materials=p.materials, albedo_median=p.albedo_median, albedo_log_sigma=p.albedo_log_sigma)
        pop = ordem_population(n, pop_rng, table, min_lifetime_yr=p.min_lifetime_yr, **kw)

        def source(k, rng, t, w):  # background objects: same distribution, at the run's sample weight
            new = ordem_population(k, rng, table, epoch_s=t, **kw)
            new.weight[:] = w
            return new
    else:
        raise NotImplementedError(f"population kind {p.kind!r}")
    pl = sc.planner
    planner = (CommitPlanner(plats, sc.ablation, max_committed=pl.max_committed, horizon_s=pl.horizon_s,
                             dt=sc.run.dt_s, min_dv_frac=pl.min_dv_frac, quality=pl.quality, probe_s=pl.probe_s)
               if pl.enabled else None)
    rng = np.random.default_rng(sc.run.seed + 1) if sc.run.stochastic else None
    dyn = (PopulationDynamics(sc.dynamics, p, float(p.n_real) / n, np.random.default_rng(sc.run.seed + 2),
                              source=source)
           if sc.dynamics.enabled else None)
    return Built(plats, pop, sc.ablation, env, planner, rng, make_atmosphere(sc), dyn)


def run_scenario(sc: Scenario, on_chunk=None):
    """Install the scenario atmosphere, build, run, restore the previous atmosphere."""
    from .propagator import get_atmosphere, set_atmosphere
    from .simulation import run_engagements

    prev = get_atmosphere()
    set_atmosphere(make_atmosphere(sc))
    try:
        b = build(sc)
        return run_engagements(b.platforms, b.population, sc.run.duration_s, dt=sc.run.dt_s,
                               chunk_s=sc.run.chunk_s, params=b.params, rng=b.rng, score=sc.run.score,
                               planner=b.planner, env=b.env, dynamics=b.dynamics,
                               on_chunk=on_chunk, risk_shells=sc.risk.shells())
    finally:
        set_atmosphere(prev)
