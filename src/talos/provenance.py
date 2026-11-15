"""Run provenance: which code, configuration and data tables produced a result.

``collect(sc)`` returns a flat dict that goes into every summary row (as ``prov.<key>``
columns) and into every checkpoint (``meta["provenance"]``), so a result can be matched to
the code that made it without reading dates:

  * ``git_sha`` / ``git_dirty``: commit of the working tree and whether it had uncommitted
    changes to tracked files (None when this is not a git checkout or git is missing).
  * ``config_hash``: SHA-256 (first 16 hex) of the resolved scenario, every key after unit
    conversion and defaults, without ``run.name``. Equal hashes mean equal inputs.
  * ``seed``, ``eta1_basis``: copied out of the scenario because results depend on them in
    ways that changed over the project's history.
  * ``population_table_sha`` / ``f107_table_sha`` / ``density_table_sha``: SHA-256 (first
    16 hex) of the data files the scenario reads ("" when the scenario does not use that
    table, "missing" when the file is absent). The platform flux table is identified by the
    ``impact_flux_year`` summary column.
  * ``talos`` / ``python`` / ``numpy`` / ``numba``: versions (numba "off" on the numpy path).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def git_state() -> tuple[str | None, bool | None]:
    """(commit SHA, dirty flag) of the checkout this package runs from; (None, None) if unknown."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(_ROOT), *args], capture_output=True, text=True,
                              check=True, timeout=20).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != _ROOT:
            return None, None  # installed inside some other repository
        sha = git("rev-parse", "HEAD")
        return sha, bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None, None


@lru_cache(maxsize=16)
def _file_sha(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def file_sha(path) -> str:
    """Short SHA-256 of a data file; "missing" if it does not exist."""
    p = Path(path)
    if not p.exists():
        return "missing"
    st = p.stat()
    return _file_sha(str(p), st.st_mtime_ns, st.st_size)


def config_hash(sc) -> str:
    """Short SHA-256 of the resolved scenario (``run.name`` excluded)."""
    from .config import scenario_dict

    d = {k: v for k, v in scenario_dict(sc).items() if k != "run.name"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _versions() -> dict:
    import numpy as np

    from . import _kernels

    try:
        from importlib.metadata import version

        talos = version("talos")
    except Exception:  # not pip-installed (PYTHONPATH=src)
        talos = "uninstalled"
    return {"talos": talos, "python": platform.python_version(), "numpy": np.__version__,
            "numba": _kernels.nb.__version__ if _kernels.HAVE_NUMBA else "off"}


def collect(sc) -> dict:
    """Provenance of a run of scenario ``sc`` in this process (see the module docstring)."""
    from . import atmosphere, ordem

    sha, dirty = git_state()
    pop_sha = f107_sha = rho_sha = ""
    if sc.population.kind == "ordem":
        pop_sha = file_sha(ordem.table_path(int(sc.population.ordem_year or sc.run.epoch.year)))
    if sc.atmosphere.model == "msis_table":
        rho_sha = file_sha(atmosphere._TABLE)
    if sc.atmosphere.activity == "table":
        f107_sha = file_sha(sc.atmosphere.f107_table or atmosphere._F107_TABLE)
    return {"git_sha": sha, "git_dirty": dirty, "config_hash": config_hash(sc), "seed": sc.run.seed,
            "eta1_basis": sc.ablation.eta1_basis, "population_table_sha": pop_sha,
            "f107_table_sha": f107_sha, "density_table_sha": rho_sha,
            **_versions()}


def describe_change(old: dict, new: dict) -> str:
    """What differs between a checkpoint's provenance and the resuming process ("" = nothing
    that changes results). Compares the commit, the dirty flag and the data tables."""
    keys = ("git_sha", "git_dirty", "population_table_sha", "f107_table_sha", "density_table_sha")
    out = []
    for k in keys:
        a, b = old.get(k), new.get(k)
        if a != b and not (k == "git_dirty" and not a and not b):
            out.append(f"{k}: {a!r} -> {b!r}")
    if not out and new.get("git_dirty"):
        out.append("same commit, but the working tree has uncommitted changes")
    return ", ".join(out)
