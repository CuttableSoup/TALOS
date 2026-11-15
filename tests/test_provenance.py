"""Run provenance: summary rows and checkpoints say which code, config and data made them."""

import warnings

import pandas as pd
import pytest

from talos import provenance
from talos.config import load_scenario, run_scenario
from talos.simulation import load_checkpoint
from talos.sweep import RunConfig, run_sweep

TOY = {"population.kind": "toy", "population.n_real": 3000, "population.min_lifetime_yr": None,
       "constellation.n_sats": 1, "atmosphere.activity": "cycle", "survival.enabled": False,
       "run.checkpoint_s": 600.0}


def sc(seconds=900.0, **kw):
    return load_scenario(overrides={**TOY, "run.duration_s": seconds, **kw})


def test_collect_reports_code_config_and_data():
    p = provenance.collect(sc())
    assert {"git_sha", "git_dirty", "config_hash", "seed", "eta1_basis", "population_table_sha",
            "f107_table_sha", "density_table_sha", "talos", "python", "numpy", "numba"} <= set(p)
    assert p["seed"] == 0 and p["eta1_basis"] == "along_beam"
    assert p["population_table_sha"] == "" and p["f107_table_sha"] == ""  # toy sky, parametric cycle
    assert len(p["density_table_sha"]) == 16  # the shipped MSIS table
    sha, dirty = provenance.git_state()
    assert (sha is None and dirty is None) or (len(sha) == 40 and isinstance(dirty, bool))


def test_config_hash_tracks_inputs_but_not_the_run_name():
    base = provenance.config_hash(sc())
    assert provenance.config_hash(sc(**{"run.name": "other"})) == base
    assert provenance.config_hash(sc(**{"run.seed": 1})) != base
    assert provenance.config_hash(sc(**{"laser.eta2": 0.41})) != base
    # unit and SI spellings of one value resolve to the same scenario
    assert (provenance.config_hash(sc(**{"laser.burst_power_kw": 20.0}))
            == provenance.config_hash(sc(**{"laser.burst_power_w": 20000.0})))


def test_missing_table_is_reported_not_raised(tmp_path):
    assert provenance.file_sha(tmp_path / "nope.npz") == "missing"
    f = tmp_path / "t.bin"
    f.write_bytes(b"abc")
    assert provenance.file_sha(f) == "ba7816bf8f01cfea"


def test_summary_rows_and_checkpoints_carry_provenance(tmp_path):
    df = run_sweep([RunConfig("p", overrides={**TOY, "run.duration_s": 900.0})], tmp_path,
                   n_jobs=1, threads_per_job=1, progress=False)
    want = provenance.collect(sc(**{"run.name": "p"}))
    row = pd.read_parquet(tmp_path / "summaries" / "p.parquet").iloc[0]
    for k in ("git_sha", "config_hash", "eta1_basis", "density_table_sha", "numpy"):
        assert df.iloc[0][f"prov.{k}"] == want[k] and row[f"prov.{k}"] == want[k]
    assert row["prov.seed"] == 0
    meta = load_checkpoint(tmp_path / "checkpoints" / "p.pkl")["meta"]
    assert meta["provenance"]["config_hash"] == want["config_hash"]


def _rewrite_provenance(ck, **changes):
    import pickle

    blob = load_checkpoint(ck)
    if changes.get("drop"):
        del blob["meta"]["provenance"]
    else:
        blob["meta"]["provenance"] = {**blob["meta"]["provenance"], "git_dirty": False, **changes}
    with open(ck, "wb") as fh:
        pickle.dump(blob, fh)


def test_resume_from_other_code_warns_or_raises(tmp_path, monkeypatch):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(), checkpoint_path=ck)
    monkeypatch.setattr(provenance, "git_state", lambda: ("b" * 40, False))
    _rewrite_provenance(ck, git_sha="a" * 40)
    with pytest.warns(UserWarning, match="cannot be matched to this code and data .*git_sha"):
        res = run_scenario(sc(1200.0), resume_path=ck)
    assert res.provenance["resumed_from_git_sha"] == "a" * 40
    assert res.provenance["resumed_code_changed"] is True
    _rewrite_provenance(ck, git_sha="a" * 40)  # the resumed run did not write a new checkpoint
    with pytest.raises(ValueError, match="strict_resume"):
        run_scenario(sc(1200.0), resume_path=ck, strict_resume=True)


def test_resume_from_the_same_clean_code_is_silent(tmp_path, monkeypatch):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(), checkpoint_path=ck)
    monkeypatch.setattr(provenance, "git_state", lambda: ("b" * 40, False))
    _rewrite_provenance(ck, git_sha="b" * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run_scenario(sc(1200.0), resume_path=ck, strict_resume=True)
    assert res.provenance["resumed_code_changed"] is False


def test_checkpoints_without_provenance_still_resume(tmp_path):
    ck = tmp_path / "c.pkl"
    run_scenario(sc(), checkpoint_path=ck)
    _rewrite_provenance(ck, drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run_scenario(sc(1200.0), resume_path=ck, strict_resume=True)
    assert res.provenance["resumed_from_git_sha"] == "unknown"
    assert res.provenance["resumed_code_changed"] is None
