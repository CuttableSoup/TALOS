"""Golden-event regression: a short toy run must reproduce the committed event log.

The scenario exercises screening, scheduling, bursts, the resource state machine, drag,
breakups, natural decay and ablation fragments, and needs no external table (toy population,
parametric solar cycle). It runs on whichever kernel path is active, so running the suite with
and without ``TALOS_NUMBA=0`` checks both against one reference.

Discrete columns must match exactly. Floats are compared to 1e-9 relative, which allows for
last-bit libm differences between platforms and nothing else; on one machine the numba and
numpy paths give identical bits.

When a change is meant to alter results, rebuild the reference and commit it with the change:

    TALOS_REGOLDEN=1 pytest tests/test_golden.py        (PowerShell: $env:TALOS_REGOLDEN=1)

and say in the commit message why the events moved. ``git diff tests/golden`` shows what did.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from talos.config import load_scenario, run_scenario

REF = Path(__file__).parent / "golden" / "toy_events.csv"
SUMMARY_REF = Path(__file__).parent / "golden" / "toy_summary.csv"
OVERRIDES = {
    "population.kind": "toy", "population.n_real": 10000, "population.min_lifetime_yr": None,
    "population.areal_density_model": "sbm", "constellation.n_sats": 2,
    "run.duration_s": 4 * 3600.0, "run.seed": 7,
    "dynamics.update_days": 0.03, "dynamics.breakup_rate_per_yr": 30000.0, "dynamics.breakup_fragments": 50,
    "ablation.fragment_prob_per_burst": 0.05,
    "atmosphere.activity": "cycle",  # no F10.7 table needed
    "survival.enabled": False,  # no platform flux table needed
}
SUMMARY_KEYS = ["useful_engagements", "removed", "debris_years_removed", "duty_cycle_mean",
                "mean_dv_retro_per_engagement", "natural_decays", "breakups", "added_breakup",
                "ablation_fragmentations", "ablation_fragments", "pop_final_live"]


@pytest.fixture(scope="module")
def run():
    res = run_scenario(load_scenario(overrides=OVERRIDES))
    events = pd.DataFrame(res.records())
    s = res.summary()
    summary = pd.DataFrame({"key": SUMMARY_KEYS, "value": [float(s[k]) for k in SUMMARY_KEYS]})
    if os.environ.get("TALOS_REGOLDEN") == "1":
        REF.parent.mkdir(exist_ok=True)
        events.to_csv(REF, index=False, float_format="%.17g", lineterminator="\n")
        summary.to_csv(SUMMARY_REF, index=False, float_format="%.17g", lineterminator="\n")
    return events, summary


def test_events_match_the_golden_log(run):
    events, _ = run
    ref = pd.read_csv(REF)
    assert list(events.columns) == list(ref.columns)
    assert len(events) == len(ref), f"{len(events)} events, reference has {len(ref)}"
    for col in ref.columns:
        got, want = events[col].to_numpy(), ref[col].to_numpy()
        if want.dtype.kind == "f":
            np.testing.assert_allclose(got.astype(float), want, rtol=1e-9, atol=0.0, equal_nan=True,
                                       err_msg=f"column {col}")
        else:
            assert (got.astype(want.dtype) == want).all(), f"column {col} differs"


def test_summary_matches_the_golden_run(run):
    _, summary = run
    ref = pd.read_csv(SUMMARY_REF)
    assert list(summary["key"]) == list(ref["key"])
    np.testing.assert_allclose(summary["value"], ref["value"], rtol=1e-9, atol=0.0)


def test_golden_run_exercises_the_model(run):
    """The reference is only a guard if the run does something: bursts, a removal, breakups,
    natural decays and ablation fragments all occur in it."""
    events, summary = run
    s = dict(zip(summary["key"], summary["value"]))
    assert len(events) > 50 and events["n_fired"].gt(0).sum() > 50
    assert s["removed"] >= 1 and s["breakups"] > 0 and s["natural_decays"] > 0
    assert s["ablation_fragments"] > 0
