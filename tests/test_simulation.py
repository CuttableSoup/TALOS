import numpy as np
import pytest

from talos.constants import R_EARTH
from talos.debris import DebrisPopulation
from talos.elements import KeplerElements
from talos.engagement import AblationParams
from talos.metrics import raw_encounter_rate, required_removal_rate
from talos.platform import baseline_platform
from talos.population import toy_population
from talos.simulation import run_engagements


def coorbital_pop(sat, dM_deg, mu=1.0):
    o = sat.orbit
    dM = np.deg2rad(np.atleast_1d(dM_deg))
    el = KeplerElements(a=o.a, e=0.0, i=o.i, raan=o.raan, argp=0.0, M=o.M + dM)
    return DebrisPopulation(elements=el, diameter_m=0.05, areal_density=mu, weight=1.0)


def test_trailing_object_is_engaged_and_lowered():
    sat = baseline_platform(800.0)
    pop = coorbital_pop(sat, [-1.5, 1.5])  # ~190 km behind / ahead
    res = run_engagements([sat], pop, duration_s=600.0, params=AblationParams(eta1_range=(1, 1)))
    targets = {e.debris_index for e in res.events}
    assert targets == {0}  # pushing the leading object would be prograde
    ev = res.events[0]
    assert ev.dv_retrograde > 0
    assert ev.perigee_alt_after_m < ev.perigee_alt_before_m
    assert res.population.elements.rp[0] < pop.elements.rp[0]
    assert res.population.elements.rp[1] == pytest.approx(pop.elements.rp[1])


def test_repeated_engagement_removes_light_object_and_duty_cycle_bounded():
    sat = baseline_platform(800.0)
    pop = coorbital_pop(sat, -1.5, mu=0.2)
    res = run_engagements([sat], pop, duration_s=6 * 3600.0, chunk_s=300.0)
    assert res.removed[0]
    s = res.summary()
    r = sat.resources
    assert 0 < s["duty_cycle_mean"] <= r.burst_s / (r.burst_s + r.cooldown_s) + 1e-9
    assert s["removed"] == 1


def test_stochastic_run_reproducible(rng):
    pop = toy_population(300, rng)
    sat = baseline_platform(800.0)
    a = run_engagements([sat], pop, 1200.0, rng=np.random.default_rng(1))
    b = run_engagements([sat], pop, 1200.0, rng=np.random.default_rng(1))
    assert a.records() == b.records()


def test_raw_encounter_rate_counts_entries():
    sat = baseline_platform(800.0)
    # Antipodal object never enters; nothing else -> zero rate.
    pop = coorbital_pop(sat, 180.0)
    assert raw_encounter_rate(sat, pop, np.arange(0.0, 600.0)) == 0.0


def test_required_removal_rate():
    assert required_removal_rate(1e6, 10.0) == pytest.approx(1e6 * np.log(10) / 10)
