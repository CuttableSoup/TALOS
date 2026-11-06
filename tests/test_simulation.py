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
    res = run_engagements([sat], pop, duration_s=600.0, params=AblationParams(eta1_mean=1.0, off_axis_mean_rad=0.0, off_axis_sigma_rad=0.0))
    assert all(e.t_stop - e.t_start >= sat.resources.min_burst_s - 1e-9 for e in res.events)
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
    # initial empty heat buffer allows a little over the steady-state ceiling
    assert 0 < s["duty_cycle_mean"] <= 1.05 * res.max_duty
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


def test_engagement_rate_respects_thermal_ceiling():
    sat = baseline_platform(800.0)
    pop = toy_population(20000, np.random.default_rng(5))
    res = run_engagements([sat], pop, 3 * 3600.0)
    s = res.summary()
    r, atp = sat.resources, sat.atp
    # each full burst needs ~burst_s/max_duty of wall time; shorter bursts are >= min_burst_s
    cap_per_hr = 3600.0 / (r.min_burst_s / res.max_duty)
    assert s["useful_engagements_per_hr_per_sat"] <= cap_per_hr
    assert s["duty_cycle_mean"] <= 1.05 * res.max_duty


def test_removal_credit_excludes_short_lived():
    sat = baseline_platform(800.0)
    pop = coorbital_pop(sat, -1.5, mu=0.2)  # 800 km, mu=0.2: natural lifetime ~0.4 yr
    res = run_engagements([sat], pop, duration_s=6 * 3600.0, chunk_s=300.0)
    s = res.summary()
    assert s["removed"] == 1
    assert s["removed_credited"] == 0
    assert s["debris_years_removed"] > 0
    assert res.summary(credit_min_lifetime_s=0.0)["removed_credited"] == 1


def test_toy_population_min_lifetime_filter():
    from talos.constants import YEAR
    from talos.lifetime import orbital_lifetime

    pop = toy_population(2000, np.random.default_rng(2), min_lifetime_yr=1.0)
    assert 0 < len(pop) < 2000
    life = orbital_lifetime(pop.elements.a, pop.elements.e, pop.ballistic_coeff, max_s=2 * YEAR)
    assert np.all(life > YEAR)
