import copy

import numpy as np
import pytest

from talos.atmosphere import Atmosphere, AtmosphereConfig
from talos.config import load_scenario
from talos.constants import DAY, R_EARTH, YEAR
from talos.debris import DebrisPopulation
from talos.dynamics import DynamicsConfig, PopulationDynamics, breakup_elements
from talos.elements import KeplerElements
from talos.lifetime import decay_to, orbital_lifetime
from talos.platform import baseline_platform
from talos.population import toy_population
from talos.propagator import drag_da_dt, exponential_density, get_atmosphere, set_atmosphere, states_at
from talos.simulation import run_engagements


@pytest.fixture
def msis():
    prev = get_atmosphere()
    atm = Atmosphere(AtmosphereConfig(), epoch_year=2026.97)
    set_atmosphere(atm)
    yield atm
    set_atmosphere(prev)


def test_msis_table_brackets_static_model():
    lo = Atmosphere(AtmosphereConfig(activity="constant", f107_constant=70.0))
    hi = Atmosphere(AtmosphereConfig(activity="constant", f107_constant=230.0))
    mid = Atmosphere(AtmosphereConfig(activity="constant", f107_constant=130.0))
    for h in (400e3, 800e3):
        assert lo.density(h) < exponential_density(h) < hi.density(h)
    assert mid.density(800e3) == pytest.approx(exponential_density(800e3), rel=0.3)
    assert hi.density(800e3) / lo.density(800e3) > 10  # strong solar-cycle swing at 800 km
    h = np.linspace(150e3, 1400e3, 50)
    assert np.all(np.diff(mid.density(h)) < 0)


def test_solar_cycle_shape():
    atm = Atmosphere(AtmosphereConfig(), epoch_year=2019.96)
    yr = lambda y: (y - 2019.96) * YEAR
    assert atm.f107(yr(2019.96)) == pytest.approx(70.0)
    assert atm.f107(yr(2024.8)) == pytest.approx(155.0)
    assert atm.f107(yr(2030.96)) == pytest.approx(70.0, abs=0.5)
    assert atm.f107(yr(2027.0)) < atm.f107(yr(2026.0)) < atm.f107(yr(2025.0))


def test_lifetime_depends_on_start_in_cycle(msis):
    a = R_EARTH + 600e3
    at_max = orbital_lifetime(a, 0.0, 2.2, t0_s=(2024.8 - 2026.97) * YEAR)
    at_min = orbital_lifetime(a, 0.0, 2.2, t0_s=(2030.96 - 2026.97) * YEAR)
    assert at_min > at_max  # starting into solar minimum lives longer


def test_per_object_epoch_kernel_matches_numpy():
    rng = np.random.default_rng(0)
    pop = toy_population(3000, rng)
    t = np.arange(0.0, 50.0) + 5 * DAY
    ep = rng.uniform(0, 5 * DAY, len(pop))
    rk, vk = states_at(pop.elements, t, pop.ballistic_coeff, use_kernel=True, epoch=ep)
    rn, vn = states_at(pop.elements, t, pop.ballistic_coeff, use_kernel=False, epoch=ep)
    assert np.allclose(rk, rn, rtol=0, atol=1e-3) and np.allclose(vk, vn, rtol=0, atol=1e-6)
    # the epoch is honoured: elements at epoch ep reproduce their own position at t = ep
    from talos.elements import elements_to_rv

    r_ep, _ = states_at(pop.elements[:5], ep[:5], use_kernel=False, epoch=ep[:5])
    assert np.allclose(np.diagonal(r_ep, axis1=0, axis2=1).T, elements_to_rv(pop.elements[:5])[0], atol=1e-3)


def test_decay_to_matches_drag_rate():
    el = KeplerElements(a=np.array([R_EARTH + 600e3]), e=np.array([0.0]), i=np.array([1.7]), raan=np.array([0.0]),
                        argp=np.array([0.0]), M=np.array([0.0]))
    out = decay_to(el, np.array([2.2]), 0.0, DAY)
    # ~1.5 km drop in a day; density rises ~2 % over it (scale height ~70 km)
    assert out.a[0] - el.a[0] == pytest.approx(drag_da_dt(el.a[0], 2.2) * DAY, rel=0.02)
    # very low object: no overshoot below Earth, stops at re-entry
    low = KeplerElements(a=np.array([R_EARTH + 180e3]), e=np.array([0.0]), i=np.array([1.7]), raan=np.array([0.0]),
                         argp=np.array([0.0]), M=np.array([0.0]))
    out = decay_to(low, np.array([2.2]), 0.0, 5 * DAY)
    assert np.isfinite(out.a[0]) and out.a[0] - R_EARTH < 125e3


def test_breakup_fragments_spread_around_parent():
    el = breakup_elements(np.random.default_rng(1), 500, 50.0)
    alt = (el.a - R_EARTH) / 1e3
    # 50 m/s per axis ~ +-100 km (1 sigma) in semi-major axis
    assert len(el) > 450 and 40 < np.std(alt) < 200 and 550 < np.median(alt) < 1050


def tiny_pop():
    return toy_population(5000, np.random.default_rng(7), n_real=5000)


def test_dynamics_step_adds_and_decays():
    pop = tiny_pop()
    # one very low object that must decay
    pop.elements.a[0] = R_EARTH + 180e3
    sc = load_scenario(overrides={"population.n_real": 5000})
    dyn = PopulationDynamics(DynamicsConfig(source_mode="fraction", background_frac_per_yr=10.0, breakup_rate_per_yr=0.0), sc.population,
                             1.0, np.random.default_rng(0))
    gone = np.zeros(len(pop), bool)
    new, dmask = dyn.step(copy.deepcopy(pop), gone, DAY)
    expect = 10.0 * 5000 / 365.25
    assert abs(dyn.stats["added_background"] - expect) < 5 * np.sqrt(expect)
    assert len(new) == len(pop) + dyn.stats["added_background"]
    assert dmask[0] and dmask.sum() == dyn.stats["natural_decays"]
    assert np.all(new.epoch_s[len(pop):] == DAY) and np.all(new.epoch_s[1:len(pop)] == DAY)


def test_simulation_with_dynamics_grows_population():
    sat = baseline_platform(800.0)
    pop = tiny_pop()
    sc = load_scenario(overrides={"population.n_real": 5000})
    dyn = PopulationDynamics(DynamicsConfig(update_s=1800.0, source_mode="fraction", background_frac_per_yr=5.0, breakup_rate_per_yr=50.0,
                                            breakup_fragments=200), sc.population, 1.0, np.random.default_rng(2))
    res = run_engagements([sat], pop, 6 * 3600.0, dynamics=dyn)
    s = res.summary()
    assert s["added_background"] > 0 and len(res.removed) == len(pop) + s["added_background"] + s["added_breakup"]
    assert s["pop_final_live"] == len(res.removed) - s["removed"] - s["natural_decays"]


def test_replace_decays_keeps_untouched_population_steady():
    pop = tiny_pop()
    pop.elements.a[:3] = R_EARTH + 180e3  # three objects that decay at once
    pop.areal_density[:3] = [0.3, 0.7, 2.0]
    sc = load_scenario(overrides={"population.n_real": 5000})
    dyn = PopulationDynamics(DynamicsConfig(source_mode="replace_decays", breakup_rate_per_yr=0.0),
                             sc.population, 1.0, np.random.default_rng(0))
    initial = copy.deepcopy(pop)
    engaged = np.zeros(len(pop), bool)
    engaged[2] = True  # laser-touched: its decay is not replaced
    new, dmask = dyn.step(copy.deepcopy(pop), np.zeros(len(pop), bool), DAY, initial=initial, engaged=engaged)
    assert dmask[:3].all() and dyn.stats["replaced_decays"] == dmask.sum() - 1
    assert dyn.stats["added_background"] == 0
    rep = new.subset(np.arange(len(pop), len(new)))
    assert len(rep) == dyn.stats["replaced_decays"]
    np.testing.assert_array_equal(rep.areal_density[:2], [0.3, 0.7])
    np.testing.assert_array_equal(rep.elements.a[:2], initial.elements.a[:2])
    assert np.all(rep.epoch_s == DAY)
    with pytest.raises(ValueError, match="initial"):
        dyn.step(copy.deepcopy(pop), np.zeros(len(pop), bool), 2 * DAY)
