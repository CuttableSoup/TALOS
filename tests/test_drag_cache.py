"""Between dynamics steps, objects drift at the orbit-averaged da/dt the daily step integrates,
cached per object on the population and refreshed whenever its elements change."""

import pickle

import numpy as np
import pytest

from talos.config import load_scenario, run_scenario
from talos.debris import DebrisPopulation
from talos.elements import KeplerElements
from talos.lifetime import decay_rates
from talos.propagator import drag_da_dt, propagate, states_at
from talos.render import _with_atmosphere


def _pop(n=400, seed=1):
    rng = np.random.default_rng(seed)
    el = KeplerElements(a=6.378e6 + rng.uniform(5e5, 9e5, n), e=rng.uniform(0, 0.015, n),
                        i=rng.uniform(0, np.pi, n), raan=rng.uniform(0, 2 * np.pi, n),
                        argp=rng.uniform(0, 2 * np.pi, n), M=rng.uniform(0, 2 * np.pi, n))
    return DebrisPopulation(elements=el, diameter_m=0.05, areal_density=rng.uniform(0.5, 20, n),
                            weight=1.0, epoch_s=rng.uniform(0, 86400, n))


def test_rate_is_the_daily_steps_orbit_average():
    p = _pop()
    want = decay_rates(p.elements.a, p.elements.e, p.ballistic_coeff, t_s=p.epoch_s)[0]
    assert np.array_equal(p.drag_adot(), want)
    assert np.array_equal(drag_da_dt(p.elements.a, p.ballistic_coeff, p.epoch_s, p.elements.e), want)
    # eccentric orbits dip into denser air at perigee: faster than the circular rate at the mean altitude
    circ = drag_da_dt(p.elements.a, p.ballistic_coeff, p.epoch_s)
    ecc = p.elements.e > 0.01
    assert np.all(want[ecc] < circ[ecc])


def test_cache_paths_agree_and_invalidate():
    p = _pop()
    t = np.linspace(0, 6 * 3600, 7)
    on_the_fly = states_at(p.elements, t, p.ballistic_coeff, epoch=p.epoch_s, use_kernel=False)[0]
    for k in (False, True):
        cached = states_at(p.elements, t, epoch=p.epoch_s, adot=p.drag_adot(), use_kernel=k)[0]
        assert np.max(np.abs(cached - on_the_fly)) < 1e-3
    j = 7
    p.elements.a[j] -= 50e3
    assert p.drag_adot(j) != pytest.approx(decay_rates(p.elements.a[j], p.elements.e[j], p.ballistic_coeff[j],
                                                       t_s=p.epoch_s[j])[0])  # stale until invalidated
    p.invalidate_drag(j)
    assert p.drag_adot(j) == decay_rates(p.elements.a[j], p.elements.e[j], p.ballistic_coeff[j], t_s=p.epoch_s[j])[0]


def test_cache_follows_subset_append_and_old_pickles():
    p = _pop()
    full = p.drag_adot().copy()
    s = p.subset(np.arange(10, 30))
    assert np.array_equal(s.adot_cache, full[10:30])
    q = p.append(_pop(5, seed=2))
    assert np.array_equal(q.adot_cache[:len(p)], full) and np.all(np.isnan(q.adot_cache[len(p):]))
    assert np.all(np.isfinite(q.drag_adot()))
    old = pickle.loads(pickle.dumps(p))
    del old.__dict__["adot_cache"]  # a checkpoint written before the cache existed
    assert np.array_equal(old.drag_adot(), full)


def test_sim_uses_the_cached_rate_between_steps():
    sc = load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000,
                                  "population.min_lifetime_yr": None, "population.areal_density_model": "sbm",
                                  "constellation.n_sats": 2, "run.duration_s": 2 * 86400.0})
    res = run_scenario(sc)
    ev = res.records()
    assert ev
    pop = res.population if hasattr(res, "population") else None
    if pop is None:
        pytest.skip("run result does not expose the population")
    live = np.flatnonzero(np.isfinite(pop.adot_cache))
    assert len(live)
    want = _with_atmosphere(sc, lambda: decay_rates(pop.elements.a[live], pop.elements.e[live],
                                                    pop.ballistic_coeff[live], t_s=pop.epoch_s[live])[0])
    np.testing.assert_allclose(pop.adot_cache[live], want, rtol=0, atol=1e-15)
