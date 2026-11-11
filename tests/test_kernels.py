import numpy as np
import pytest

from talos import _kernels
from talos.metrics import raw_encounter_rate
from talos.platform import baseline_platform
from talos.population import toy_population
from talos.propagator import states_at
from talos.screening import screen_windows

pytestmark = pytest.mark.skipif(not _kernels.HAVE_NUMBA, reason="numba not installed")


def test_states_kernel_matches_numpy(rng):
    pop = toy_population(500, rng)
    t = np.linspace(0.0, 3 * 86400.0, 40)
    r0, v0 = states_at(pop.elements, t, pop.ballistic_coeff, use_kernel=False)
    r1, v1 = states_at(pop.elements, t, pop.ballistic_coeff, use_kernel=True)
    assert np.allclose(r1, r0, rtol=0, atol=1e-4)
    assert np.allclose(v1, v0, rtol=0, atol=1e-7)


@pytest.mark.parametrize("coarse", [None, 30.0])
def test_kernel_screening_matches_numpy(coarse):
    pop = toy_population(5000, np.random.default_rng(11))
    sat = baseline_platform(800.0)
    t = np.arange(0.0, 900.0)
    tup = lambda ws: sorted((w.debris_index, w.start, w.stop) for w in ws)
    ref = tup(screen_windows(sat, pop, t, coarse_step_s=coarse, use_kernel=False))
    got = tup(screen_windows(sat, pop, t, coarse_step_s=coarse, use_kernel=True))
    assert len(ref) > 0
    assert got == ref


def test_kernel_respects_field_of_regard():
    from dataclasses import replace

    pop = toy_population(5000, np.random.default_rng(12))
    sat = baseline_platform(800.0)
    sat = replace(sat, atp=replace(sat.atp, field_of_regard_half_angle_rad=np.deg2rad(60)))
    t = np.arange(0.0, 600.0)
    tup = lambda ws: sorted((w.debris_index, w.start, w.stop) for w in ws)
    assert tup(screen_windows(sat, pop, t, use_kernel=True)) == tup(screen_windows(sat, pop, t, use_kernel=False))


def test_screen_with_precomputed_sat_states_matches():
    pop = toy_population(5000, np.random.default_rng(13))
    sat = baseline_platform(800.0)
    t = 123.0 + np.arange(0.0, 1000.0)
    tup = lambda ws: sorted((w.debris_index, w.start, w.stop) for w in ws)
    ref = tup(screen_windows(sat, pop, t, use_kernel=True))
    got = tup(screen_windows(sat, pop, t, use_kernel=True, sat_states=states_at(sat.orbit, t)))
    assert len(ref) > 0
    assert got == ref


def test_decay_rates_kernel_matches_numpy(monkeypatch):
    from talos.atmosphere import Atmosphere, AtmosphereConfig
    from talos.constants import R_EARTH, YEAR
    from talos.lifetime import decay_rates, orbital_lifetime
    from talos.propagator import get_atmosphere, set_atmosphere

    rng = np.random.default_rng(5)
    a = R_EARTH + rng.uniform(150e3, 2000e3, 3000)  # includes extrapolation above the table top
    e = rng.uniform(0.0, 0.2, 3000)
    B = rng.uniform(0.01, 5.0, 3000)
    t = rng.uniform(0.0, 20 * YEAR, 3000)
    prev = get_atmosphere()
    set_atmosphere(Atmosphere(AtmosphereConfig()))
    try:
        got = decay_rates(a, e, B, t_s=t)
        life_k = orbital_lifetime(a[:200], e[:200], B[:200], t0_s=t[:200])
        monkeypatch.setattr(_kernels, "HAVE_NUMBA", False)
        ref = decay_rates(a, e, B, t_s=t)
        life_n = orbital_lifetime(a[:200], e[:200], B[:200], t0_s=t[:200])
        scalar = decay_rates(R_EARTH + 500e3, 0.0, 1.0, t_s=0.0)
    finally:
        set_atmosphere(prev)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10)
    np.testing.assert_allclose(got[1], ref[1], rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(life_k, life_n, rtol=1e-9)
    assert np.shape(scalar[0]) == ()


def test_find_windows_kernel_matches_numpy(monkeypatch):
    from talos.screening import find_windows

    mask = np.random.default_rng(3).uniform(size=(300, 400)) < 0.3
    mask[:, 5] = True  # a window spanning the whole grid
    got = [(w.debris_index, w.start, w.stop) for w in find_windows(mask, 2)]
    monkeypatch.setattr(_kernels, "HAVE_NUMBA", False)
    ref = [(w.debris_index, w.start, w.stop) for w in find_windows(mask, 2)]
    assert got == ref and (5, 0, 300) in got


def test_shared_element_args_match_per_platform():
    pop = toy_population(5000, np.random.default_rng(13))
    pop.epoch_s[:] = np.random.default_rng(14).uniform(0, 3600, len(pop))
    sat = baseline_platform(800.0)
    t = np.arange(3600.0, 4500.0)
    from talos.screening import chunk_cache

    cand = np.flatnonzero(np.arange(len(pop)) % 3 != 0)
    tup = lambda ws: sorted((w.debris_index, w.start, w.stop) for w in ws)
    ref = tup(screen_windows(sat, pop, t, candidates=cand))
    assert len(ref) > 0
    assert tup(screen_windows(sat, pop, t, candidates=cand, cache=chunk_cache(pop))) == ref


@pytest.mark.parametrize("activity", ["cycle", "constant", "table"])
def test_lifetime_kernel_matches_numpy(monkeypatch, activity):
    from talos.atmosphere import Atmosphere, AtmosphereConfig
    from talos.constants import R_EARTH, YEAR
    from talos.lifetime import orbital_lifetime
    from talos.propagator import get_atmosphere, set_atmosphere

    rng = np.random.default_rng(8)
    n = 400
    rp = R_EARTH + rng.uniform(100e3, 900e3, n)  # includes objects already below re-entry
    ra = rp + rng.uniform(0, 600e3, n)
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    B = rng.uniform(0.05, 5.0, n)
    t0 = rng.uniform(0, 11 * YEAR, n)
    sh = np.array([[370e3, 430e3], [530e3, 575e3]])
    prev = get_atmosphere()
    set_atmosphere(Atmosphere(AtmosphereConfig(activity=activity)))
    try:
        lk, rk = orbital_lifetime(a, e, B, max_s=30 * YEAR, t0_s=t0, shells_m=sh)
        lk2 = orbital_lifetime(a, e, B, max_s=30 * YEAR, t0_s=t0)
        s_k = orbital_lifetime(R_EARTH + 500e3, 0.0, 1.0)
        monkeypatch.setattr(_kernels, "HAVE_NUMBA", False)
        ln, rn = orbital_lifetime(a, e, B, max_s=30 * YEAR, t0_s=t0, shells_m=sh)
        s_n = orbital_lifetime(R_EARTH + 500e3, 0.0, 1.0)
    finally:
        set_atmosphere(prev)
    assert np.any(np.isinf(ln)) and np.any(ln == 0) and np.any(np.isfinite(ln) & (ln > 0))
    np.testing.assert_array_equal(np.isinf(lk), np.isinf(ln))
    fin = np.isfinite(ln)
    np.testing.assert_allclose(lk[fin], ln[fin], rtol=1e-9)
    np.testing.assert_allclose(lk2, lk, rtol=0)
    np.testing.assert_allclose(rk, rn, rtol=1e-6, atol=1.0)  # 1 s: shell-skip shortcut
    assert isinstance(s_k, float) and s_k == pytest.approx(s_n, rel=1e-9)


def test_density_kernel_matches_numpy(monkeypatch):
    from talos.atmosphere import Atmosphere, AtmosphereConfig
    from talos.constants import YEAR

    atm = Atmosphere(AtmosphereConfig())
    rng = np.random.default_rng(9)
    h = rng.uniform(50e3, 2500e3, (40, 50))  # below the table, inside, above (extrapolated)
    t = rng.uniform(0, 11 * YEAR, 50)
    got = atm.density(h, t)
    s = atm.density(500e3, 0.0)
    monkeypatch.setattr(_kernels, "HAVE_NUMBA", False)
    np.testing.assert_allclose(got, atm.density(h, t), rtol=1e-12)
    assert np.shape(s) == () and s == pytest.approx(atm.density(500e3, 0.0), rel=1e-12)


@pytest.mark.parametrize("cm_model", ["plasma", "constant"])
def test_candidate_kernel_matches_numpy(cm_model):
    from dataclasses import replace

    from talos.engagement import AblationParams
    from talos.scheduler import evaluate_candidates

    pop = toy_population(20000, np.random.default_rng(21), areal_density=0.5)
    pop.areal_density[:] = np.random.default_rng(22).uniform(0.2, 5.0, len(pop))
    sat = baseline_platform(800.0)
    t = np.arange(0.0, 600.0)
    wins = screen_windows(sat, pop, t)
    params = replace(AblationParams(), cm_model=cm_model)
    args = _kernels.element_args(pop.elements, pop.ballistic_coeff, pop.epoch_s)
    n = 0
    for k in sorted({w.start for w in wins})[:40]:
        open_w = [w for w in wins if w.start <= k < w.stop]
        ref = evaluate_candidates(sat, pop, open_w, t, k, params)
        got = evaluate_candidates(sat, pop, open_w, t, k, params, elem_args=args)
        assert [c.debris_index for c in got] == [c.debris_index for c in ref]
        for a, b in zip(got, ref):
            assert a.n_fired == b.n_fired and a.t_stop == b.t_stop
            assert a.dv_retrograde == pytest.approx(b.dv_retrograde, rel=1e-6)
        n += len(ref)
    assert n > 20
