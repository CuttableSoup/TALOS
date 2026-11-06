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
