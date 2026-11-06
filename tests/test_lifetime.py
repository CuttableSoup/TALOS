import numpy as np
import pytest

from talos.constants import DAY, MU_EARTH, R_EARTH, YEAR
from talos.elements import KeplerElements, elements_to_rv
from talos.lifetime import apply_dv, decay_rates, is_removed, move_epoch, orbital_lifetime
from talos.propagator import drag_da_dt


def circ(alt_km, M=0.0):
    return KeplerElements(a=R_EARTH + alt_km * 1e3, e=0.0, i=1.7, raan=0.5, argp=0.0, M=M)


def test_retrograde_dv_lowers_perigee_by_4a_dv_over_v():
    el = circ(800.0)
    r, v = elements_to_rv(el)
    vmag = np.linalg.norm(v)
    dv = 10.0
    new = apply_dv(el, -dv * v / vmag)
    assert new.ra == pytest.approx(el.a, rel=1e-6)  # burn point becomes apogee
    assert el.a - new.rp == pytest.approx(4 * el.a * dv / vmag, rel=2e-2)


def test_perigee_to_200km_needs_about_150_mps():
    el = circ(800.0)
    _, v = elements_to_rv(el)
    vhat = v / np.linalg.norm(v)
    new = apply_dv(el, -155.0 * vhat)
    assert 150e3 < new.rp - R_EARTH < 250e3


def test_radial_dv_keeps_semimajor_axis():
    el = circ(800.0)
    r, _ = elements_to_rv(el)
    new = apply_dv(el, 5.0 * r / np.linalg.norm(r))
    assert new.a == pytest.approx(el.a, rel=1e-6)
    assert new.e > 0


def test_move_epoch_roundtrip():
    el = circ(750.0, M=1.0)
    back = move_epoch(move_epoch(el, 3 * DAY), -3 * DAY)
    r0, _ = elements_to_rv(el)
    r1, _ = elements_to_rv(back)
    assert np.allclose(r0, r1, atol=1e-3)


def test_decay_rates_circular_match_simple_drag():
    a = R_EARTH + np.array([400e3, 700e3])
    da, de = decay_rates(a, 0.0, 2.2)
    assert np.allclose(da, drag_da_dt(a, 2.2), rtol=1e-12)
    assert np.allclose(de, 0.0)


def test_eccentric_orbit_circularizes():
    a = R_EARTH + 900e3
    _, de = decay_rates(a, 0.05, 2.2)
    assert de < 0


def test_circular_lifetime_matches_fine_euler():
    a0, B = R_EARTH + 400e3, 0.05
    a, t, dt = a0, 0.0, 600.0
    while a - R_EARTH >= 120e3:
        a += drag_da_dt(a, B) * dt
        t += dt
    assert orbital_lifetime(a0, 0.0, B) == pytest.approx(t, rel=0.02)


def test_lifetime_ordering():
    alts = R_EARTH + np.array([500e3, 700e3, 900e3])
    life = orbital_lifetime(alts, 0.0, 2.2)
    assert np.all(np.diff(life) > 0)
    # Same perigee, eccentric lives longer than circular
    rp = R_EARTH + 500e3
    ecc = orbital_lifetime(rp / (1 - 0.02), 0.02, 2.2)
    assert ecc > life[0]
    assert orbital_lifetime(alts[0], 0.0, 22.0) < life[0]


def test_lifetime_cap_returns_inf():
    assert np.isinf(orbital_lifetime(R_EARTH + 1000e3, 0.0, 0.01, max_s=YEAR))


def test_is_removed():
    a = R_EARTH + 800e3
    el = KeplerElements(
        a=[a, a, (2 * R_EARTH + 150e3 + 800e3) / 2],
        e=[0.0, 0.0, 0.0],
        i=1.7, raan=0, argp=0, M=0,
    )
    el.e[2] = 1 - (R_EARTH + 150e3) / el.a[2]  # perigee 150 km
    el.a[1] = R_EARTH + 180e3  # circular 180 km
    assert is_removed(el, 2.2).tolist() == [False, True, True]
