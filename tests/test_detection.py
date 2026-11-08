from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from talos.constants import R_EARTH
from talos.platform import ATPConfig, baseline_platform
from talos.population import toy_population
from talos.screening import apparent_magnitude, screen_windows, track_mask
from talos.sun import SunConfig, SunEnvironment


def test_magnitude_of_reference_sphere():
    """5 cm sphere, albedo 0.1, 500 km, phase 90 deg: ~13.9 mag (hand calc)."""
    r_sat = np.array([[[R_EARTH + 800e3, 0.0, 0.0]]])
    r_deb = r_sat + np.array([[[0.0, 500e3, 0.0]]])
    sun = np.array([[[1.0, 0.0, 0.0]]])  # perpendicular to the LOS: phase 90 deg
    m = apparent_magnitude(r_sat, r_deb, sun, np.array([0.025]), np.array([0.1]))
    flux = 2 / (3 * np.pi) * 0.1 * 0.025 ** 2 / 500e3 ** 2 * 1.0
    assert m[0, 0] == pytest.approx(-26.74 - 2.5 * np.log10(flux))
    assert 13.5 < m[0, 0] < 14.5
    # inverse-square: twice the range is 1.505 mag fainter
    m2 = apparent_magnitude(r_sat, r_sat + np.array([[[0.0, 1000e3, 0.0]]]), sun, np.array([0.025]), np.array([0.1]))
    assert m2[0, 0] - m[0, 0] == pytest.approx(5 * np.log10(2))


def test_sunlit_earth_background_penalty():
    r_sat = np.array([[[R_EARTH + 800e3, 0.0, 0.0]]])
    r_deb = r_sat + np.array([[[-200e3, 0.0, 0.0]]])  # looking straight down
    lit = apparent_magnitude(r_sat, r_deb, np.array([[[1.0, 0, 0]]]), np.array([0.025]), np.array([0.1]), 3.0)
    raw = apparent_magnitude(r_sat, r_deb, np.array([[[1.0, 0, 0]]]), np.array([0.025]), np.array([0.1]))
    assert lit[0, 0] == pytest.approx(raw[0, 0] + 3.0)
    # sun along the orbit-normal: Earth point behind the target is on the terminator/night side
    night = np.array([[[0.0, 0.0, -1.0]]])
    bg = apparent_magnitude(r_sat, r_deb, night, np.array([0.025]), np.array([0.1]), 3.0)
    nobg = apparent_magnitude(r_sat, r_deb, night, np.array([0.025]), np.array([0.1]))
    assert bg[0, 0] == pytest.approx(nobg[0, 0])


def test_track_needs_acquisition_and_drops_when_faint():
    atp = ATPConfig(acquisition_s=3.0, detect_mag_limit=16.0, track_mag_limit=18.0)
    vis = np.ones((10, 1), bool)
    mag = np.array([20, 15, 15, 15, 17, 17, 19, 17, 15, 15], float)[:, None]
    tr = track_mask(vis, mag, atp, 1.0)[:, 0]
    # detected from k=1, tracked after 3 steps (k=3); held at 17 (k=4,5); lost at 19; 17 can't re-acquire
    assert tr.tolist() == [False, False, False, True, True, True, False, False, False, False]
    tr0 = track_mask(vis, np.full((10, 1), 15.0), atp, 1.0)[:, 0]
    assert tr0.all()  # detectable at chunk start: already tracked


def setup(detection_model="passive_optical", **atp_kw):
    env = SunEnvironment(SunConfig(), datetime.fromisoformat("2026-12-21T00:00:00+00:00"))
    sat = baseline_platform(800.0, raan_rad=env.raan_for_ltan(6.0))
    sat = replace(sat, atp=replace(sat.atp, detection_model=detection_model, **atp_kw))
    pop = toy_population(20000, np.random.default_rng(3))
    t = np.arange(0.0, 3600.0)
    return sat, pop, t, env.sun_hat(t)


def windows(sat, pop, t, sun, kernel):
    return [(w.debris_index, w.start, w.stop) for w in screen_windows(sat, pop, t, sun_hat=sun, use_kernel=kernel)]


def test_kernel_matches_numpy_with_detection():
    sat, pop, t, sun = setup()
    assert windows(sat, pop, t, sun, True) == windows(sat, pop, t, sun, False)


def test_detection_reduces_windows_and_favours_large_objects():
    sat, pop, t, sun = setup()
    ideal = windows(replace(sat, atp=replace(sat.atp, detection_model="ideal")), pop, t, sun, True)
    det = windows(sat, pop, t, sun, True)
    steps = lambda ws: sum(b - a for _, a, b in ws)
    assert 0 < steps(det) < steps(ideal)
    d_ideal = np.median(pop.diameter_m[[j for j, _, _ in ideal]])
    d_det = np.median(pop.diameter_m[[j for j, _, _ in det]])
    assert d_det > d_ideal
    # a hopeless sensor sees nothing; a perfect one matches ideal (up to the acquisition delay)
    blind = windows(replace(sat, atp=replace(sat.atp, detect_mag_limit=0.0, track_mag_limit=0.0)), pop, t, sun, True)
    assert blind == []
