"""Gate 5 spot check vs NASA DAS 3.2.7 (skipped when DAS is not installed)."""

import numpy as np
import pytest

from talos.atmosphere import Atmosphere, AtmosphereConfig
from talos.constants import R_EARTH, YEAR
from talos.das import das_home
from talos.lifetime import orbital_lifetime
from talos.propagator import get_atmosphere, set_atmosphere

pytestmark = pytest.mark.skipif(not (das_home() / "Fortran.dll").exists(), reason="DAS 3.2.7 not installed")


def test_lowered_orbit_lifetime_within_25pct_of_das():
    """Near the removal threshold (perigee 300 km, apogee 800 km) TALOS is ~0.8x DAS (runs/das_lifetime)."""
    from talos.das import lifetime

    d, _, msg = lifetime(2026.97, 300.0, 800.0, 98.0, 0.5)
    assert msg == 2
    prev = get_atmosphere()
    set_atmosphere(Atmosphere(AtmosphereConfig(), 2026.97))
    try:
        a = R_EARTH + 550e3
        t = orbital_lifetime(np.array([a]), np.array([250e3 / a]), np.array([2.2 * 0.5]), max_s=10 * YEAR)[0] / YEAR
    finally:
        set_atmosphere(prev)
    assert 0.75 < t / d < 1.25
