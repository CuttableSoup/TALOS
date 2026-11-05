"""Physical constants (SI units throughout the package)."""

import math

MU_EARTH = 3.986004418e14  # m^3/s^2 (EGM-96 / WGS-84)
R_EARTH = 6378137.0  # m, equatorial
J2 = 1.08262668e-3
DAY = 86400.0  # s
YEAR = 365.25 * DAY  # s, Julian year

# RAAN precession rate required for a sun-synchronous orbit (one turn per tropical year).
SSO_RAAN_RATE = 2.0 * math.pi / (365.2421897 * DAY)  # rad/s

FT2_TO_M2 = 0.09290304
