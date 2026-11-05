"""Module 7 — orbit update and lifetime.

TODO: apply burst dv -> rv_to_elements; semi-analytic decay with a pymsis
density table. Removed = lifetime < 1 month or perigee < ~200 km.
"""

REMOVAL_PERIGEE_ALT_M = 200e3
REMOVAL_LIFETIME_S = 30 * 86400.0


def apply_dv(*args, **kwargs):
    raise NotImplementedError("module 7")


def orbital_lifetime(*args, **kwargs):
    raise NotImplementedError("module 7")
