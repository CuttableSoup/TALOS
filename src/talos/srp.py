"""Solar radiation pressure on debris: orbit-averaged eccentricity forcing.

A constant perturbing acceleration F on a near-circular orbit changes the
eccentricity vector at the averaged rate (Gauss equations at e ~ 0, averaged over
the argument of latitude):

    de/dt = 3 / (2 n a) * F x h_hat

F = -P Cr (A/m) (AU / d)^2 s_hat (away from the Sun), P = 4.56e-6 N/m^2. Only the
in-plane part of F contributes; the out-of-plane part moves i / RAAN at O(e) and is
ignored, as is the small change in a from the shadowed arc. The shadow is a
cylinder: the force is scaled by the lit fraction of the circular orbit. Applied
once per population-dynamics step (operator splitting with the drag / J2 step);
J2 rotates the eccentricity vector between steps, which bounds the SRP-driven
oscillation to roughly |de/dt| / |apsidal rate| (tens of km of perigee for A/m ~ 1-2).

Matters for light, high-A/m debris: at 800 km, A/m = 2 m^2/kg, SRP is ~10x drag.
Lifetime estimates (talos.lifetime.orbital_lifetime) stay drag-only.
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .elements import KeplerElements

P_SUN = 4.56e-6  # N/m^2 at 1 AU (1361 W/m^2 / c)
AU = 1.495978707e11


def lit_fraction_circular(a, h_hat, sun_hat) -> np.ndarray:
    """Fraction of a circular orbit of radius ``a`` outside a cylindrical Earth shadow."""
    cosb = np.sqrt(np.clip(1.0 - np.einsum("...k,...k->...", h_hat, sun_hat) ** 2, 0.0, 1.0))
    x = np.sqrt(np.maximum(a**2 - R_EARTH**2, 0.0)) / np.maximum(a * cosb, 1e-12)
    return 1.0 - np.where(x < 1.0, np.arccos(np.clip(x, -1.0, 1.0)) / np.pi, 0.0)


def srp_step(el: KeplerElements, area_to_mass, sun_hat, dt, cr: float = 1.2,
             sun_dist_m: float = AU) -> KeplerElements:
    """Elements after ``dt`` seconds of averaged SRP (sun direction held at ``sun_hat``).
    Keeps a, i, RAAN and the mean argument of latitude (argp + M)."""
    ci, si = np.cos(el.i), np.sin(el.i)
    cO, sO = np.cos(el.raan), np.sin(el.raan)
    node = np.stack([cO, sO, np.zeros_like(cO)], -1)
    perp = np.stack([-sO * ci, cO * ci, si], -1)  # h_hat x node, in plane
    h_hat = np.stack([si * sO, -si * cO, ci], -1)
    s = np.broadcast_to(np.asarray(sun_hat, float), h_hat.shape)
    n = np.sqrt(MU_EARTH / el.a**3)
    k = P_SUN * cr * np.asarray(area_to_mass, float) * (AU / sun_dist_m) ** 2
    k = k * lit_fraction_circular(el.a, h_hat, s)
    F = -k[..., None] * s
    de = (1.5 / (n * el.a))[..., None] * np.cross(F, h_hat) * np.asarray(dt, float)[..., None]
    ex = el.e * np.cos(el.argp) + np.einsum("...k,...k->...", de, node)
    ey = el.e * np.sin(el.argp) + np.einsum("...k,...k->...", de, perp)
    e = np.clip(np.hypot(ex, ey), 0.0, 0.999)
    argp = np.mod(np.arctan2(ey, ex), 2 * np.pi)
    M = np.mod(el.M + el.argp - argp, 2 * np.pi)
    return KeplerElements(a=el.a, e=e, i=el.i, raan=el.raan, argp=argp, M=M)
