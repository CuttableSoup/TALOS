"""Keplerian element container and element <-> Cartesian conversions.

All functions are vectorized: element arrays of any (broadcastable) shape map to
position/velocity arrays with a trailing axis of length 3.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from .constants import MU_EARTH

TWO_PI = 2.0 * np.pi


@dataclass
class KeplerElements:
    """Mean Keplerian elements, structure-of-arrays.

    a [m], e [-], i, raan, argp, M [rad]. Scalars are allowed (shape ``()``),
    which is how a single platform orbit is represented.
    """

    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    raan: np.ndarray
    argp: np.ndarray
    M: np.ndarray

    def __post_init__(self) -> None:
        names = [f.name for f in fields(self)]
        arrs = np.broadcast_arrays(*(np.asarray(getattr(self, n), dtype=float) for n in names))
        for n, x in zip(names, arrs):
            setattr(self, n, np.array(x))  # copy: broadcast views are read-only

    @property
    def shape(self) -> tuple[int, ...]:
        return self.a.shape

    def __len__(self) -> int:
        if self.a.ndim == 0:
            raise TypeError("scalar KeplerElements has no len()")
        return self.a.shape[0]

    def __getitem__(self, idx) -> KeplerElements:
        return KeplerElements(*(getattr(self, f.name)[idx] for f in fields(self)))

    def replace(self, **changes) -> KeplerElements:
        return replace(self, **changes)

    @property
    def rp(self) -> np.ndarray:
        """Perigee radius [m]."""
        return self.a * (1.0 - self.e)

    @property
    def ra(self) -> np.ndarray:
        """Apogee radius [m]."""
        return self.a * (1.0 + self.e)

    @property
    def n(self) -> np.ndarray:
        """Unperturbed mean motion [rad/s]."""
        return np.sqrt(MU_EARTH / self.a**3)

    @property
    def p(self) -> np.ndarray:
        """Semi-latus rectum [m]."""
        return self.a * (1.0 - self.e**2)


def solve_kepler(M, e, tol: float = 1e-12, max_iter: int = 50) -> np.ndarray:
    """Solve Kepler's equation E - e sin E = M for E (elliptic orbits)."""
    M = np.mod(np.asarray(M, dtype=float), TWO_PI)
    e = np.asarray(e, dtype=float)
    E = np.where(e < 0.8, M, np.pi) + 0.0 * e
    for _ in range(max_iter):
        dE = -(E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def elements_to_rv(el: KeplerElements) -> tuple[np.ndarray, np.ndarray]:
    """Convert elements to ECI position [m] and velocity [m/s], shape ``el.shape + (3,)``."""
    E = solve_kepler(el.M, el.e)
    cE, sE = np.cos(E), np.sin(E)
    sq = np.sqrt(1.0 - el.e**2)
    r = el.a * (1.0 - el.e * cE)

    x = el.a * (cE - el.e)
    y = el.a * sq * sE
    k = np.sqrt(MU_EARTH * el.a) / r
    vx = -k * sE
    vy = k * sq * cE

    cO, sO = np.cos(el.raan), np.sin(el.raan)
    cw, sw = np.cos(el.argp), np.sin(el.argp)
    ci, si = np.cos(el.i), np.sin(el.i)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)

    r_vec = x[..., None] * P + y[..., None] * Q
    v_vec = vx[..., None] * P + vy[..., None] * Q
    return r_vec, v_vec


def rv_to_elements(r_vec, v_vec, ecc_tol: float = 1e-10) -> KeplerElements:
    """Convert ECI state(s) to osculating Keplerian elements.

    Near-circular orbits (e < ``ecc_tol``) get argp = 0 and M = argument of latitude.
    Equatorial orbits use the x-axis as the node line.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    r = np.linalg.norm(r_vec, axis=-1)
    v2 = np.sum(v_vec * v_vec, axis=-1)
    rv = np.sum(r_vec * v_vec, axis=-1)

    h = np.cross(r_vec, v_vec)
    hn = np.linalg.norm(h, axis=-1)
    h_hat = h / hn[..., None]

    node = np.stack([-h[..., 1], h[..., 0], np.zeros_like(hn)], axis=-1)
    nn = np.linalg.norm(node, axis=-1)
    equatorial = nn < 1e-12 * hn
    x_axis = np.broadcast_to(np.array([1.0, 0.0, 0.0]), node.shape)
    n_hat = np.where(equatorial[..., None], x_axis, node / np.where(equatorial, 1.0, nn)[..., None])
    m_hat = np.cross(h_hat, n_hat)  # in-plane, 90 deg ahead of node

    e_vec = ((v2 - MU_EARTH / r)[..., None] * r_vec - rv[..., None] * v_vec) / MU_EARTH
    e = np.linalg.norm(e_vec, axis=-1)
    a = 1.0 / (2.0 / r - v2 / MU_EARTH)
    inc = np.arccos(np.clip(h[..., 2] / hn, -1.0, 1.0))
    raan = np.where(equatorial, 0.0, np.mod(np.arctan2(h[..., 0], -h[..., 1]), TWO_PI))

    u = np.arctan2(np.sum(r_vec * m_hat, -1), np.sum(r_vec * n_hat, -1))
    circular = e < ecc_tol
    argp = np.where(circular, 0.0, np.arctan2(np.sum(e_vec * m_hat, -1), np.sum(e_vec * n_hat, -1)))
    e = np.where(circular, 0.0, e)
    nu = u - argp

    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2), np.sqrt(1.0 + e) * np.cos(nu / 2))
    M = np.mod(E - e * np.sin(E), TWO_PI)
    return KeplerElements(a=a, e=e, i=inc, raan=raan, argp=np.mod(argp, TWO_PI), M=M)
