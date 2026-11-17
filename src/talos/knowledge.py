"""Orbit knowledge (missing-elements item 1): what the platform knows about a target.

The scheduler used to plan with every object's true state. Onboard, a target's
state comes from the platform's own track: after acquisition (10 s of
detection) the 1.5 m optics give angles to ~0.3 urad and, with laser ranging,
range to ~cm, so a 10 s arc fixes velocity to ~cm/s and the semi-major axis to
tens of metres; each pass re-estimates the orbit, so the greedy scheduler needs
no catalogue. This model lets that knowledge be degraded:

* ``od_sigma_pos_m`` / ``od_sigma_vel_m_s``: per-axis error of the scheduler's
  state estimate at the burst start (e.g. angles-only tracking without laser
  ranging). One error per object per chunk (one track); burst predictions and
  the dv still needed use the estimate; the burst itself uses the truth.
* ``pointing_bias_rad``: per-axis constant pointing offset over a burst, of no stated
  cause, added to the per-pulse jitter (the pre-budget knob; kept).

Pointing error budget (2026-11-17; each term per axis, 1 sigma, all 0 by default). With
the platform's ATP jitter (``atp.jitter_rad``, per pulse) these are the terms a flight
allocation would carry:

* ``boresight_bias_rad``: residual tracker-to-laser boresight misalignment after
  calibration; fixed per platform for the whole run.
* ``centroid_rad``: track centroiding error for a target of ``centroid_ref_mag``,
  constant over a burst (the track filter averages the photon noise within it); scales
  with photon noise, 10^(0.2 (m - centroid_ref_mag)), at the target's V magnitude at
  mid-burst (needs [sun]; without it the reference value is used).
* ``drift_rad_s``: thermal / structural line-of-sight drift during a burst: a random
  rate per burst, offset = rate x time since the burst start.
* ``point_ahead_frac``: fractional error of the point-ahead angle 2 v_perp / c (light
  time out and back; ~50-100 urad here), along the direction of apparent motion.

The mean fraction of the on-axis fluence delivered under a Gaussian error of total
per-axis sigma s is 1 / (1 + 4 s^2 / w^2) (``talos.engagement.delivered_fraction``), so
the Gaussian terms add in quadrature for the mean; constant-per-burst terms hurt more
than that, because whole bursts fall below the ablation threshold.

All zero (default) = perfect knowledge, identical to the old behaviour.
Not modelled: catalogue maintenance and re-identification across passes (only
the commitment planner would need it).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class KnowledgeConfig:
    od_sigma_pos_m: float = 0.0
    od_sigma_vel_m_s: float = 0.0
    pointing_bias_rad: float = 0.0
    # pointing error budget (see module docstring); PICKS, 0 = off
    boresight_bias_rad: float = 0.0
    centroid_rad: float = 0.0
    centroid_ref_mag: float = 14.0
    drift_rad_s: float = 0.0
    point_ahead_frac: float = 0.0

    @property
    def pointing_active(self) -> bool:
        return (self.pointing_bias_rad > 0 or self.boresight_bias_rad > 0 or self.centroid_rad > 0
                or self.drift_rad_s > 0 or self.point_ahead_frac > 0)

    @property
    def active(self) -> bool:
        return self.od_sigma_pos_m > 0 or self.od_sigma_vel_m_s > 0 or self.pointing_active


class Knowledge:
    def __init__(self, cfg: KnowledgeConfig, rng: np.random.Generator, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = rng
        self.seed = int(seed)
        self._chunk = None
        self._err: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def new_chunk(self, c0: float) -> None:
        if c0 != self._chunk:
            self._chunk, self._err = c0, {}

    @property
    def has_state_error(self) -> bool:
        return self.cfg.od_sigma_pos_m > 0 or self.cfg.od_sigma_vel_m_s > 0

    def state_error(self, js) -> tuple[np.ndarray, np.ndarray]:
        """(dr, dv) per object in ``js``, fixed per object for the current chunk."""
        c = self.cfg
        for j in map(int, js):
            if j not in self._err:
                self._err[j] = (c.od_sigma_pos_m * self.rng.standard_normal(3),
                                c.od_sigma_vel_m_s * self.rng.standard_normal(3))
        dr = np.array([self._err[int(j)][0] for j in js]).reshape(-1, 3)
        dv = np.array([self._err[int(j)][1] for j in js]).reshape(-1, 3)
        return dr, dv

    def pointing_bias(self) -> np.ndarray | None:
        b = self.cfg.pointing_bias_rad
        return b * self.rng.standard_normal(2) if b > 0 else None

    def boresight(self, platform: str) -> np.ndarray:
        """(2,) fixed boresight offset of ``platform``: its own generator (run seed and the
        platform name), so it does not move the other draws and survives a resume."""
        g = np.random.default_rng([self.seed + 6, zlib.crc32(platform.encode())])
        return self.cfg.boresight_bias_rad * g.standard_normal(2)

    def pointing_offsets(self, platform: str, tp: np.ndarray, r_sat: np.ndarray, r_deb: np.ndarray,
                         mag: float | None = None) -> np.ndarray | None:
        """(P, 2) pointing offset [rad] of every pulse of a burst at times ``tp`` with states
        (P, 3), in a frame whose first axis is the target's apparent motion; None if no term
        is on. Per-burst draws come from the knowledge generator in a fixed order (the
        pre-budget ``pointing_bias_rad`` first, so runs using only it are unchanged)."""
        c = self.cfg
        if not c.pointing_active:
            return None
        P = len(tp)
        off = np.zeros((P, 2))
        b = self.pointing_bias()
        if b is not None:
            off += b
        if c.boresight_bias_rad > 0:
            off += self.boresight(platform)
        if c.centroid_rad > 0:
            scale = 10.0 ** (0.2 * (mag - c.centroid_ref_mag)) if mag is not None and np.isfinite(mag) else 1.0
            off += c.centroid_rad * scale * self.rng.standard_normal(2)
        if c.drift_rad_s > 0:
            off += (tp - tp[0])[:, None] * (c.drift_rad_s * self.rng.standard_normal(2))[None, :]
        if c.point_ahead_frac > 0 and P > 1:
            d = r_deb - r_sat
            R = np.linalg.norm(d, axis=-1)
            u = d / R[:, None]
            dudt = np.gradient(u, tp, axis=0)  # apparent angular rate [rad/s]
            off[:, 0] += c.point_ahead_frac * 2.0 * R * np.linalg.norm(dudt, axis=-1) / C_LIGHT
        return off
