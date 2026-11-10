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
* ``pointing_bias_rad``: per-axis constant pointing offset over a burst
  (centroiding / point-ahead error), added to the per-pulse jitter.

All zero (default) = perfect knowledge, identical to the old behaviour.
Not modelled: catalogue maintenance and re-identification across passes (only
the commitment planner would need it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KnowledgeConfig:
    od_sigma_pos_m: float = 0.0
    od_sigma_vel_m_s: float = 0.0
    pointing_bias_rad: float = 0.0

    @property
    def active(self) -> bool:
        return self.od_sigma_pos_m > 0 or self.od_sigma_vel_m_s > 0 or self.pointing_bias_rad > 0


class Knowledge:
    def __init__(self, cfg: KnowledgeConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
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
