"""Module 8 — battery SoC + lumped thermal-node state machine gating bursts.

Duty cycle is an output of this model, not an input.

Thermal model (one node = PCM store + coolant loop, see ResourceConfig):
  * State is the enthalpy ``heat_j`` above the heater floor ``t_min_k``.
    Temperature from enthalpy: sensible (heat_capacity_j_k) up to the melt
    point, a latent plateau of ``pcm_latent_j`` at ``pcm_melt_k``, sensible above.
  * Heat in: laser waste P_elec - P_optical and battery discharge losses while
    firing; bus power always.
  * Heat out: radiator at T_store - radiator_dT_k, with
    q_net(T) = fin * sides * [eps sigma T^4 - eps F_E IR_E - other]  (W/m^2).
  * Heaters hold T >= t_min_k (heater power not charged to the battery).
  * A burst is allowed if the node stays <= t_max_k and the battery stays >= 0.

Power: array always sunlit (dawn-dusk SSO; the Nov-Feb eclipse season is
ignored), charging at array - bus; the battery supplies P_elec - P_charge while
firing. Battery cycle life is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .platform import LaserConfig, ResourceConfig

_EPS = 1e-9
SIGMA = 5.670374419e-8
_H_STEP = 5.0  # s, RK2 step in the sensible regions


@dataclass
class ResourceState:
    cfg: ResourceConfig
    laser: LaserConfig
    t: float = 0.0
    heat_j: float = 0.0  # enthalpy above the heater floor
    soc_j: float | None = None  # None -> full
    _allow: dict = field(default_factory=dict, repr=False)  # f -> max starting enthalpy (shared by copies)

    def __post_init__(self) -> None:
        if self.soc_j is None:
            self.soc_j = self.cfg.battery_capacity_j
        c = self.cfg
        C = c.heat_capacity_j_k
        self._H1 = C * (c.pcm_melt_k - c.t_min_k)  # melt starts
        self._H2 = self._H1 + c.pcm_latent_j  # fully melted
        self.h_max = self._H2 + C * (c.t_max_k - c.pcm_melt_k)

    # --- rates [W] ---
    @property
    def p_elec(self) -> float:
        return self.laser.burst_power_w / self.cfg.wallplug_eff

    @property
    def p_heat(self) -> float:
        """Laser waste heat while firing."""
        return self.p_elec - self.laser.burst_power_w

    @property
    def p_charge(self) -> float:
        return self.cfg.array_area_m2 * self.cfg.array_specific_power_w_m2 - self.cfg.bus_power_w

    @property
    def p_batt_loss(self) -> float:
        return (1.0 - self.cfg.battery_rt_eff) * max(self.p_elec - self.p_charge, 0.0)

    @property
    def p_fire_in(self) -> float:
        """Total heat into the node while firing."""
        return self.p_heat + self.p_batt_loss + self.cfg.bus_power_w

    @property
    def p_idle_in(self) -> float:
        return self.cfg.bus_power_w

    def q_net(self, t_rad_k: float) -> float:
        """Net radiator rejection per m^2 of panel (planform) at radiator temperature T."""
        c = self.cfg
        per_side = (c.radiator_emissivity * SIGMA * t_rad_k ** 4
                    - c.radiator_emissivity * c.radiator_earth_view * c.earth_ir_w_m2
                    - c.radiator_other_load_w_m2)
        return c.fin_efficiency * c.radiator_sides * per_side

    def p_rad_at(self, temp_k: float) -> float:
        """Radiator rejection [W] with the store at ``temp_k``."""
        return max(0.0, self.cfg.radiator_area_m2 * self.q_net(temp_k - self.cfg.radiator_dT_k))

    @property
    def p_rad(self) -> float:
        return self.p_rad_at(self.temperature_k)

    # --- enthalpy <-> temperature ---
    def temp_of(self, H: float) -> float:
        c = self.cfg
        if H <= self._H1:
            return c.t_min_k + H / c.heat_capacity_j_k
        if H <= self._H2:
            return c.pcm_melt_k
        return c.pcm_melt_k + (H - self._H2) / c.heat_capacity_j_k

    @property
    def temperature_k(self) -> float:
        return self.temp_of(self.heat_j)

    @property
    def melt_fraction(self) -> float:
        return float(np.clip((self.heat_j - self._H1) / max(self.cfg.pcm_latent_j, _EPS), 0.0, 1.0))

    def _evolve(self, H: float, p_in: float, dt: float, goal: float | None = None) -> tuple[float, float]:
        """Integrate dH/dt = p_in - P_rad(T(H)), floored at 0, for ``dt`` s or until H
        reaches ``goal``. Returns (H, elapsed). Exact on the latent plateau, RK2 elsewhere."""
        t = 0.0
        if goal is not None and H == goal:
            return H, 0.0
        while t < dt - 1e-12:
            rem = dt - t
            r = p_in - self.p_rad_at(self.temp_of(H))
            if abs(r) < 1e-9 or (H <= 0.0 and r < 0.0):
                return max(H, 0.0), dt  # equilibrium, or held at the heater floor
            if self._H2 > self._H1 and ((self._H1 < H < self._H2) or (H == self._H1 and r > 0)
                                        or (H == self._H2 and r < 0)):
                edge = self._H2 if r > 0 else self._H1
                h = min(rem, (edge - H) / r)
                Hn = edge if h < rem else H + r * h
            else:
                h = min(rem, _H_STEP)
                Hm = max(0.0, H + 0.5 * h * r)
                Hn = H + h * (p_in - self.p_rad_at(self.temp_of(Hm)))
                for edge in (self._H1, self._H2):  # stop on a plateau edge
                    if (H - edge) * (Hn - edge) < 0:
                        h *= (edge - H) / (Hn - H)
                        Hn = edge
                        break
                if Hn < 0.0:
                    h *= H / (H - Hn)
                    Hn = 0.0
            if goal is not None and (H - goal) * (Hn - goal) <= 0 and Hn != H:
                return goal, t + h * (goal - H) / (Hn - H)
            H, t = Hn, t + h
        return H, t

    # --- dynamics ---
    def advance(self, t_new: float) -> None:
        """Idle (not firing) until ``t_new``."""
        dt = t_new - self.t
        if dt < -_EPS:
            raise ValueError(f"cannot advance backwards ({self.t} -> {t_new})")
        if dt > 0:
            self.heat_j, _ = self._evolve(self.heat_j, self.p_idle_in, dt)
            self.soc_j = min(self.cfg.battery_capacity_j, self.soc_j + self.p_charge * dt)
        self.t = max(self.t, t_new)

    def _soc_drain(self) -> float:
        return self.p_elec - self.p_charge  # > 0: battery drains while firing

    def max_fire_s(self, cap: float = 3600.0) -> float:
        """Longest burst that can start now."""
        if self.heat_j >= self.h_max:
            f_heat = 0.0
        else:
            H, t_hit = self._evolve(self.heat_j, self.p_fire_in, cap, goal=self.h_max)
            f_heat = t_hit if H >= self.h_max - 1e-6 else np.inf
        de = self._soc_drain()
        f_soc = self.soc_j / de if de > 0 else np.inf
        return max(0.0, min(f_heat, f_soc))

    def fire(self, f: float) -> None:
        if f > self.max_fire_s() + 1e-6:
            raise ValueError(f"burst {f:.3f}s exceeds resource limit {self.max_fire_s():.3f}s")
        H, _ = self._evolve(self.heat_j, self.p_fire_in, f)
        self.heat_j = min(H, self.h_max)
        self.soc_j = min(self.cfg.battery_capacity_j, self.soc_j - self._soc_drain() * f)
        self.t += f

    def _allowed_start(self, f: float) -> float:
        """Largest starting enthalpy from which an f-second burst stays <= h_max (-1: never)."""
        key = round(f, 9)
        if key not in self._allow:
            end = lambda H0: self._evolve(H0, self.p_fire_in, f)[0]
            if end(0.0) > self.h_max + 1e-6:
                val = -1.0
            else:
                lo, hi = 0.0, self.h_max
                while hi - lo > 1.0:
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if end(mid) <= self.h_max else (lo, mid)
                val = lo
            self._allow[key] = val
        return self._allow[key]

    def wait_for(self, f: float, cap: float = 6 * 3600.0) -> float:
        """Idle seconds needed before a burst of ``f`` seconds fits (inf if never)."""
        H_ok = self._allowed_start(f)
        if H_ok < 0:
            return np.inf
        waits = [0.0]
        if self.heat_j > H_ok:
            H, t_hit = self._evolve(self.heat_j, self.p_idle_in, cap, goal=H_ok)
            waits.append(t_hit if H <= H_ok + 1e-6 else np.inf)
        de = self._soc_drain()
        if de > 0:
            need = f * de
            if need > self.cfg.battery_capacity_j:
                return np.inf
            if self.soc_j < need:
                waits.append(np.inf if self.p_charge <= 0 else (need - self.soc_j) / self.p_charge)
        return max(waits)

    def steady_state_max_duty(self) -> float:
        """Long-run duty ceiling: heat balance with the store on its melt plateau
        (fire: p_fire_in - rad, idle: rad - p_idle_in), and orbit-average power."""
        lims = [1.0]
        rad = self.p_rad_at(self.cfg.pcm_melt_k)
        if self.p_fire_in > rad:
            lims.append(max(0.0, rad - self.p_idle_in) / (self.p_fire_in - self.p_idle_in))
        if self.p_elec > 0 and self.p_charge < self.p_elec:
            lims.append(self.p_charge / self.p_elec)
        return float(min(lims))
