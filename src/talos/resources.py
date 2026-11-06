"""Module 8 — battery SoC + heat-buffer state machine gating bursts.

Duty cycle is an output of this model, not an input.

Model (all placeholders until research item 5 comes back):
  * Array is always sunlit (dawn-dusk SSO; eclipse season ignored) and charges
    the battery at P_charge = array_area * specific_power - bus_power.
  * Firing draws P_elec = P_optical / wallplug_eff; the non-optical part,
    P_heat = P_elec - P_optical, goes into a heat buffer.
  * The radiator rejects P_rad = area * net_flux continuously (buffer floors at 0).
  * A burst of duration f is allowed if the heat buffer and battery both stay in bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .platform import LaserConfig, ResourceConfig

_EPS = 1e-9


@dataclass
class ResourceState:
    cfg: ResourceConfig
    laser: LaserConfig
    t: float = 0.0
    heat_j: float = 0.0
    soc_j: float | None = None  # None -> full

    def __post_init__(self) -> None:
        if self.soc_j is None:
            self.soc_j = self.cfg.battery_capacity_j

    # --- rates [W] ---
    @property
    def p_elec(self) -> float:
        return self.laser.burst_power_w / self.cfg.wallplug_eff

    @property
    def p_heat(self) -> float:
        return self.p_elec - self.laser.burst_power_w

    @property
    def p_rad(self) -> float:
        return self.cfg.radiator_area_m2 * self.cfg.radiator_net_flux_w_m2

    @property
    def p_charge(self) -> float:
        return self.cfg.array_area_m2 * self.cfg.array_specific_power_w_m2 - self.cfg.bus_power_w

    # --- dynamics ---
    def advance(self, t_new: float) -> None:
        """Idle (not firing) until ``t_new``."""
        dt = t_new - self.t
        if dt < -_EPS:
            raise ValueError(f"cannot advance backwards ({self.t} -> {t_new})")
        dt = max(dt, 0.0)
        self.heat_j = max(0.0, self.heat_j - self.p_rad * dt)
        self.soc_j = min(self.cfg.battery_capacity_j, self.soc_j + self.p_charge * dt)
        self.t = t_new

    def max_fire_s(self) -> float:
        """Longest burst that can start now."""
        dh = self.p_heat - self.p_rad
        de = self.p_elec - self.p_charge
        f_heat = (self.cfg.heat_buffer_j - self.heat_j) / dh if dh > 0 else np.inf
        f_soc = self.soc_j / de if de > 0 else np.inf
        return max(0.0, min(f_heat, f_soc))

    def fire(self, f: float) -> None:
        if f > self.max_fire_s() + 1e-6:
            raise ValueError(f"burst {f:.3f}s exceeds resource limit {self.max_fire_s():.3f}s")
        self.heat_j = max(0.0, self.heat_j + (self.p_heat - self.p_rad) * f)
        self.soc_j = min(self.cfg.battery_capacity_j, self.soc_j - (self.p_elec - self.p_charge) * f)
        self.t += f

    def wait_for(self, f: float) -> float:
        """Idle seconds needed before a burst of ``f`` seconds fits (inf if never)."""
        dh = self.p_heat - self.p_rad
        de = self.p_elec - self.p_charge
        waits = [0.0]
        if dh > 0:
            target = self.cfg.heat_buffer_j - f * dh
            if target < 0:
                return np.inf
            if self.heat_j > target:
                waits.append(np.inf if self.p_rad <= 0 else (self.heat_j - target) / self.p_rad)
        if de > 0:
            need = f * de
            if need > self.cfg.battery_capacity_j:
                return np.inf
            if self.soc_j < need:
                waits.append(np.inf if self.p_charge <= 0 else (need - self.soc_j) / self.p_charge)
        return max(waits)

    def steady_state_max_duty(self) -> float:
        """Long-run duty-cycle ceiling from heat rejection and power balance."""
        lims = [1.0]
        if self.p_heat > 0:
            lims.append(self.p_rad / self.p_heat)
        if self.p_elec > 0 and self.p_charge < self.p_elec:
            lims.append(self.p_charge / self.p_elec)
        return float(min(lims))
