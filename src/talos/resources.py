"""Module 8 — battery SoC + lumped thermal-node state machine gating bursts.

Duty cycle is an output of this model, not an input.

Thermal model (one node = PCM store + coolant loop, see ResourceConfig):
  * State is the enthalpy ``heat_j`` above the heater floor ``t_min_k``.
    Temperature from enthalpy: sensible (heat_capacity_j_k) up to the melt
    point, a latent plateau of ``pcm_latent_j`` at ``pcm_melt_k``, sensible above.
  * Heat in: laser waste P_elec - P_optical and battery discharge losses while
    firing; bus power always.
  * Heat out: radiator at T_store - radiator_dT_k, with
    q_net = fin * sides * [eps sigma T^4 - eps F_E IR_E - alpha * albedo - other].
  * At the floor, louvers/bypass throttle the radiator down to
    ``radiator_turndown_min`` of full rejection; any remaining deficit is made up
    by heaters drawing from the battery.
  * A burst is allowed if the node stays <= t_max_k, the battery stays >= 0 and
    (unless ``fire_in_eclipse``) the platform is sunlit for the whole burst.

Power: array output = nominal x illumination timeline factor (sun distance,
pointing, eclipse; see sun.py); without a timeline the platform is always
sunlit at nominal output. The battery supplies P_elec - P_charge while firing
and the bus + heaters in eclipse. Battery cycle life is not modelled.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .platform import LaserConfig, ResourceConfig
from .sun import Timeline

_EPS = 1e-9
SIGMA = 5.670374419e-8
_H_STEP = 5.0  # s, RK2 step in the sensible regions
_PROBE_S = 10.0  # wait_for search step before bisection


@dataclass
class ResourceState:
    cfg: ResourceConfig
    laser: LaserConfig
    t: float = 0.0
    heat_j: float = 0.0  # enthalpy above the heater floor
    soc_j: float | None = None  # None -> full
    env: Timeline | None = None
    fire_in_eclipse: bool = False
    heater_j: float = 0.0  # cumulative heater energy drawn
    brownout_s: float = 0.0  # cumulative time with an empty battery

    def __post_init__(self) -> None:
        if self.soc_j is None:
            self.soc_j = self.cfg.battery_capacity_j
        c = self.cfg
        C = c.heat_capacity_j_k
        self._H1 = C * (c.pcm_melt_k - c.t_min_k)  # melt starts
        self._H2 = self._H1 + c.pcm_latent_j  # fully melted
        self.h_max = self._H2 + C * (c.t_max_k - c.pcm_melt_k)

    # --- nominal rates [W] ---
    @property
    def p_elec(self) -> float:
        return self.laser.burst_power_w / self.cfg.wallplug_eff

    @property
    def p_heat(self) -> float:
        """Laser waste heat while firing."""
        return self.p_elec - self.laser.burst_power_w

    @property
    def p_array_nominal(self) -> float:
        return self.cfg.array_area_m2 * self.cfg.array_specific_power_w_m2

    @property
    def p_charge(self) -> float:
        """Nominal (fully lit, 1 AU) array power left after the bus."""
        return self.p_array_nominal - self.cfg.bus_power_w

    def _batt_loss(self, p_charge: float) -> float:
        return (1.0 - self.cfg.battery_rt_eff) * max(self.p_elec - p_charge, 0.0)

    @property
    def p_fire_in(self) -> float:
        """Total heat into the node while firing (nominal illumination)."""
        return self.p_heat + self._batt_loss(self.p_charge) + self.cfg.bus_power_w

    @property
    def p_idle_in(self) -> float:
        return self.cfg.bus_power_w

    def q_net(self, t_rad_k: float, albedo_w_m2: float = 0.0) -> float:
        """Net radiator rejection per m^2 of panel (planform) at radiator temperature T."""
        c = self.cfg
        per_side = (c.radiator_emissivity * SIGMA * t_rad_k ** 4
                    - c.radiator_emissivity * c.radiator_earth_view * c.earth_ir_w_m2
                    - c.radiator_absorptivity * albedo_w_m2
                    - c.radiator_other_load_w_m2)
        return c.fin_efficiency * c.radiator_sides * per_side

    def p_rad_at(self, temp_k: float, albedo_w_m2: float = 0.0) -> float:
        """Radiator rejection [W] with the store at ``temp_k``."""
        return max(0.0, self.cfg.radiator_area_m2 * self.q_net(temp_k - self.cfg.radiator_dT_k, albedo_w_m2))

    @property
    def p_rad(self) -> float:
        return self.p_rad_at(self.temperature_k, self._seg(self.t)[2])

    # --- environment segment at time t: (lit, p_charge, albedo, segment end) ---
    def _seg(self, t: float):
        if self.env is None:
            return True, self.p_charge, 0.0, np.inf
        k = self.env.index(t)
        end = np.inf if k == len(self.env.lit) - 1 else (k + 1) * self.env.step_s
        return (bool(self.env.lit[k]), self.p_array_nominal * float(self.env.array_factor[k]) - self.cfg.bus_power_w,
                float(self.env.albedo_w_m2[k]), end)

    @property
    def sunlit(self) -> bool:
        return self._seg(self.t)[0]

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

    def _evolve(self, H, p_in, dt, albedo=0.0, goal=None):
        """Integrate dH/dt = p_in - P_rad(T(H)) for ``dt`` s (constant environment), or
        until H reaches ``goal``. Below the floor the radiator is throttled.
        Returns (H, elapsed, time spent held at the floor)."""
        t = 0.0
        if goal is not None and H == goal:
            return H, 0.0, 0.0
        while t < dt - 1e-12:
            rem = dt - t
            r = p_in - self.p_rad_at(self.temp_of(H), albedo)
            if abs(r) < 1e-9:
                return H, dt, 0.0
            if H <= 0.0 and r < 0.0:
                return 0.0, dt, rem  # held at the floor (louvers + heaters)
            if self._H2 > self._H1 and ((self._H1 < H < self._H2) or (H == self._H1 and r > 0)
                                        or (H == self._H2 and r < 0)):
                edge = self._H2 if r > 0 else self._H1  # latent plateau: constant rate
                h = min(rem, (edge - H) / r)
                Hn = edge if h < rem else H + r * h
            else:
                h = min(rem, _H_STEP)
                Hm = max(0.0, H + 0.5 * h * r)
                Hn = H + h * (p_in - self.p_rad_at(self.temp_of(Hm), albedo))
                for edge in (self._H1, self._H2):  # stop on a plateau edge
                    if (H - edge) * (Hn - edge) < 0:
                        h *= (edge - H) / (Hn - H)
                        Hn = edge
                        break
                if Hn < 0.0:
                    h *= H / (H - Hn)
                    Hn = 0.0
            if goal is not None and (H - goal) * (Hn - goal) <= 0 and Hn != H:
                return goal, t + h * (goal - H) / (Hn - H), 0.0
            H, t = Hn, t + h
        return H, t, 0.0

    def _heater_w(self, albedo: float) -> float:
        """Heater power needed at the floor with the radiator fully throttled."""
        floor_rej = self.cfg.radiator_turndown_min * self.p_rad_at(self.cfg.t_min_k, albedo)
        return max(0.0, floor_rej - self.p_idle_in)

    # --- dynamics ---
    def advance(self, t_new: float) -> None:
        """Idle (not firing) until ``t_new``."""
        if t_new - self.t < -_EPS:
            raise ValueError(f"cannot advance backwards ({self.t} -> {t_new})")
        cap = self.cfg.battery_capacity_j
        while self.t < t_new - 1e-12:
            lit, pc, alb, end = self._seg(self.t)
            dt = min(t_new, end) - self.t
            self.heat_j, _, t_floor = self._evolve(self.heat_j, self.p_idle_in, dt, alb)
            heater = self._heater_w(alb) * t_floor
            self.heater_j += heater
            soc = self.soc_j + pc * dt - heater
            if soc < 0.0:
                rate = (heater / t_floor if t_floor > 0 else 0.0) - pc
                self.brownout_s += -soc / rate if rate > 0 else dt
                soc = 0.0
            self.soc_j = min(cap, soc)
            self.t = min(t_new, end) if end < t_new else t_new

    def max_fire_s(self, cap: float = 600.0) -> float:
        """Longest burst that can start now."""
        if self.heat_j >= self.h_max:
            return 0.0
        H, soc, t, f = self.heat_j, self.soc_j, self.t, 0.0
        while f < cap:
            lit, pc, alb, end = self._seg(t)
            if not lit and not self.fire_in_eclipse:
                return f
            dt = min(end - t, cap - f)
            de = self.p_elec - pc
            t_soc = soc / de if de > 0 else np.inf
            p_in = self.p_heat + self._batt_loss(pc) + self.cfg.bus_power_w
            H, el, _ = self._evolve(H, p_in, min(dt, t_soc), alb, goal=self.h_max)
            if H >= self.h_max - 1e-6:
                return f + el
            if t_soc <= dt:
                return f + t_soc
            f += dt
            t += dt
            soc -= de * dt
        return np.inf if self.env is None else f

    def fire(self, f: float) -> None:
        if f > self.max_fire_s() + 1e-6:
            raise ValueError(f"burst {f:.3f}s exceeds resource limit {self.max_fire_s():.3f}s")
        t_end = self.t + f
        while self.t < t_end - 1e-12:
            lit, pc, alb, end = self._seg(self.t)
            dt = min(t_end, end) - self.t
            p_in = self.p_heat + self._batt_loss(pc) + self.cfg.bus_power_w
            H, _, _ = self._evolve(self.heat_j, p_in, dt, alb)
            self.heat_j = min(H, self.h_max)
            self.soc_j = float(np.clip(self.soc_j - (self.p_elec - pc) * dt, 0.0, self.cfg.battery_capacity_j))
            self.t = min(t_end, end) if end < t_end else t_end

    def _never_fits(self, f: float) -> bool:
        fresh = ResourceState(self.cfg, self.laser)
        return fresh.max_fire_s() < f - 1e-9

    def wait_for(self, f: float, cap: float = 6 * 3600.0) -> float:
        """Idle seconds until a burst of ``f`` seconds fits (inf if not within ``cap``)."""
        if self.max_fire_s() >= f - 1e-9:
            return 0.0
        if self._never_fits(f):
            return np.inf
        s = copy.copy(self)
        while s.t - self.t < cap:
            prev = copy.copy(s)
            s.advance(s.t + _PROBE_S)
            if s.max_fire_s() >= f - 1e-9:
                lo, hi = 0.0, _PROBE_S
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    c = copy.copy(prev)
                    c.advance(prev.t + mid)
                    lo, hi = (lo, mid) if c.max_fire_s() >= f - 1e-9 else (mid, hi)
                return prev.t + hi - self.t
        return np.inf

    def steady_state_max_duty(self) -> float:
        """Long-run duty ceiling: heat balance with the store on its melt plateau
        (fire: p_fire_in - rad, idle: rad - p_idle_in), orbit-average power and,
        without eclipse firing, the sunlit fraction."""
        lims = [1.0]
        rad = self.p_rad_at(self.cfg.pcm_melt_k)
        if self.p_fire_in > rad:
            lims.append(max(0.0, rad - self.p_idle_in) / (self.p_fire_in - self.p_idle_in))
        if self.env is not None:
            p_avg = self.p_array_nominal * float(np.mean(self.env.array_factor)) - self.cfg.bus_power_w
            lit = float(np.mean(self.env.lit))
        else:
            p_avg, lit = self.p_charge, 1.0
        if self.p_elec > 0 and p_avg < self.p_elec:
            lims.append(max(p_avg, 0.0) / self.p_elec)
        if not self.fire_in_eclipse:
            lims.append(lit)
        return float(min(lims))
