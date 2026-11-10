"""Mass, mission life and cost per removal (roadmap items 4-5).

Per platform:

* mass = radiator + solar array + energy storage + PCM + laser + beam director,
  plus a bus fraction of that payload;
* mission life = min(design life, storage end of life at the run's cycling,
  laser shot life at the run's firing rate);
* cost = mass x (hardware + launch $/kg) + operations x life.

Headline: cost per debris-year = constellation cost / (debris-years removed per
year x life), debris-years being the summed reduction in orbital lifetime. Cost per
(credited) removal is kept, but it misleads across mission phases: late in a
mission most removals finish objects already lowered (little lifetime left, not
credited) while heavy objects carry long lifetimes, so per removal the late phase
looked ~85x dearer than the early one and per debris-year ~2x (README, "Hardware
against the depleted sky"). Rates are the run's own annualised rates: a 30-day
run sees only the light-target phase, so compare designs over both phases
(early sky and a depleted checkpoint, run_baseline.py --fresh-platforms).
Every coefficient is a PICK pending sourced values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostConfig:
    radiator_kg_m2: float = 10.0  # deployable radiator incl. loop (research pass: ~10 kg/m^2)
    array_w_kg: float = 100.0  # solar array specific power, EOL, incl. structure (PICK)
    laser_kg_per_kw: float = 20.0  # laser head + pump + loop per kW optical (PICK)
    beam_director_kg: float = 300.0  # 1.5 m aperture beam director + ATP (PICK)
    bus_fraction: float = 0.3  # bus (structure, avionics, AOCS, propulsion) / payload mass (PICK)
    design_life_yr: float = 5.0  # PICK
    laser_rated_shots: float = 1.0e10  # pump-diode / optics shot life (PICK)
    hardware_usd_per_kg: float = 50e3  # PICK: flight hardware incl. integration
    launch_usd_per_kg: float = 3e3  # PICK: LEO launch
    ops_usd_per_sat_yr: float = 2e6  # PICK: operations per platform-year


def platform_mass_kg(resources, laser, cfg: CostConfig) -> dict:
    """Mass breakdown [kg] of one platform."""
    r = resources
    parts = {
        "radiator": cfg.radiator_kg_m2 * r.radiator_area_m2,
        "array": r.array_area_m2 * r.array_specific_power_w_m2 / cfg.array_w_kg,
        "storage": r.battery_capacity_j / 3600.0 / r.storage_specific_energy_wh_kg,
        "pcm": r.pcm_mass_kg,  # pcm_specific_j_kg is system level (fins, container)
        "laser": cfg.laser_kg_per_kw * laser.burst_power_w / 1e3,
        "beam_director": cfg.beam_director_kg,
    }
    payload = sum(parts.values())
    parts["bus"] = cfg.bus_fraction * payload
    parts["total"] = payload + parts["bus"]
    return parts


def evaluate(sc, summary: dict) -> dict:
    """Mass / life / cost columns for one run, from its scenario and ``summary()`` row."""
    cfg = sc.cost
    m = platform_mass_kg(sc.resources, sc.laser, cfg)
    n = int(summary.get("n_platforms", sc.constellation.n_sats))
    shots = summary.get("laser_shots_per_sat_per_yr", 0.0)
    lives = {
        "design": cfg.design_life_yr,
        "storage": summary.get("storage_life_yr", np.inf),
        "laser": cfg.laser_rated_shots / shots if shots > 0 else np.inf,
    }
    limit = min(lives, key=lives.get)
    life = float(lives[limit])
    per_sat = m["total"] * (cfg.hardware_usd_per_kg + cfg.launch_usd_per_kg) + cfg.ops_usd_per_sat_yr * life
    total = n * per_sat
    removals = summary.get("removed_credited_per_yr", 0.0) * life
    dyears = summary.get("debris_years_removed_per_yr", 0.0) * life
    removed = summary.get("removed", 0)
    return {
        **{f"mass_{k}_kg": v for k, v in m.items()},
        "laser_life_yr": float(lives["laser"]),
        "mission_life_yr": life,
        "life_limit": limit,
        "cost_musd": total / 1e6,
        "removals_mission": removals,
        "debris_years_mission": dyears,
        "cost_per_debris_year_usd": total / dyears if dyears > 0 else np.inf,
        "cost_per_removal_kusd": total / removals / 1e3 if removals > 0 else np.inf,
        # share of removals credited (> 1 yr natural lifetime left): low late in a mission
        "credited_fraction": summary.get("removed_credited", 0) / removed if removed > 0 else np.nan,
        # active-satellite collisions avoided over the mission (talos.risk), annualised rate x life
        "cost_per_collision_avoided_musd": (total / (summary["active_collisions_avoided_per_yr"] * life) / 1e6
                                            if summary.get("active_collisions_avoided_per_yr", 0) > 0 else np.inf),
    }
