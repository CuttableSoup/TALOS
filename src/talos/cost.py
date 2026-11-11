"""Mass, mission life and cost per removal (roadmap items 4-5).

Per platform:

* mass = radiator + solar array + energy storage + PCM + laser + beam director,
  plus a bus fraction of that payload, plus station-keeping propellant and tankage;
* mission life = min(design life, storage end of life at the run's cycling,
  laser shot life at the run's firing rate);
* station keeping: the platforms fly fixed orbits in the sim (no drag), so the drag
  make-up is costed here: dv = integral of 1/2 rho v^2 Cd A / m over the mission life
  at each platform's altitude, with the scenario's solar-cycle atmosphere, and
  propellant from the rocket equation (station_keeping_dv);
* debris impacts (talos.survival, needs the ORDEM platform flux table): a constant
  hazard of mission-ending hits shortens the expected service life, which then
  scales operations cost and the debris-years delivered (hardware is paid regardless);
* cost = dry mass x (hardware + launch $/kg) + propellant x (launch + propellant $/kg)
  + operations x life.

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

from .constants import MU_EARTH, R_EARTH, YEAR

G0 = 9.80665


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
    # station keeping (drag make-up; the sim holds platform orbits fixed)
    drag_cd: float = 2.2  # PICK: flat-plate free-molecular, the usual LEO value
    drag_area_fixed_m2: float = 10.0  # PICK: bus + 1.5 m beam director cross-section
    drag_area_frac: float = 0.15  # PICK: share of array + radiator area facing the flow; dawn-dusk
    #   SSO sun-tracking arrays sit near edge-on to the velocity (sun near the orbit normal)
    sk_isp_s: float = 1500.0  # PICK: Hall-effect thruster
    sk_tank_fraction: float = 0.1  # PICK: tank + feed mass per kg of propellant
    propellant_usd_per_kg: float = 3e3  # PICK: xenon / krypton, bought (charged with launch $/kg)


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


def station_keeping_dv(sc, dry_mass_kg: float, drag_area_m2: float, life_yr: float,
                       n_steps: int = 400) -> np.ndarray:
    """Drag make-up dv [m/s] over ``life_yr`` from the run epoch, one value per platform
    (altitude round-robin as in baseline_constellation), at the scenario's atmosphere."""
    from .config import make_atmosphere

    atm = make_atmosphere(sc)
    n = sc.constellation.n_sats
    alt = np.array([sc.constellation.altitudes_km[k % len(sc.constellation.altitudes_km)] for k in range(n)])
    if not np.isfinite(life_yr) or life_yr <= 0:
        return np.zeros(n)
    t = (np.arange(n_steps) + 0.5) / n_steps * life_yr * YEAR
    r = R_EARTH + alt[:, None] * 1e3
    rho = atm.density(r - R_EARTH, t[None, :])
    accel = 0.5 * rho * MU_EARTH / r * sc.cost.drag_cd * drag_area_m2 / dry_mass_kg
    return accel.mean(axis=1) * life_yr * YEAR


def drag_area_m2(resources, cfg: CostConfig) -> float:
    return cfg.drag_area_fixed_m2 + cfg.drag_area_frac * (resources.array_area_m2 + resources.radiator_area_m2)


def survival_columns(sc, life_yr: float) -> dict:
    """Impact rates and expected service life per platform (talos.survival); NaN columns and
    service life = ``life_yr`` when disabled or the platform flux table is missing."""
    from . import survival

    n = sc.constellation.n_sats
    alts = np.array([sc.constellation.altitudes_km[k % len(sc.constellation.altitudes_km)] for k in range(n)])
    cols = {k: np.nan for k in ("impacts_bus_lethal_per_yr", "impacts_radiator_puncture_per_yr",
                                 "impacts_array_1mm_per_yr", "impacts_array_1cm_per_yr", "impact_hazard_per_yr",
                                 "survival_prob_mission")}
    cols["_service_life"] = np.full(n, life_yr)
    cols["service_life_yr"] = life_yr
    cfg = sc.survival
    from .atmosphere import decimal_year

    year = cfg.year or survival.nearest_table_year(decimal_year(sc.run.epoch) + 0.5 * min(life_yr, 50.0))
    if not cfg.enabled or not np.isfinite(life_yr) or year is None or not survival.table_path(year).exists():
        return cols
    table = survival.PlatformFluxTable.load(int(year))
    r = survival.impact_rates(table, cfg, alts, sc.resources.array_area_m2, sc.resources.radiator_area_m2)
    service, p = survival.expected_life(r["hazard"], life_yr)
    for k in ("bus_lethal", "radiator_puncture", "array_1mm", "array_1cm"):
        cols[f"impacts_{k}_per_yr"] = float(np.mean(r[k]))
    cols["impact_hazard_per_yr"] = float(np.mean(r["hazard"]))
    cols["survival_prob_mission"] = float(np.mean(p))
    cols["_service_life"] = service
    cols["service_life_yr"] = float(np.mean(service))
    cols["impact_flux_year"] = int(year)
    return cols


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
    area = drag_area_m2(sc.resources, cfg)
    dv = station_keeping_dv(sc, m["total"], area, life)
    prop = m["total"] * np.expm1(dv / (G0 * cfg.sk_isp_s))  # per platform, by altitude
    tank = cfg.sk_tank_fraction * prop
    dry = m["total"] + tank
    surv = survival_columns(sc, life)
    service = surv.pop("_service_life")  # per platform; = life without the impact model
    hw = cfg.hardware_usd_per_kg + cfg.launch_usd_per_kg
    per_sat = dry * hw + prop * (cfg.launch_usd_per_kg + cfg.propellant_usd_per_kg) + cfg.ops_usd_per_sat_yr * service
    total = float(np.sum(per_sat)) if len(per_sat) == n else n * float(np.mean(per_sat))
    service_mean = float(np.mean(service))
    removals = summary.get("removed_credited_per_yr", 0.0) * service_mean
    dyears = summary.get("debris_years_removed_per_yr", 0.0) * service_mean
    removed = summary.get("removed", 0)
    return {
        **{f"mass_{k}_kg": v for k, v in m.items()},
        "drag_area_m2": area,
        "sk_dv_mission_m_s": float(np.mean(dv)) if len(dv) else 0.0,
        "sk_dv_max_m_s": float(np.max(dv)) if len(dv) else 0.0,
        "mass_propellant_kg": float(np.mean(prop)) if len(prop) else 0.0,
        "mass_tank_kg": float(np.mean(tank)) if len(tank) else 0.0,
        **surv,
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
        "cost_per_collision_avoided_musd": (total / (summary["active_collisions_avoided_per_yr"] * service_mean) / 1e6
                                            if summary.get("active_collisions_avoided_per_yr", 0) > 0 else np.inf),
    }
