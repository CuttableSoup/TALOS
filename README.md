TALOS: Targeted Ablation Laser Orbital Sweeper.

# Project: Space-Based Laser Debris Removal — Monte Carlo Engagement Sim

## Goal
Quantify useful engagements/hr, Δv per engagement, duty cycle, and objects
removed/yr vs laser power, sat count, and orbit config. This is the key
unknown (estimates range ~4/hr to ~28/hr) driving array size, sat count, cost.

## Baseline point design (L'ADROIT class)
- Dawn-dusk SSO, 700/800/900 km staggered; 4–8 sats
- 355 nm, ~100 ps pulses, 1.5 m mirror, M²≈2, ~33 Hz
- 20–30 kW burst optical, 10–20% duty, burst/cooldown (e.g. 10 s on/120 s off)
- UV wall-plug ~15% (realistic 8–18%); arrays ~250 W/m² EOL
- Array ~1,300–2,000 ft² per sat; radiator ~600 W/m² net
- Engagement range 100–500 km; detection ~800 km; slew limit 2°/s
- Target: 1–10 cm debris, 600–1000 km band, n0 = 0.5–1.2M objects

## Key equations (Rogers & Lee, arXiv:2512.14682)
- Fluence: φ = η₂(E/π)·[2D/(M²·a·λ·R)]², a≈1.27
- Δv per pulse: Δv = η₁·Cm·φ/μ, along beam line of sight
- Fire only if φ ≥ ablation threshold (~0.85 J/cm² at 100 ps; 8.5 kJ/m² design)
- Cm: 30 (conservative) to 100 µN·s/J; make η₁ and off-axis Δv stochastic
- Removed = lifetime < 1 month or perigee < ~200 km
- Depletion: N·R₀ = n₀·ln10/T for 90% removal in T years

## Modules
1. Population: numpy sampler fit to ESA MASTER-2408 (alt, inc, RAAN, size, μ);
   5k–50k samples, scale results linearly
2. Propagator: vectorized numpy J2 secular + drag (mean elements);
   python-sgp4 SatrecArray for real TLEs
3. Platform/ATP: dataclasses — orbit, E, PRF, D, λ, M², FOV, slew, detection range, jitter (~0.3 µrad)
4. Screening: coarse altitude/apsis filter → 1 s fine stepping in windows;
   range, FOV, slew-rate checks
5. Engagement physics: equations above; integrate burst; project onto
   velocity for retrograde component
6. Scheduler: greedy (max retrograde Δv per burst energy); later OR-Tools CP-SAT
   or Pyomo+HiGHS receding-horizon ILP
7. Orbit update/lifetime: apply Δv, recompute elements; semi-analytic decay
   with pymsis density table
8. Resources: battery SoC + heat-buffer state machine gating bursts
   (duty cycle is an output, not an input)
9. Metrics + sweep driver: pandas + parquet + joblib; optional pySSEM for
   long-term environment impact

## Sweep matrix (~240 runs, 7–30 sim days each, extrapolate to 1 yr)
- Burst power: 5, 10, 20, 30, 50 kW
- Sats: 1, 2, 4, 8
- Orbits: all 800 km vs staggered 700/800/900
- Cm: 30, 100
- μ: 0.2, 1, 10 kg/m²

## Validation gates (in order)
1. Unit tests: spot size/range vs L'ADROIT (175–325 km small debris)
2. Reproduce Pieters & Noomen (ESA SDC8 2021) single-platform lifetime results
3. Slew limit reached at ~63 km for head-on pass
4. Raw encounter rate ~25/min at 325 km; useful fraction a few percent
5. Propagation vs orekit-jpype (30–90 days); screening vs ESA cascade;
   lifetimes vs DRAMA/OSCAR, NASA DAS, CNES STELA

## Stack
- Prototype: Python 3.10+, numpy, python-sgp4, pymsis, joblib, pandas, parquet
- Validation: heyoka.py, ESA cascade, orekit-jpype, Basilisk (power check),
  OR-Tools/Pyomo
- Avoid: poliastro (archived 2023); GPL libs (dSGP4, NASA-breakup-model-cpp)
  in core if distributing; MSIS2 is non-commercial
- Output format: time-stamped states + event log, designed for later playback
  (CesiumJS/CZML preferred over Unreal for visualization)

## Admin to-dos
- Register: ESA SDUP (MASTER, DRAMA), Space-Track.org
- Request NASA DAS/ORDEM via software usage agreement (slow)
- Consider contacting WVU (Hang Woon Lee lab) re: code

## Status
- Done (prototype): modules 2, 3, 4, 5, 6 (greedy), 7, end-to-end loop
  (`talos.simulation.run_engagements`), toy population, raw encounter rate.
- Run: `pip install -e .[dev]` then `pytest`; demo: `python scripts/demo_run.py N SATS HOURS MU`
- Engagement/removal counts are NOT weight-scaled (one target per burst);
  they're valid only when `saturation`≈1 and converged in N.

## Placeholders (replace with sourced values — see research prompt)
| Parameter | Where | Value | Basis |
|---|---|---|---|
| η₂ (transmission × Strehl) | `LaserConfig.eta2` | 0.35 | back-solved: threshold range ≈ 325 km |
| Cm model | `AblationParams.cm_model` | plasma | log ramp to φ_opt, then ∝ φ^-1/4 (Phipps) |
| Cm peak | `AblationParams.cm` | 30 µN·s/J | README conservative end |
| φ_opt | `AblationParams.phi_opt_factor` | 3 × threshold | guess |
| η₁ | `AblationParams.eta1_range` | U(0.3, 1.0) | guess |
| Δv off-axis angle | `AblationParams.off_axis_sigma_rad` | half-normal σ=20° | guess |
| Settle / retarget | `ATPConfig.settle_s` | 5 s, not overlapped | guess (conservative) |
| Min burst | `ResourceConfig.min_burst_s` | 5 s | guess |
| Heat buffer | `ResourceConfig.heat_buffer_j` | 1.5 MJ | ≈ one 10 s burst of net heat |
| Radiator area | `ResourceConfig.radiator_area_m2` | 20 m² | gives ~118 s recovery ≈ README's 120 s |
| Battery / bus | `ResourceConfig` | 2 kWh / 1.5 kW | guess |
| Field of regard | `ATPConfig.field_of_regard_half_angle_rad` | unrestricted | guess |
| Sunlight | `resources.py` | always sunlit | dawn-dusk SSO, eclipse season ignored |

## Open issues
- η₂ is back-solved to L'ADROIT's 325 km, so gate 1 is not an independent check yet
- Gate 4: geometry gives ~500/min within 325 km for 1M objects uniform in
  600–1000 km (hand check ~780/min) vs ~25/min target — reconcile definition/density
- Subsampling (README's "5k–50k samples, scale linearly") does not work for a
  target-saturated greedy platform: results depend on sample density because the
  scheduler picks the best of what's in view. Toy pop, 1 sat-hour:
  N=10k→100k→300k→1M gives mean Δv/eng 25→66→71→106 m/s, removals 4→22→31→34.
  Full density (N=n0) costs ~2 min per sat-hour after the coarse sieve.
- Removals look implausibly high (~34/hr/sat at 1M, toy pop) — driven by η₂=1,
  constant Cm at φ≫threshold, μ=1 everywhere, and no retarget/settle time

## Next step
Cm(φ) model, retarget/settle time, η₂ calibration, density strategy (full-density vs encounter-flux sampling), MASTER population (module 1), module 8.