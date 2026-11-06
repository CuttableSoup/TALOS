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

## Open issues
- η₂ uncalibrated (default 1.0 → ~560 km threshold range at 25 kW; L'ADROIT 175–325 km)
- Constant Cm overstates Δv at fluence ≫ threshold; need Cm(φ)
- Gate 4: geometry gives ~500/min within 325 km for 1M objects uniform in
  600–1000 km (hand check ~780/min) vs ~25/min target — reconcile definition/density
- Metrics not converged at N=10k (mean Δv/engagement 8→25 m/s from 2k→10k);
  need coarse window prediction for speed (~33 s per sat-2h at 10k)
- Module 8 placeholder: cooldown ∝ fire time

## Next step
Coarse time-window prediction in screening, Cm(φ) model, MASTER population (module 1), module 8.