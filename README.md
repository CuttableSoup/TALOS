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
- Cm: 25, 50, 100 N/MW
- η₂: 0.3, 0.4, 0.67
- μ: 0.2, 1, 10 kg/m²

## Validation gates (in order)
1. Unit tests: spot size/range vs L'ADROIT (175–325 km small debris)
2. Reproduce Pieters & Noomen (ESA SDC8 2021) single-platform lifetime results
3. Slew limit reached at ~63 km for head-on pass
4. Raw encounter rate ~25/min at 325 km; useful fraction a few percent
5. Propagation vs orekit-jpype (30–90 days); screening vs ESA cascade;
   lifetimes vs DRAMA/OSCAR, NASA DAS, CNES STELA

## Stack
- Prototype: Python 3.10+, numpy, numba (optional), python-sgp4, pymsis, pandas, pyarrow
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
- Done (prototype): modules 2–9 — propagator, platform/ATP, screening (coarse
  sieve + fine gates), engagement physics, scheduler (removal-progress scoring,
  batched prediction), lifetime/removal, power/thermal state machine, metrics,
  parallel sweep driver. Population is still a toy sampler (module 1 pending MASTER).
- Run: `pip install -e .[dev]` (+ `numba`, `pyarrow` for speed/parquet), then `pytest`.
  Demo: `python scripts/demo_run.py N SATS HOURS MU`; scheduler comparison:
  `python scripts/compare_scores.py DAYS`; sweeps: `talos.sweep.run_sweep(readme_matrix(), "runs/x")`.
- Engagement/removal counts are NOT weight-scaled (one target per burst);
  run at full density (N = n_real).

## Performance
- Numba `prange` kernels fuse propagation + Kepler + gates (`talos/_kernels.py`);
  numpy fallback with `TALOS_NUMBA=0`. 4 sats × 1 h × 100k objects: 16.5 s → 2.1 s.
- Sweeps fan out over processes (`n_jobs × threads_per_job = cpu_count`).
- GPU not used: the dev box has an AMD RX 6950 XT on Windows — no CUDA, no
  CuPy-ROCm on Windows, DirectML lacks usable float64. The CPU path is now
  dominated by Python-level scheduling rather than array math, so a GPU would
  not help much without restructuring. Revisit (pyopencl / ROCm on Linux) if
  1M-object runs become the norm.

## Placeholders (replace with sourced values — see research prompt)
| Parameter | Where | Value | Basis |
|---|---|---|---|
| η₂ (fluence efficiency) | `LaserConfig.eta2` | 0.40 (sweep 0.3–0.67) | **sourced**: backed out of L'ADROIT spot size + fluence |
| Fire threshold | `AblationParams.threshold_fluence` | 4.5 kJ/m² | **sourced** (research pass) |
| φ_opt (Cm peak) | `AblationParams.phi_opt` | 8.5 kJ/m² | **sourced** |
| Cm model | `AblationParams.cm_model` | plasma: log ramp, then ∝ φ^-1/4 | **sourced** falloff; ramp shape is a guess |
| Cm peak | `AblationParams.cm` | 50 N/MW (sweep 25–100) | **sourced** range; only ~100 ps Al point is ≤30 N/MW, IR not UV |
| η₁ | `AblationParams.eta1_mean/kappa` | Beta, mean 0.3, κ=10 | mean **sourced**; spread a guess |
| Δv off-axis angle | `AblationParams.off_axis_*` | N(45°, 15°) clipped [0, 90°] | centre **sourced**; spread a guess |
| Debris count in band | `toy_population(n_real=)` | 100k (Phipps) | MASTER all-LEO 1–10 cm ≈ 1.2M; 600–1000 km share unsourced |
| Settle / retarget | `ATPConfig.settle_s` | 5 s, not overlapped | guess (conservative) |
| Min burst | `ResourceConfig.min_burst_s` | 5 s | guess |
| Heat buffer | `ResourceConfig.heat_buffer_j` | 1.5 MJ | ≈ one 10 s burst of net heat |
| Radiator area | `ResourceConfig.radiator_area_m2` | 20 m² | gives ~118 s recovery ≈ README's 120 s |
| Battery / bus | `ResourceConfig` | 2 kWh / 1.5 kW | guess |
| Field of regard | `ATPConfig.field_of_regard_half_angle_rad` | unrestricted | guess |
| Sunlight | `resources.py` | always sunlit | dawn-dusk SSO, eclipse season ignored |

## Open issues
- η₁ (mean 0.3) and the 45° thrust angle may double-count: if the sourced η₁ is
  already a beam-axis average, applying cos(45°) on top cuts Δv by another ~30%
- Gate 1 not independent (η₂ backed out of the same L'ADROIT numbers)
- Gate 3: 63 km confirmed (Pieters & Noomen, SDC8 paper 43) but miss distance unstated
- Gate 4: 38/min within 250 km at 100k objects vs Phipps 19/min; ~2× gap
  attributed to full-sphere cross-section and relative-speed differences
- Subsampling invalid for a target-saturated greedy platform: run at full
  density (N = n_real). At 100k that's ~15 s per 2 sat-hours.
- Single-engagement Δv at baseline (μ=1, 50 N/MW, η₂=0.4) is ~10 m/s vs
  ~164 m/s needed from 800 km: μ=1 removal takes ~15–25 engagements, so it is
  revisit-limited. 30-day, 4-sat, 100k toy pop, μ=1 (runs/compare_30d):
  | score | removed (30 d) | debris-yr removed/yr | max hits/object |
  |---|---|---|---|
  | removal_progress (default) | 942 | 482k | 29 |
  | closest_to_removal | 982 | 272k | 69 |
  | dv_per_energy | 635 | 579k | 19 |
  Removal-aware scoring gives ~50% more removals; removal_progress keeps most of
  the debris-years benefit, so it stays the default.
- Removal rate is still climbing at day 30 (weekly removals 41→127→285→~380 for
  removal_progress): the README's "7–30 sim days, extrapolate to 1 yr" plan
  under-counts. Need longer runs or a steady-state detector. A 30-day 4-sat run
  takes ~30 min (3 runs in parallel, 10 threads each).
- Receding-horizon commitment planner (`talos/planner.py`, `RunConfig(planner=True)`)
  does NOT beat greedy removal_progress. 30-day, same setup (runs/compare_planner*):
  | variant | removed (30 d) | Δv/eng | committed frac | wall |
  |---|---|---|---|---|
  | greedy removal_progress | 942 | 9.7 | — | ~30 min |
  | v1 M=50 / 200 / 800 (book window open, always commit) | 637 / 304 / 167 | 8.7 / 6.6 / 4.3 | 15 / 46 / 81 % | 52–100 min |
  | v2 M=50 (best-in-window booking + 0.8 quality gate) | 895 | 9.73 | 4 % | 55 min |
  | v2 M=200 | 865 | 9.79 | 15 % | 108 min |
  v1 committed bursts got 2–3 m/s (half the pulses above threshold). v2 fixes
  burst quality, but reservations block fillers (~5–9 % fewer engagements) and
  most committed opportunities are declined anyway. Greedy removal_progress
  already revisits partly lowered objects; at ~10 m/s/engagement there is no
  margin for waiting. Kept as an option, off by default. Might matter when Δv per
  engagement is larger (μ=0.2) or for a platform not saturated with targets.
- `debris_years_removed` is dominated by small Δv on very long-lived (μ=10)
  objects and depends on the 200-yr cap

## Next step
Longer runs / steady-state detection; MASTER population (module 1). Optionally
re-test the planner at μ=0.2 and with fewer sats.
