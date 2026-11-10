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
- Δv per pulse: Δv = η₁·Cm·φ/μ, along beam line of sight (η₁ is the useful, along-beam
  efficiency; direction, shape and tumble are already in it — Phipps 2012 Eq. 6)
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
- ~~Request NASA DAS/ORDEM~~ done 2026-11: ORDEM 3.2 and DAS 3.2.7 installed locally (NASA SUA; not in git)
- Consider contacting WVU (Hang Woon Lee lab) re: code

## Status
- Done (prototype): modules 2–9 — propagator, platform/ATP, screening (coarse
  sieve + fine gates), engagement physics, scheduler (removal-progress scoring,
  batched prediction), lifetime/removal, power/thermal state machine, metrics,
  parallel sweep driver. Population: NASA ORDEM 3.2 sampler (module 1; MASTER
  cross-check pending); the uniform toy sampler remains as `population.kind = "toy"`.
- Scenario values live in `config/baseline.toml` (units in key suffixes, basis
  tags [sourced]/[derived]/[pick]/[guess]); `talos.config.load_scenario(path,
  overrides)` → `run_scenario(sc)`. Sweeps: `RunConfig(name, config=None,
  overrides={"laser.burst_power_kw": 10, ...})`, `run_sweep(readme_matrix(), "runs/x")`.
- Run: `pip install -e .[dev]` (+ `numba`, `pyarrow` for speed/parquet), then `pytest`.
  Demo: `python scripts/demo_run.py N SATS HOURS MU`; scheduler comparison:
  `python scripts/compare_scores.py DAYS`. Single run with progress log:
  `python scripts/run_baseline.py DAYS --out runs/x [--threads N] [--config path.toml] [--set section.key=value ...]`.
  Without `pip install -e .`, set `PYTHONPATH=src` first.
- 3D playback: `python scripts/render_run.py --config config/design.toml --days 1 --out
  runs/render/x.html --serve`, or `run_baseline.py ... --render-days 1` (see "Playback viewer").
- Scenarios: `config/baseline.toml` (reference, all earlier results) and `config/design.toml`
  (design baseline for long runs: 60 m² radiator, 2 kWh supercap, 20 s bursts). A TOML may
  start with `base = "other.toml"` and list only the keys that differ.
- Engagement/removal counts are NOT weight-scaled (one target per burst);
  run at full density (N = n_real).

## Performance
- Numba `prange` kernels fuse propagation + Kepler + gates (`talos/_kernels.py`);
  numpy fallback with `TALOS_NUMBA=0`. 4 sats × 1 h × 100k objects: 16.5 s → 2.1 s.
- Sweeps fan out over processes (`n_jobs × threads_per_job = cpu_count`).
- 2026-11 pass (ORDEM baseline, 4 sats, 69.6k objects, 32 threads): drag rates
  and the whole lifetime integrator (`orbital_lifetime`, incl. shell residence)
  run as numba kernels on the MSIS table; the kernel element arguments are
  built once per chunk for all platforms; window extraction is a numba scan.
  The fine-gate kernel skips steps that cannot be back in range yet
  (range-rate bound), and the coarse sieve runs at 8×/2×/1× its step. Build (1-yr
  lifetime filter) 61 s → 2.3 s, end-of-run lifetime/risk ~10 s → ~1 s, 33 → ~16 s
  per sim day; quarter-day run 73 s → 4.8 s with identical events. A 1-yr run is
  now ~1.6 h (before the loop was compiled; see below).
- Scheduling loop compiled (2026-11): candidate burst prediction, the
  resource state machine (evolve / advance / max_fire_s / wait_for; same
  operation order, bit-identical to the Python fallback) and window extraction
  run in numba; screening kernels index whole-population arrays from a
  per-chunk cache. 3-day ORDEM baseline 51 s → 30 s (~9.5 s per sim day, 3.5×
  vs 33 s before the speed work); identical events throughout. A 1-yr run is
  ~1 h. What is left is mostly the compiled kernels themselves plus Python
  glue per engagement (orbit updates, event records); runs do not scale past
  ~8 threads.
  Sweeps should use `n_jobs = 4, threads_per_job = 8` (4 runs for the price of one).
- Checkpoints: runs with an out_dir write `checkpoints/<name>.pkl` (full state,
  ~15 MB + events) every `run.checkpoint_days` (30). `run_baseline.py DAYS --resume
  <pkl>` continues or extends a run (same config; only name / duration may
  differ) and reproduces the uninterrupted run exactly. Year 2 or a longer
  horizon no longer re-runs year 1.
- `run.predict_pulse_stride` (scheduler predicts every n-th pulse) tested at 5
  and 10: 3-day baseline 58 → 50 s, Δv/engagement 5.84 → 6.10 / 6.16 m/s vs a
  seed-to-seed spread of 5.84 / 6.05. Not worth the possible bias; default 1.
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
| η₁ | `AblationParams.eta1_mean/kappa`, `eta1_basis` | Beta, mean 0.3, κ=10; `along_beam` | mean **sourced** (Phipps 2012 η_c = 0.3 after Liedahl); basis **sourced** (η_c covers improper thrust direction); spread a guess |
| Δv off-axis angle | `AblationParams.off_axis_*` | N(45°, 15°) clipped [0, 90°] | centre **sourced**; spread a guess. With `along_beam` it sets only the sideways scatter (magnitude η₁/cos 45°) |
| Debris count in band | `population.kind = "ordem"` | 75.9k 1–10 cm, 600–1000 km (ORDEM 3.2, 2026) | **sourced** (ORDEM); inclination mix is a fit to ORDEM flux spectra; toy: 100k (Phipps) |
| Settle / retarget | `ATPConfig.settle_s` | 5 s, not overlapped | guess (conservative) |
| Min burst | `ResourceConfig.min_burst_s` | 5 s | guess |
| Wall-plug (elec → 355 nm) | `ResourceConfig.wallplug_eff` | 0.10 (range 0.04–0.17) | **sourced** chain 0.9×0.55×0.45×0.40; ATLAS flight 8.2 % at 532 nm |
| Radiator net flux | `ResourceState.q_net(T)` | fin·sides·[εσT⁴ − εF_E·237 − 5], ε 0.9, F_E 0.22, fin 0.85, 2-sided → 445 W/m² at 280 K | **sourced** inputs, derived relation |
| Store → radiator ΔT | `radiator_dT_k` | 8 K (pumped two-phase; LHP ~20 K) | **sourced** (ATLAS, AMS-02 TTCS) |
| PCM store | `pcm_melt_k`, `pcm_mass_kg`, `pcm_specific_j_kg` | 15 °C, 25 kg × 120 kJ/kg = 3 MJ | storage density **sourced**; melt point / salt-hydrate choice unsourced |
| Sensible heat capacity | `heat_capacity_j_k` | 0.5 MJ/K | **guess** (0.2–1) |
| Store temperature band | `t_min_k`, `t_max_k` | melt −5 K (heaters) … melt +2 K | guess |
| Radiator area | `ResourceConfig.radiator_area_m2` | 20 m² | **design choice, dominant**: 20 m² → 2.9 % duty; ~60 m² for 10 % |
| Battery / storage | `ResourceConfig`, `STORAGE_PRESETS` | 2 kWh; Li-ion 85 % RT (baseline), supercap 95 % (design) | capacity pick. Cycle life: Li-ion shape consistent with ~30k cycles at < 40 % DoD (LEO requirement); LTO ≥ 25k (vendor); supercap 500k cycles, 3.8–5.4 Wh/kg (Maxwell 160 V module datasheets) |
| Fragmentation | `fragment_prob_per_burst` | 0 (off) | **guess**: no published probability (NASA OTPS 2024, JAXA 2023: no systematic data); see Sourcing pass 2 for a derived melt bound |
| Bus | `bus_power_w` | 1.5 kW, dissipated into the thermal node | guess |
| Field of regard | `ATPConfig.field_of_regard_half_angle_rad` | unrestricted | guess |
| Epoch / LTAN | `[run] epoch`, `[constellation] ltan_hours` | 2026-12-21, 06:00 | pick: worst-case eclipse (~17 min/orbit at 800 km) |
| Sun model | `talos/sun.py`, `[sun]` | Almanac ephemeris, conical shadow, 2-axis array, no eclipse firing | physics; pointing + eclipse policy are picks |
| Target lighting | `[atp] require_target_sunlit` | true (passive optical tracking) | pick |
| Sun exclusion | `[atp] sun_exclusion_deg` | 30° | pick (30–45 typical, unsourced) |
| Detection model | `[atp] detection_model` | passive_optical: Lambertian sphere magnitude, acquire ≤ 16 mag for 10 s, track ≤ 18 mag, +3 mag on sunlit-Earth background | all picks; research needed |
| Debris albedo | `[population] albedo_median/log_sigma` | 0.1, lognormal σ=0.5 | pick (debris ~0.1–0.2) |
| Atmosphere | `[atmosphere]` | NRLMSIS 2.1 global mean (Ap 15) vs F10.7; cycle min 70 / max 155, cycle-25 timing | table derived from MSIS; forecast is a pick (use NOAA SWPC) |
| Background source | `[dynamics] source_mode` | `replace_decays`: each natural decay of a never-engaged object is replaced by an identical object (fresh phases), i.e. the un-lasered band is in steady state; `fraction` = `background_frac_per_yr` (2 %/yr) | assumption (ORDEM forecasts growth, not modelled) |
| Protected shells | `[risk] shell_*_km` | crewed 370–430 km, Starlink 530–575 km | pick |
| Uncertainty ranges | `talos.sweep.UNCERTAIN_INPUTS` | η₂ 0.3–0.67, Cm 25–100 (log), η₁ 0.2–0.4, thrust angle 30–60°, wall-plug 0.04–0.17 (log), sun excl. 30–45°, settle 2–10 s | sourced ranges where the table has one, else picks |
| Breakups | `breakup_rate_per_yr`, `breakup_fragments`, `breakup_dv_sigma_m_s` | 0.5/yr × 2000 fragments, σ 50 m/s per axis | pick (NASA SBM gives ~100 m/s for A/m ~1) |
| Radiator absorptivity | `radiator_absorptivity` | 0.20 (albedo load) | 0.16 BOL sourced; EOL pick |
| Radiator turndown | `radiator_turndown_min` | 10 % (louvers) before heaters | pick |
| Storage cycle life | `storage_cycles_100`, `storage_cycle_exponent` | Li-ion 5000 cycles at 100 % DoD, k 1.7 (~41k at 29 %); LTO 20k, k 1.2; supercap 1e6, k 0 | picks (order of magnitude) |
| Storage specific energy | `storage_specific_energy_wh_kg` | Li-ion 120, LTO 70, supercap 5 Wh/kg (pack) | picks |
| Burst readiness | `ready_burst_s` | = burst_s, capped at 95 % of the feasible burst | policy |
| Mass / cost | `[cost]` | radiator 10 kg/m², array 100 W/kg, laser 20 kg/kW, beam director 300 kg, bus 30 %; $50k/kg hardware, $3k/kg launch, $2M/sat-yr ops; 5 yr design life, 1e10 shots | picks except radiator (research ~10 kg/m²) |

## Open issues
- Thermal model (2026-11 research pass) is now a lumped PCM/loop node with T⁴
  radiator. Baseline duty fell 8.5 % → 2.9 % (wall-plug 15 → 10 %, net flux
  600 → 445 W/m², bus + battery heat). 1-day, 4 sats, 100k (engagements/hr/sat):
  | radiator | wall-plug 0.10 | wall-plug 0.15 |
  |---|---|---|
  | 20 m² | 10.9 (2.9 % duty) | 17.5 (4.7 %) |
  | 40 m² | 23.9 (6.4 %) | 38.4 (10.3 %) |
  | 60 m² | 37.0 (9.9 %) | — |
  Engagement rate is linear in radiator area at ~10 kg/m² (research). **All
  30-day results below used the old 8.5 % thermal model and overstate removals
  at a 20 m² radiator by ~7× (942 → 134 removed in 30 d).**
- Sun model (2026-11): 1-day, 4 sats, 100k, 20 m² radiator (engagements/hr/sat):
  sun off 10.9 · June (no eclipse) 10.8 · December 10.1 (17.6 % eclipse) ·
  December without the sunlit-target gate 10.1. Eclipse costs ~7 % because
  the platform is heat-limited and the radiator keeps rejecting in shadow; the
  sunlit-target gate costs ~5 % Δv/engagement (fewer choices), not rate. No
  heater energy or brownouts at 1.5 kW bus. A 06:00 LTAN gives the Dec–Jan
  eclipse season (18:00 → June); eclipse onset is ~2 months either side.
- 30-day, TOML baseline (Dec 21 start, sun + thermal model, 4 sats, 100k, μ=1),
  `scripts/compare_radiator.py` → runs/compare_radiator:
  | radiator | eng/hr/sat | duty | removed (30 d) | credited/yr | debris-yr/yr | weekly removals |
  |---|---|---|---|---|---|---|
  | 20 m² | 10.2 | 2.7 % | 134 | 1.6k | 169k | 8, 29, 28, 52 |
  | 40 m² | 21.1 | 5.6 % | 432 | 5.3k | 330k | 20, 75, 141, 159 |
  | 60 m² | 32.0 | 8.5 % | 776 | 9.4k | 477k | 28, 143, 206, 303 |
  Removals grow ~quadratically with engagement rate (3.1× rate → 5.8×
  removals): an object needs ~17 bursts, so more shots per day finish more
  objects inside the window. 60 m² matches the old model's 8.5 % duty but
  removes 776 vs 942: eclipse (15.6 %) and the sun gates cost ~18 %. Removal
  rate still climbing at day 30 at every area. Heaters draw 0.3–2.8 kWh/day
  at 40–60 m² (bigger radiator overcools in eclipse); no brownouts.
- Detection & tracking (2026-11, picks): a target is engageable only after
  10 s of continuous detection (sunlit, outside sun exclusion, ≤ 800 km,
  V ≤ 16 incl. +3 mag on a sunlit-Earth background), and stays so while
  V ≤ 18. A target detectable at a chunk start is assumed already tracked.
  1-day, 4 sats, 100k, 20 m² (runs/detection_1d):
  | sensor (detect/track) | eng/hr/sat | Δv/eng | median target | < 3 cm |
  |---|---|---|---|---|
  | ideal | 10.1 | 9.18 | 2.9 cm | 51 % |
  | 18/20 mag | 10.2 | 9.32 | 3.1 cm | 49 % |
  | 16/18 (baseline) | 10.2 | 8.73 | 3.9 cm | 40 % |
  | 14/16 | 9.9 | 7.06 | 5.4 cm | 20 % |
  Rate is unchanged because the heat-limited platform still has many targets
  to choose from; detection costs Δv/engagement (fewer choices) and shifts
  removals toward larger objects (Δv is size-independent at fixed μ). Expect a
  bigger hit at larger radiators / smaller populations, where the platform is
  less target-rich. 1 cm objects are V ≈ 17 at 500 km.
- Atmosphere + population dynamics (2026-11): drag and lifetimes use a
  solar-cycle-dependent NRLMSIS 2.1 table (src/talos/data/msis_density.npz,
  rebuilt by `scripts/build_density_table.py`; MSIS is NRL non-commercial —
  the table is derived output). Solar max/min density ratio is ~35× at
  800 km. Every debris object now carries its own epoch, so propagation,
  engagements and the daily decay step use one consistent drag model (this
  also fixes post-engagement elements accumulating drag from t = 0). The
  1-yr lifetime filter now keeps ~69k of 100k toy objects at a Dec-2026
  epoch (F10.7 ≈ 128). Natural re-entries leave the screening set but are not
  credited as removals; `net_change_per_yr` = live objects end − start.
- Battery cycle life: ~600 bursts/day ≈ 1.1 M cycles in 5 yr at ~30 % DoD and
  ~70C; space Li-ion is tested to ~27–65 k cycles. Not modelled; needs a
  supercapacitor/LTO buffer or a much larger battery. Unsourced.
- Not modelled: M²/thermal-lens transients at burst start
  (fluence ∝ 1/M⁴), THG crystal detuning (±0.1 K), diode wavelength drift,
  heater power, radiator EOL absorptivity and self-view (−10–30 %).
- ~~η₁ and the 45° thrust angle may double-count~~ resolved 2026-11-09: they did
  (see Sourcing pass 2); `eta1_basis = "along_beam"` is now the default.
- Gate 1 not independent (η₂ backed out of the same L'ADROIT numbers)
- Gate 3: 63 km confirmed (Pieters & Noomen, SDC8 paper 43) but miss distance unstated
- Gate 4 (toy population): 38/min within 250 km at 100k objects vs Phipps 19/min; ~2× gap
  (see "Gate 4 vs ORDEM" below: the gap is in the population density, not the geometry)
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

## Roadmap: elements still to add (besides ORDEM / DAS, pending approval)
Likely to change the bottom line:
1. **Detection & tracking** — onboard detection probability vs size, range,
   phase angle and background; acquisition time; track loss. The sim used to
   assume perfect knowledge of every object. *(first version done: passive
   optical, see Open issues; detection probability is a hard threshold, no
   orbit-knowledge pointing error yet)*
2. **Population sources & sinks** — new debris from breakups/launches and
   natural decay of the whole population; needed for net benefit and for
   meaningful steady-state runs. *(first version done: daily whole-population
   decay + background source + Poisson breakups; rates are picks)*
3. **Solar-cycle atmosphere** — pymsis with an F10.7/Ap forecast from 2026;
   density varies several-fold over the cycle, shifting lifetimes and the Δv
   needed for removal. *(done: NRLMSIS 2.1 global-mean table vs F10.7 +
   parametric cycle-25 forecast; forecast values are picks)*
4. **Hardware life** — battery cycle life (~1.1 M cycles needed), laser diode
   life by pulse count, UV optics damage/contamination, array degradation:
   removals over the mission, not per year. *(done, picks: storage cycle life by
   Miner's rule on per-burst DoD, N = N100·DoD^-k, presets li_ion / lto /
   supercap in `STORAGE_PRESETS`; `storage_life_yr`; laser shot life in
   talos.cost. No array degradation or optics damage yet)*
5. **Mass & cost model** — radiator (~10 kg/m²), PCM, arrays, battery, laser →
   launch cost → cost per removal. *(done, picks: `talos.cost`, `[cost]`; every
   sweep row gets the mass breakdown, mission life and its limit, constellation
   cost and cost per credited removal)*

Worth adding:
6. **Risk transfer & safety** — time lowered debris spends crossing ISS /
   Starlink shells; keep-out for active satellites. *(residence done: object-years
   inside each `[risk]` shell for every engaged object, natural path vs after the
   laser, to re-entry or 200 yr; no collision probability or keep-out yet)*
7. **Uncertainty propagation** — Latin-hypercube over η₂, Cm, wall-plug, η₁,
   sun exclusion, etc., for confidence bands. *(driver done: `lhs_matrix`,
   `uncertainty_bands` (quantiles + Spearman rank correlation),
   `scripts/run_uncertainty.py N DAYS`)*
8. **Attitude / slew dynamics** — reaction-wheel torque and momentum for a
   1.5 m beam director, replacing the 5 s settle guess.
9. **Material-dependent coupling** — Al vs MLI vs paint vs composite;
   fragmentation risk from ablation.
10. **Constellation coordination** — cross-platform hand-off, avoiding
    duplicate shots.

Validation still owed: gate 2 (Pieters & Noomen reproduction), gate 5
(orekit / STELA). MASTER also needs ESA SDUP registration.

## ORDEM population (2026-11)
`talos.ordem` runs ORDEM 3.2 spacecraft mode for circular orbits (600–1000 km
every 20 km × 10 spacecraft inclinations), keeps the flux per relative-velocity
bin for each material class and half-decade size, and fits non-negative
inclination-bin counts (5° bins) that reproduce those spectra at every
spacecraft inclination at once. The forward model is the single-inclination
density 1/(2π²r²√(cos²φ − cos²i)) integrated over debris heading (exact vs the
isotropic analytic flux). Build: `python scripts/build_ordem_table.py` (~40 min,
writes the gitignored `src/talos/data/ordem_<year>.npz`; `--refit` reuses the
raw spectra). Installed at `%LOCALAPPDATA%\NASA\ORDEM 3.2` (or `TALOS_ORDEM_HOME`);
`ORDEM32.exe` must run from the install root.
- 2026, 1–10 cm, 600–1000 km: 75.9k objects (toy guess was 100k). MD 64 %,
  NaK 22 % (RORSAT droplets, 60–70°), HD 10 %, LD 5 %; 1–3.16 cm 86 %.
  Inclination: 95–105° (SSO) 37 %, 60–85° ~47 %. Altitude peak 840–900 km.
- Check: fitted density / ORDEM density along each test orbit is 0.95–1.04
  for spacecraft inclinations ≥ 30°; equatorial orbits are 8–20 % low (eccentric
  debris counted as circular). Spectral fit residual 7–13 % by material.
- With μ = 1 the 1-yr lifetime filter keeps 63.5k of the 75.9k.
- 1-day baseline, ORDEM vs toy (runs/ordem_vs_toy_1d): 9.99 vs 10.17 eng/hr/sat,
  9.73 vs 9.31 m/s Δv/engagement, duty 2.7 % both: still heat-limited, slightly
  better Δv from the SSO-concentrated population.
- 30-day baseline, ORDEM vs toy (runs/ordem_baseline_30d vs runs/baseline_30d;
  4 sats, 20 m², μ=1, Dec 21 start):
  | population | live objects | eng/hr/sat | Δv/eng | duty | removed (30 d) | debris-yr/yr | eng/target | targets ≥10 eng |
  |---|---|---|---|---|---|---|---|---|
  | toy 100k | 68.7k | 10.21 | 9.21 | 2.7 % | 138 | 241k | 1.75 | 133 |
  | ORDEM 2026 | 63.5k | 10.09 | 9.29 | 2.7 % | 335 | 235k | 2.17 | 368 |
  Same shot rate and Δv, but 2.4× the removals: the SSO-concentrated debris
  stays near-coplanar with the platforms, so the same objects come back into
  range and the removal-progress scheduler finishes them (~11 engagements
  each). Debris-years removed is about equal despite 2.4× the removals: it sums
  lifetime reduction over every engaged object (200-yr cap), so it tracks total
  Δv delivered (29.1k × 9.29 ≈ 270k m/s vs 29.4k × 9.21 ≈ 271k m/s), mostly
  partial shots on long-lived high objects; a removal adds at most the object's
  remaining lifetime (~years at 750 km). No breakup was drawn
  in the ORDEM run (Poisson, 0.5/yr) and no natural decays (fewer low objects),
  so net change is −2.4k/yr vs +23k/yr in the toy run, which had one breakup.
  Removals are still climbing at day 30.
- Areal density from the NASA standard breakup model (`areal_density_model =
  "sbm"`, now the baseline): fragment log10(A/m) ~ N(μ_soc(λc), σ_soc(λc))
  (Johnson et al. 2001 small-object distribution, ±3σ clip), μ = 1/(A/m);
  NaK droplets are solid spheres (μ = ⅔ρd, ~6–60 kg/m²). Population median μ
  4.45 kg/m² (10–90 %: 0.95–12.9) vs the old fixed 1.0. 30-day baseline
  (runs/compare_mu_model vs runs/ordem_baseline_30d):
  | μ model | eng/hr/sat | Δv/eng | removed (30 d) | debris-yr/yr | eng/target | max hits |
  |---|---|---|---|---|---|---|
  | fixed 1.0 | 10.09 | 9.29 | 335 | 235k | 2.17 | — |
  | SBM | 10.10 | 5.66 | 250 | 171k | 2.66 | 41 |
  Only −25 % removals for −39 % Δv because the scheduler finishes the light
  tail: removed objects have median μ 0.54 (10–90 %: 0.23–1.35), 82 % have
  μ < 1 vs 11 % of the population, median size 1.4 cm. Removal counts therefore
  hinge on the SBM high-A/m tail (MLI/paint-like flakes) and its clip; targets
  overall have median μ 1.8.
- Approximations: circular orbits at the altitude seen (no eccentricity
  structure), ORDEM uncertainty files unused, power-law size inside each
  half-decade, background sources drawn from the same table. Material classes
  give an optional μ = (2/3)ρd sphere model (`areal_density_model = "sphere"`);
  the baseline is SBM (above).
- Licence: ORDEM is NASA SUA software; tables are derived data and stay out of git.

## 1-year baseline (2026-11)
ORDEM 2026 + SBM μ, 4 sats, 20 kW, 365 sim days (`python scripts/compare_mu_model.py
365 --models sbm --out runs/baseline_1yr`, 3.4 h wall). 69.6k samples at weight
≈ 1, so counts are physical. Saturation 0.99, no brownout, no breakup drawn.
- Year totals: 10.63 useful eng/hr/sat, Δv 4.38 m/s per engagement, **5,443
  removed** (5,412 credited), 144k debris-yr/yr, 3,483 natural decays,
  1,588 background additions, net −7.3k/yr.
- The 30-day × 12 extrapolation (3.0k/yr) is not the annual rate. Removals
  need repeated engagements (median 13 per removed object, median 113 d
  from first engagement to removal), so the monthly count first rises, then
  falls as the light tail is used up:
  | month | removed | Δv/eng (m/s) | median μ, removed | engaged with μ < 1 |
  |---|---|---|---|---|
  | 1 | 254 | 5.66 | 0.54 | 34 % |
  | 2 | 425 | 5.50 | 0.58 | 33 % |
  | 3 | 505 | 5.31 | 0.60 | 32 % |
  | 4 | 565 | 4.92 | 0.64 | 29 % |
  | 5 | 577 | 4.61 | 0.71 | 26 % |
  | 6 | 533 | 4.45 | 0.76 | 25 % |
  | 7 | 552 | 4.13 | 0.80 | 21 % |
  | 8 | 497 | 3.94 | 0.85 | 19 % |
  | 9 | 544 | 3.74 | 0.92 | 17 % |
  | 10 | 433 | 3.69 | 0.89 | 14 % |
  | 11 | 329 | 3.53 | 0.99 | 12 % |
  | 12 | 229 | 3.09 | 1.01 | 8 % |
  Engagements hold at ~31k/month throughout. Only the target mix changes.
- The low-μ pool gets used up. Share of the initial population removed in
  one year, by μ (kg/m²): < 0.5: 71 % (1,265/1,775); 0.5–1: 40 %; 1–2: 13 %;
  2–5: 1.3 %; ≥ 5: 2 of 32.6k. 68 % of removals have μ < 1 (11 % of the
  population).
- So no steady state in year 1. The sustained rate is set by how fast low-μ
  objects are replenished (background sources, breakups), not by the year-1
  harvest of the existing stock. Month 12 is at 229 and still falling; a
  year-2 run is needed to find the floor.
- Side issue: without the laser the sampled population also shrinks: background
  additions are 1.6k/yr against 3.5k/yr natural decays. The background source rate is a pick.
- Runs started from now on also write `logs/<name>.log`, one line per sim
  day, and `events/<name>.partial.parquet`, an event checkpoint saved every sim week.

## Thermal / burst design levers, ORDEM + SBM baseline (2026-11)
30 days, 4 sats, Dec 21 start (`python scripts/compare_thermal.py 30` →
runs/compare_thermal, runs/compare_thermal_b; ~6 min per run, 4 in parallel).
Removals are 30-day counts and still climbing (weekly 30/60/81/84 at baseline),
so compare variants, don't annualise.
| variant | eng/hr/sat | Δv/eng (m/s) | duty | removed | vs base | battery EFC/sat/yr | DoD/burst |
|---|---|---|---|---|---|---|---|
| base (20 m², 10 s bursts, PCM 25 kg, wall-plug 0.10) | 10.1 | 5.64 | 2.7 % | 267 | 1.00 | 26.6k | 29 % |
| PCM 75 kg | 10.6 | 5.54 | 2.9 % | 260 | 0.97 | 27.8k | 29 % |
| wall-plug 0.15 | 16.4 | 5.19 | 4.4 % | 423 | 1.58 | 26.2k | 18 % |
| bursts 5 s | 20.0 | 2.68 | 2.8 % | 160 | 0.60 | 27.1k | 15 % |
| bursts 20 s | 5.0 | 10.71 | 2.6 % | 320 | 1.20 | 25.7k | 56 % |
| bursts 30 s | 0 | — | 0 | 0 | 0 | — | — |
| radiator 40 m² | 20.8 | 5.06 | 5.6 % | 545 | 2.04 | 53.8k | 29 % |
| radiator 60 m² | 31.7 | 4.54 | 8.5 % | 809 | 3.03 | 81.3k | 29 % |
| 40 m² + PCM 75 kg | 22.2 | 4.88 | 6.0 % | 562 | 2.10 | 57.3k | 29 % |
| 40 m² + 20 s bursts | 10.6 | 9.64 | 5.5 % | 721 | 2.70 | 52.6k | 56 % |
| 60 m² + 20 s bursts | 16.1 | 8.86 | 8.4 % | 1091 | 4.09 | 79.8k | 55 % |
| 20 s bursts + wall-plug 0.15 | 8.3 | 9.80 | 4.3 % | 505 | 1.89 | 25.6k | 34 % |
- Removals scale ~linearly with radiator area under SBM μ (2.0× at 40 m²,
  3.0× at 60 m²; the old μ = 1 toy runs grew faster than linear). Δv per
  engagement falls as the platform fires more often (fewer targets to choose
  from per opportunity); saturation drops to 0.87 at 60 m².
- PCM mass does nothing at 10 s bursts: 25 kg already covers a burst, and
  sustained duty is set by radiator rejection, not buffer size. It does set
  the longest possible burst (below).
- Burst length is the cheapest lever: 20 s bursts give +20 % at 20 m² and
  +32–35 % at 40–60 m² (same duty, half the engagements, twice the Δv each:
  fewer 5 s settles, Δv concentrated on the objects that are being finished).
  5 s bursts lose 40 %. The ceiling is whichever store fills first: from a
  cold store and full battery the baseline fires 26.0 s before the thermal
  store (25 kg PCM + 0.5 MJ/K) tops out; the 2 kWh battery alone would allow
  33.6 s. (Corrected: this section first blamed the battery.) Longest burst,
  thermal store vs battery: PCM 25 kg → 26 s at any battery size; 75 kg → 34 s
  (2 kWh) / 50 s (4–8 kWh); 150 kg → 34 / 67 / 86 s (2 / 4 / 8 kWh). 30 s
  bursts never fired because the scheduler waited for a full burst that can
  never fit; `ready_burst_s` now caps that wait at 95 % of the feasible burst.
  Longer bursts mean deeper cycles (56 % DoD) on a battery that already needs
  ~26k equivalent full cycles a year.
- Wall-plug 0.15 vs 0.10: +58 % (less waste heat per joule of light).
- Battery cycling scales with duty: ~27k EFC/sat/yr at 20 m², ~80k at 60 m²,
  against ~27–65k cycles for space Li-ion over a whole life. Every variant that
  removes more needs a battery redesign (supercapacitor / LTO buffer, bigger
  pack) before cost per removal means anything.
- Risk transfer (Starlink-shell object-years, lasered − natural) is ±hundreds per
  year with no consistent sign across variants: too noisy at 30 days to rank.

## Burst length × energy storage × cost per removal (2026-11)
21 days, ORDEM + SBM baseline (`python scripts/compare_design.py 21` →
runs/compare_design). Storage presets and every cost coefficient are picks
(Placeholders table); "credited/yr" is the 21-day rate annualised (still on
its ramp), and cost per removal = constellation cost / (that rate × mission
life). Compare rows; don't quote the absolute dollars.
| variant | burst (s) | eng/hr/sat | Δv/eng | removed | credited/yr | DoD | storage life (yr) | mass/sat (kg) | life (limit) | cost/removal |
|---|---|---|---|---|---|---|---|---|---|---|
| base: 20 m², Li-ion 2 kWh, 10 s | 10 | 10.0 | 5.8 | 171 | 3.0k | 29 % | 0.46 | 1837 | 0.46 (storage) | $289k |
| 20 s bursts | 20 | 5.0 | 10.9 | 216 | 3.8k | 56 % | 0.30 | 1837 | 0.30 (storage) | $350k |
| longest feasible burst | 24.7 | 3.9 | 13.0 | 197 | 3.4k | 70 % | 0.26 | 1837 | 0.26 (storage) | $437k |
| longest, ready at 20 s | ≤ 24.7 | 4.8 | 11.1 | 186 | 3.2k | 58 % | 0.29 | 1837 | 0.29 (storage) | $420k |
| PCM 75 kg, longest | 31.9 | 3.3 | 15.7 | 240 | 4.2k | 89 % | 0.20 | 1902 | 0.20 (storage) | $481k |
| PCM 75 kg + Li-ion 4 kWh, longest | 47.5 | 2.1 | 20.7 | 215 | 3.7k | 67 % | 0.51 | 1924 | 0.51 (storage) | $214k |
| supercap 2 kWh, 20 s | 20 | 5.4 | 11.1 | 248 | 4.3k | 56 % | 20 | 2336 | 5.0 (design) | **$25k** |
| LTO 4 kWh, 20 s | 20 | 5.2 | 10.9 | 222 | 3.9k | 28 % | 2.0 | 1890 | 2.0 (storage) | $55k |
| 60 m², Li-ion 2 kWh, 20 s | 20 | 15.9 | 9.3 | 712 | 12.4k | 55 % | 0.09 | 2357 | 0.09 (storage) | $433k |
| 60 m², supercap 2 kWh, 20 s | 20 | 17.4 | 9.0 | 733 | 12.7k | 55 % | 6.2 | 2856 | 5.0 (design) | **$10k** |
| 60 m², LTO 4 kWh, 20 s | 20 | 16.7 | 9.2 | 747 | 13.0k | 28 % | 0.62 | 2410 | 0.62 (storage) | $65k |
| 60 m², PCM 75 kg, LTO 4 kWh, longest | 53.6 | 6.2 | 21.5 | 818 | 14.2k | 74 % | 0.51 | 2475 | 0.51 (storage) | $73k |
- Storage chemistry decides cost per removal (10–40×), not removals: with
  Li-ion the pack wears out in 0.1–0.5 yr in every design, so the platform's
  cost is spread over a few months of removals. A 2 kWh supercapacitor bank
  (400 kg at 5 Wh/kg) reaches the 5-yr design life and is cheapest per removal
  in spite of its mass; its 95 % round trip also cuts waste heat (+15 %
  removals vs Li-ion at 20 s bursts). LTO sits between (0.6–2 yr).
- Burst length: ~20–32 s is best. 20 s: +26 % removals over 10 s; the longest
  feasible burst with the 25 kg store (24.7 s) does worse than 20 s (longer waits
  for a near-full store, fewer engagements); with a 75 kg store, 32 s bursts give
  +40 %, but 48 s bursts fall back. Firing at 20 s readiness with a 25 s cap
  does not help.
- Radiator 60 m² ≈ 3× removals in every storage option (as before); the most
  removals (818 in 21 d) come from 60 m² + 75 kg PCM + 54 s bursts, but its LTO
  pack lasts 0.5 yr.
- Best per removal here: 60 m², supercap 2 kWh, 20 s bursts — ~$10k per
  credited removal on these picks, 4 sats × 2.9 t. That figure uses the 21-day
  rate × 5 yr; the 1-yr run showed the rate peaking and then falling as the light
  tail is used up, so the 5-yr removal total and the $/removal are optimistic.
  The next step is a multi-year run of this design (checkpoint + resume).

## Orbit knowledge, target response and risk-weighted benefit (2026-11)
New models, all off by default (baseline events unchanged):
`[knowledge]` (talos.knowledge: scheduler state-estimate error, per-burst
pointing bias), `[ablation]` `cm_scale_*` by ORDEM material class,
`attitude_model`, `fragment_*` (fragmentation by ablation), and talos.risk
(expected collisions with active satellites from shell residence; the
600–1000 km band is a shell; ablation fragments' natural-path risk counts
against the laser). 14 days, ORDEM + SBM baseline (`python
scripts/compare_response.py 14` → runs/compare_response, _frag):
| variant | eng/hr/sat | Δv/eng | removed | fragments | net change/yr | active collisions avoided (14 d) |
|---|---|---|---|---|---|---|
| base | 9.97 | 5.92 | 90 | — | −2.3k | 0.138 |
| state error 1 km / 1 m/s (angles-only class) | 9.98 | 5.86 | 86 | — | −2.2k | 0.120 |
| state error 10 km / 10 m/s | 9.96 | 5.85 | 92 | — | −2.4k | 0.125 |
| pointing bias 0.3 µrad / axis | 9.79 | 3.30 | 26 | — | −0.7k | 0.090 |
| pointing bias 1 µrad / axis | 3.82 | 1.48 | 1 | — | −0.03k | 0.018 |
| NaK immune (Cm 0) | 9.95 | 5.76 | 82 | — | −2.1k | 0.122 |
| Cm by material: LD 1.5, MD 0.7, HD 0.4 | 9.96 | 4.28 | 37 | — | −1.0k | 0.134 |
| attitude per burst (slow tumble) | 8.77 | 7.31 | 129 | — | −3.4k | 0.139 |
| fragmentation 1 % per burst | 9.96 | 5.79 | 85 | 99 | +0.4k | 0.035 |
| fragmentation 10 % per burst | 9.96 | 6.31 | 131 | 933 | +20.9k | −0.793 |
- Orbit knowledge does not limit the greedy strategy: even 10 km / 10 m/s state
  errors leave engagements and removals within noise (each pass re-detects and
  re-tracks its target; the burst itself is closed-loop). An earlier note here
  suggested the 13-visit removal strategy might need a catalogue; it does not,
  for this scheduler (the commitment planner would).
- Pointing is critical: the spot is ~12 cm at 200 km, so a 0.3 µrad bias
  (9 cm at 300 km) on top of the 0.3 µrad jitter cuts removals 71 %, and 1 µrad
  stops the system. Sub-0.3 µrad absolute pointing (centroiding, point-ahead,
  boresight alignment) is a hard requirement.
- Removals are nonlinear in Cm: 30 % less coupling on the dominant aluminium
  class removes 59 % fewer objects (the light tail just clears the 30-day
  criterion). NaK immunity costs little (−9 %): the dense droplets were rarely
  removed anyway.
- Slow tumble (per-burst attitude) adds +43 % removals but 12 % of bursts push
  the wrong way (not "useful"): sideways kicks add up within a burst, so the
  variance per burst is much larger and luck finishes objects sooner.
- Fragmentation decides the sign: at 1 % per burst the fragments (~1 per
  removal) already take 75 % of the collision benefit and the population grows
  (+0.4k/yr); at 10 % the laser is a net debris source (+21k objects/yr) and
  raises collision risk. Fragmentation probability per burst needs a sourced
  bound below ~1 %.
- Collisions avoided barely follow removals (the "materials" case removes 59 %
  fewer but avoids nearly as many collisions): most of the benefit is lowering
  long-lived band objects, not finishing them, and lasered objects cross the
  Starlink shell faster. Starlink-shell residence dominates the natural risk of
  engaged objects (3.8 of 5.3 expected collisions over their lifetimes).

## Gate 4 vs ORDEM: encounter flux along the platform orbit (2026-11)
The rate at which objects enter a sphere of radius R around the platform is
flux × πR² (any direction distribution, density uniform on the scale of R), so
the sim's raw entry rate within R, divided by πR², should reproduce ORDEM's
1–10 cm flux on that orbit. `python scripts/validate_encounters_ordem.py --alt
800` (12 h, full ORDEM population, no lifetime filter):
| R | 25 km | 50 km | 100 km | 150 km | 200 km | 250 km | R → 0 fit | fitted table |
|---|---|---|---|---|---|---|---|---|
| 800 km SSO, sim / ORDEM | 0.85 ± 0.06 | 0.92 ± 0.03 | 0.92 ± 0.01 | 0.89 | 0.82 | 0.72 | **0.95 ± 0.01** | 0.96 |
| 900 km, i 98.6° | 0.88 ± 0.05 | 0.87 ± 0.03 | 0.82 ± 0.01 | 0.73 | 0.63 | — | **0.87 ± 0.01** | 0.99 |
| 700 km (6 h) | — | 1.09 ± 0.05 | 1.04 ± 0.03 | — | — | — | — | 0.96 |
- At 800 km the chain (fit → sampling → propagation → screening) reproduces
  ORDEM to 5 %, all of it from the inclination fit. The fall with R is the
  sphere averaging density over ±R of altitude (600–1000 km band edges at
  large R); the R → 0 value fits ratio = r₀(1 + cR²) for R ≤ 200 km.
- At 900 km, at the top edge of the 830–910 km plateau, the sim is 13 % low
  against a fit that matches ORDEM to 1 %. Sphere averaging over the falling
  side (245 and 214 objects/km in the cells above vs 290) explains ~4–7 % at
  R = 25–50 km; the rest (~2σ) is open. Eccentricity is not it (e = 0 for all
  samples changes < 4 %). The sampler reproduces the table per cell and
  inclination bin.
- Forward kernel vs simulated geometry (`--kernel-check`: thick circular shells
  of one inclination bin, 900 km, R 100 km): sim / kernel 1.02 (isotropic),
  0.96 (30–35°), 1.01 (60–65°), 1.00 (80–85°), 1.01 (95–100°), all ± 0.02. The
  fit's inclination kernel is right.
- Phipps: ORDEM's own 800 km flux (1.33e-4 /m²/yr) is 50/min within 250 km
  for uniform density, and the sim's band-limited population gives 36/min.
  The ~2× gap to Phipps's 19/min is in the population density, not in the
  screening geometry.

## Gate 5: lifetime vs NASA DAS 3.2.7 (2026-11)
`talos.das` calls DAS's `Fortran.dll` LIFEDWELL (PROP3D: J2–J4, luni-solar,
SRP, Jacchia-77, Cd 2.2, ODPO F10.7 forecast) via ctypes; install at
`%LOCALAPPDATA%\NASA\DAS3.2.7` or `TALOS_DAS_HOME`. Same Cd and A/m definition
as TALOS (A/m = 1/μ). `python scripts/validate_lifetime_das.py` (7 s) →
runs/das_lifetime; `tests/test_das.py` skips without DAS.
- Circular SSO, 400–1000 km × A/m 0.05–1, starts 2026.97 and 2031.0 (121
  cases where both re-enter within 100 yr): TALOS/DAS geo-mean 0.92, log-sd
  0.34; for DAS lifetimes < 25 yr, geo-mean 0.88, range 0.38–2.00.
- Structure: (a) a diagonal band at 0.4–0.5 where DAS lifetimes jump (objects
  that ride out the next solar minimum in DAS but not in TALOS): the two solar
  cycle forecasts differ in phase/depth; (b) from 2031 (solar min) TALOS is
  ~0.5× below 500 km, i.e. TALOS's minimum-activity density is too high (f107_min
  70 is a pick); (c) above ~750 km TALOS is 1.1–1.5× (longer).
- Lowered orbits (apogee 800/950 km, perigee 250–750 km, A/m 0.2/0.5, 2026.97):
  TALOS/DAS 0.79–0.87 at perigees 250–450 km (lifetimes 3–390 d), i.e. around
  the 30-day removal criterion TALOS is ~15 % short. That is ~7 km of perigee
  (H ≈ 45 km), ~2 m/s of the ~160 m/s needed: negligible for removal counts.
  Larger gaps (0.37–0.54) again sit at the cycle-timing step.
- Not calibrated: neither model is truth, and the cycle forecast is the main
  disagreement. Replacing the parametric cycle with the ODPO F10.7 table would
  close most of it (DAS's `solarflux.bin` format is undocumented).

## Sourcing pass 2 (2026-11-09)
The four inputs that swing the result most, checked against sources.

**η₁ and the 45° thrust angle: the old model counted direction twice.** Phipps et al.
2012 (arXiv:1110.3835, Eq. 6) write Δv‖ = η_c·Cm·φ/μ for the *desired* velocity change,
with η_c "for the combined effects of improper thrust direction, target shape, target
tumbling, etc.", and "take η_c = 0.3 after Liedahl". Liedahl et al. 2010
(arXiv:1004.0390) get that from outcomes: "the average perigee change for a plate is
approximately 1/3 that found for a sphere". Rogers & Lee (arXiv:2512.14682, Eq. 2–4)
push along the line of sight with no separate angle. TALOS used η₁ = 0.3 as the impulse
magnitude and then tilted it by ~45°, costing another cos 45° ≈ 0.71. Now
`eta1_basis = "along_beam"` (default): magnitude η₁/cos(mean angle), so the mean
along-beam part is η₁ and the off-axis draw only adds sideways scatter. `"magnitude"`
reproduces earlier runs. **Every result above this section used the old basis and
understates Δv per engagement by ~29 % (×1.41 correction).**

**Supercapacitor cycle life was 2× optimistic.** Maxwell 160 V modules (BMOD0006, 160V
10F) are rated "up to 500,000 duty cycles" (V_R → V_R/2, 25 °C; end of life −20 %
capacitance, +100 % ESR) at 3.8–5.4 Wh/kg module level. The preset had 1M cycles at
5 Wh/kg; now 500k at 4 Wh/kg. The only space-qualified supercapacitor found is a Nesscap
10 F cell (ESA EPPL II), with no cycle count published. At the design's ~17 bursts/hr/sat
(~150k/yr) 500k cycles is ~3.3 yr, so storage, not the 5-yr design life, sets mission
life again. Li-ion's preset shape (24–39k cycles at 30–40 % DoD) agrees with the ~30k-cycle
LEO requirement met by cycling below 40 % DoD; LTO's 20k matches a vendor ≥ 25k (to 80 %,
DoD unstated, not space-qualified).

**Cm by material: direction sourced, size not.** No ps or UV measurement on a debris
material was found. What exists is ns / IR:
| source | regime | result |
|---|---|---|
| Keil et al. 2023, *Aerospace* 10, 947 | 10 ns, 1064 nm, 60 J, free fall | Cm 7–40 µN·s/J, thermal coupling 2–12.5 %, both by material, fluence, roughness |
| Phipps et al. 2012 | survey | Al 50–320 µN·s/J by surface condition; organics up to 300 |
| SPIE 13699 (2025) | 1064 / 532 / 355 nm, ns | CFRP threshold ~¼ of metal, Cm ~80 µN·s/J above ~2 J/cm², several times metal |
| NASA OTPS 2024 TIM | review | rep-rated pulses "may see up to 3x reduced coupling" vs single-shot data |
So low-density (CFRP, plastics) > aluminium ≥ high density, and the 33 Hz rep rate may cut
all of them by up to 3×; the 25–100 N/MW sweep range covers 2× either way, not 3× down.
The `cm_scale_*` picks stay at 1; the "materials" variant above (LD 1.5 / MD 0.7 /
HD 0.4) has the sourced ordering.

**Fragmentation probability: no published value.** NASA OTPS 2024 lists breakup as an
open "make or break" item ("debris may be small enough for punch through or
fragmentation"); JAXA's 2023 ISS experiment page says melt "can be deformed and/or
fragmented by a second shot" and that no systematic studies exist. Scharring & Kästel 2023
keep a ground laser below 10 Hz "to avoid fragment meltdown". `fragment_prob_per_burst`
stays a guess (0). A derived bound instead: a burst melts an aluminium target if the
absorbed heat f_th·Σφ exceeds μ·h_melt, with h_melt = c_p·ΔT + L ≈ 0.9 kJ/kg/K × 643 K +
397 kJ/kg ≈ 976 kJ/kg and no cooling (bare Al, ε ≈ 0.1, radiates ~4.6 kW/m² two-sided at
800 K against ~13–90 kW/m² absorbed). 33 Hz, f_th = 2–12.5 % (Keil, ns IR):
| burst | φ per pulse | absorbed per burst | melts μ below |
|---|---|---|---|
| 10 s | 10 kJ/m² (325 km) | 66–410 kJ/m² | 0.07–0.42 kg/m² |
| 20 s | 10 kJ/m² (325 km) | 130–830 kJ/m² | 0.14–0.85 kg/m² |
| 20 s | 17 kJ/m² (250 km) | 220–1400 kJ/m² | 0.23–1.4 kg/m² |
Re-engagements are days apart, so heat does not carry over between bursts. At the high
end, a 20 s burst can melt the μ < 1 objects that make up 68 % of removals; at the low end
only the lightest few percent. 20 s bursts double the exposure relative to 10 s. ps pulses
should leave less heat than 10 ns ones, but no measurement was found. This makes
thermal coupling at 355 nm / 100 ps the single most useful measurement to ask for; until
then, cap absorbed energy per burst (burst length × fluence) or treat melt-prone bursts as
fragmentation risk.

**Effect, 7 days** (ORDEM + SBM, Dec 21 start, 4 sats; runs/sourcing_7d; 7-day removals
are early on the ramp, so compare rows only; $/removal = 7-day rate × mission life):
| run | eng/hr/sat | Δv/eng (m/s) | removed (7 d) | storage life (yr) | cost/removal |
|---|---|---|---|---|---|
| baseline, old basis (`magnitude`) | 9.98 | 6.05 | 30 | 0.46 | $545k |
| baseline, `along_beam` | 9.94 | 8.51 (×1.41) | 58 (×1.9) | 0.46 | $283k |
| `config/design.toml` (60 m², supercap 500k cycles, 20 s) | 17.16 | 14.02 | 335 | 3.15 (storage) | $12k |
Removals respond more than Δv (×1.9 for ×1.41) because each object needs fewer visits.
The design's mission life is now storage-limited at 3.2 yr (was 5 yr at 1M cycles).

## Playback viewer (2026-11-09)
`scripts/render_run.py` writes a self-contained CesiumJS page for a run window: every
object as a GPU point (~70k), the platforms, a beam in the platform's colour during each
burst, the current targets' orbits, and each engaged object's new orbit after every burst
(amber = engaged, red = removed). HUD: clock, speed (1–3600×), per-platform state (firing
target / time to next burst), window tallies, recent bursts; click a point for its size,
μ, material, orbit and burst history; click a platform to follow it. Toggles: inertial
frame, target orbits, colour untouched debris by μ, thin the untouched debris.
- Simulates the scenario to the window end (1 day of `design.toml`: 13 s), or renders an
  existing run's events (`--events`). `run_baseline.py --render-days 1 [--render-at
  first|last|both]` renders automatically after a run into `<out>/render/`.
- Exact replay (`talos.render.Replay`): the initial population is stepped through the
  same `PopulationDynamics.step` calls the sim made (daily drag decay and re-epoch, natural
  decays, new objects; the dynamics have their own generator), with every logged burst's
  post-burst elements (`a_after_m` … `M_after_rad` at `t_elements_s`) written in. It
  reproduces the sim's checkpointed population bit for bit (tests/test_render.py); a
  checkpoint before the window only saves replay time (~0.12 s per sim day).
- Decay steps inside the window: the viewer runs each step itself (angles by J2, a and e
  linear at each object's drag rates from `lifetime.decay_rates`), and the exporter, which
  mirrors the viewer exactly (Node test), sends a correction only where that ends > 0.5 km
  from the sim (`SNAP_TOL_KM`). Without the local step every object needs a daily update
  (untouched objects drift a median 3 km after one day, ~150 km after ten). 30-day design
  run as one page: 167k corrections (vs 1.93M), 23 MB, built in 19 s; drawn target vs sim
  median 49 m, p95 0.38 km, max 3.2 km over 51k bursts. Pages grow ~0.75 MB per sim day,
  so keep windows to a few months. Event logs from before
  2026-11-09 lack the post-burst elements and are rejected; runs with ablation
  fragmentation are not supported (fragments come from the engagement generator).
- Browser propagation (`src/talos/viewer/orbit.js`) is the J2 mean-element model of
  `talos.propagator` without drag; tests/test_render.py checks it against Python under Node
  (< 1 mm). The exporter reports the gap between the drawn target and the sim's at
  mid-burst (design: day 0 median 20 m, max 0.84 km; day 1 median 20 m, max 2 km; the
  residual is drag within a day, not drawn). Without the replay a day-1 window was off by
  up to 106 km.
- Cesium 1.119.0 from jsDelivr (needs internet). The globe's workers do not run from a
  `file://` page: use `--serve`, or `python -m http.server 8765 -d <folder>`; the page says so.
- Headless Chrome (SwiftShader, no GPU) with the globe: ~10.5 fps with all 70k points; a
  real GPU should be well above that. The untouched-debris thinning option is there if a
  machine struggles.
- Not drawn: drag within a day (between decay steps); the sim's own propagation also
  holds the phase fixed between steps.

## Next step
Multi-year run of `config/design.toml` with the corrected η₁ basis (overnight) after a source-rate check (where the removal rate settles); Latin-hypercube bands on the design including fragmentation 0–1 %, Cm ÷3 to ×2, thermal coupling; thermal coupling at 355 nm / 100 ps (melt bound); absolute pointing budget; detection & tracking refinements; solar-cycle forecast vs ODPO; MASTER cross-check.
