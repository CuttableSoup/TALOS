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
  `python scripts/compare_scores.py DAYS`.
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
| Battery | `ResourceConfig` | 2 kWh, 85 % round trip (losses → heat) | capacity guess; cycle life NOT modelled |
| Bus | `bus_power_w` | 1.5 kW, dissipated into the thermal node | guess |
| Field of regard | `ATPConfig.field_of_regard_half_angle_rad` | unrestricted | guess |
| Epoch / LTAN | `[run] epoch`, `[constellation] ltan_hours` | 2026-12-21, 06:00 | pick: worst-case eclipse (~17 min/orbit at 800 km) |
| Sun model | `talos/sun.py`, `[sun]` | Almanac ephemeris, conical shadow, 2-axis array, no eclipse firing | physics; pointing + eclipse policy are picks |
| Target lighting | `[atp] require_target_sunlit` | true (passive optical tracking) | pick |
| Sun exclusion | `[atp] sun_exclusion_deg` | 30° | pick (30–45 typical, unsourced) |
| Detection model | `[atp] detection_model` | passive_optical: Lambertian sphere magnitude, acquire ≤ 16 mag for 10 s, track ≤ 18 mag, +3 mag on sunlit-Earth background | all picks; research needed |
| Debris albedo | `[population] albedo_median/log_sigma` | 0.1, lognormal σ=0.5 | pick (debris ~0.1–0.2) |
| Atmosphere | `[atmosphere]` | NRLMSIS 2.1 global mean (Ap 15) vs F10.7; cycle min 70 / max 155, cycle-25 timing | table derived from MSIS; forecast is a pick (use NOAA SWPC) |
| Background source | `[dynamics] background_frac_per_yr` | 2 % of n_real per year | pick |
| Breakups | `breakup_rate_per_yr`, `breakup_fragments`, `breakup_dv_sigma_m_s` | 0.5/yr × 2000 fragments, σ 50 m/s per axis | pick (NASA SBM gives ~100 m/s for A/m ~1) |
| Radiator absorptivity | `radiator_absorptivity` | 0.20 (albedo load) | 0.16 BOL sourced; EOL pick |
| Radiator turndown | `radiator_turndown_min` | 10 % (louvers) before heaters | pick |

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
   removals over the mission, not per year.
5. **Mass & cost model** — radiator (~10 kg/m²), PCM, arrays, battery, laser →
   launch cost → cost per removal.

Worth adding:
6. **Risk transfer & safety** — time lowered debris spends crossing ISS /
   Starlink shells; keep-out for active satellites.
7. **Uncertainty propagation** — Latin-hypercube over η₂, Cm, wall-plug, η₁,
   sun exclusion, etc., for confidence bands.
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
  give an optional μ = (2/3)ρd sphere model (`areal_density_model = "sphere"`),
  default stays fixed μ.
- Licence: ORDEM is NASA SUA software; tables are derived data and stay out of git.

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

## Next step
Detection & tracking refinements; solar-cycle forecast vs ODPO; MASTER cross-check.
