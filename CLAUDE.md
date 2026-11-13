# TALOS: Targeted Ablation Laser Orbital Sweeper

Monte Carlo engagement simulator for a space-based pulsed UV laser that removes 1–10 cm
debris from the 600–1000 km band. It answers: useful engagements/hr, Δv per engagement,
duty cycle, objects and debris-years removed, and cost per debris-year, against laser
power, sat count, orbit and hardware choices.

Dated results and run logs are not kept in this file. They are in git history (`README.md`
up to 2026-11-12) and under `runs/` (gitignored).

## Baseline point design (L'ADROIT class)
- Dawn-dusk SSO (LTAN 06:00), 700/800/900 km staggered; 4–8 sats
- 355 nm, ~100 ps pulses, 1.5 m mirror, M² ≈ 2, ~33 Hz, 20–30 kW burst optical
- Burst / cooldown operation; duty cycle is an output of the power / thermal state machine
- Engagement range 100–500 km; detection ~800 km; slew limit 2°/s
- Target: 1–10 cm debris, 600–1000 km (75.9k objects, ORDEM 3.2, 2026)

## Key equations (Rogers & Lee, arXiv:2512.14682)
- Fluence: φ = η₂(E/π)·[2D/(M²·a·λ·R)]², a ≈ 1.27
- Δv per pulse: Δv = η₁·Cm·φ/μ along the line of sight. η₁ is the useful along-beam
  efficiency; direction, shape and tumble are already in it (Phipps 2012 Eq. 6), so the
  impulse magnitude is η₁/cos(mean thrust angle) (`eta1_basis = "along_beam"`)
- Fire only if φ ≥ ablation threshold (4.5 kJ/m²); Cm peaks at 8.5 kJ/m², then ∝ φ^-1/4
- Removed = lifetime < 1 month or perigee < ~200 km
- Melt limit: thermal_coupling·Σφ/μ ≥ melt_margin·melt_enthalpy (976 kJ/kg, aluminium)

## Setup and commands
- `pip install -e .[dev]` (+ `numba`, `pyarrow`). Without the install, set `PYTHONPATH=src`.
- Tests: `pytest` (~1 min). `TALOS_NUMBA=0` runs the numpy fallback. DAS tests skip
  without DAS.
- Single run: `python scripts/run_baseline.py DAYS --out runs/x [--config path.toml]
  [--set section.key=value ...] [--threads N] [--resume ckpt.pkl [--fresh-platforms]]
  [--render-days 1]`
- In code: `talos.config.load_scenario(path, overrides)` → `run_scenario(sc)`;
  sweeps: `RunConfig(name, config=None, overrides={...})`, `run_sweep(cfgs, "runs/x")`.
- Speed: a 1-yr run is ~1 h; ~10 s per sim day. Runs do not scale past ~8 threads, so
  sweeps use `n_jobs = 4, threads_per_job = 8`. On Windows, call `run_sweep` under
  `if __name__ == "__main__":`.
- Checkpoints: runs with an out_dir write `checkpoints/<name>.pkl` every
  `run.checkpoint_days`. `--resume` continues or extends a run and reproduces the
  uninterrupted run exactly. `--fresh-platforms` keeps only the population and starts new
  hardware over it (tallies count from the restart).
- Run at full density (N = n_real). Engagement and removal counts are not weight-scaled,
  and subsampling is invalid for a target-saturated platform.

## Layout
`src/talos/`:
| module | role |
|---|---|
| `population`, `ordem`, `debris` | debris sampler (ORDEM 3.2 table or `toy`), μ models (`sbm`, `hanada`, `hanada_material`, `sphere`, `fixed`), albedo |
| `elements`, `propagator`, `atmosphere`, `srp`, `sun` | mean-element J2 + drag propagation, NRLMSIS table vs F10.7, SRP, ephemeris / eclipse |
| `platform` | laser, ATP and resource dataclasses; `STORAGE_PRESETS` |
| `screening` | coarse sieve → 1 s fine gates (range, slew, sun, detection) → windows |
| `engagement` | fluence, coupling, burst integration and prediction, melt gate |
| `scheduler`, `planner`, `campaign` | greedy candidate scoring; commitment planner (off); large-debris focus (off) |
| `resources` | battery + PCM / radiator state machine gating bursts |
| `lifetime`, `dynamics` | decay and removal test; daily population step, sources, breakups |
| `simulation` | the run loop, events, checkpoints |
| `knowledge`, `risk`, `survival`, `cost`, `metrics` | orbit-knowledge error, shell residence / collisions, platform impacts, mass / life / cost, summaries |
| `sweep`, `config` | parallel sweep driver, Latin hypercube; TOML scenarios |
| `das`, `oscar` | lifetime cross-checks against NASA DAS and ESA DRAMA/OSCAR |
| `render`, `viewer/` | exact population replay and the CesiumJS playback page |
| `_kernels` | numba kernels (screening, prediction, resources, lifetime) |

- `config/`: `baseline.toml` (reference), `design.toml` (60 m² radiator, 2 kWh supercap,
  20 s bursts), `design_mc.toml` (d027 from the design search). A TOML may start with
  `base = "other.toml"` and list only the keys that differ.
- `scripts/`: `run_baseline`, `compare_*` (one lever each), `mc_design` (design search over
  early and day-720 skies), `large_debris_test`, `run_uncertainty`, `render_run`,
  `validate_*` (gates), `build_*` (data tables).
- `runs/`, `research/`, ORDEM / F10.7 / flux tables: gitignored.

## Conventions
- Scenario keys carry units in suffixes (`_km`, `_kw`, `_kwh`, `_deg`, `_days`, `_c`, …);
  the loader converts to SI. Unknown keys are an error.
- Every scenario value has a basis tag: [sourced], [derived], [pick], [guess]. Keep the
  inputs table below in step with the code.
- New models go in off by default, so earlier runs stay comparable. A new scenario key
  needs its old-behaviour value in `config._LEGACY`, or existing checkpoints stop resuming.
- Numba kernels must match the numpy path (tests compare them); the resource state machine
  is bit-identical to its Python fallback.
- Source files use CRLF line endings.
- No GPL code in core (dSGP4, NASA-breakup-model-cpp); poliastro is archived; MSIS is
  non-commercial, so only the derived density table is shipped. ORDEM and DAS are NASA SUA
  software: their tables stay out of git.
- Compare variants at equal run length and do not annualise short runs: removals ramp for
  months, then fall. Do not quote absolute dollars; cost coefficients are picks.
- The headline metric is cost per debris-year. Removal counts and collisions avoided swing
  with the μ model and the mission phase.

## External tools and data
- ORDEM 3.2: `%LOCALAPPDATA%\NASA\ORDEM 3.2` or `TALOS_ORDEM_HOME`; `ORDEM32.exe` must run
  from the install root. `scripts/build_ordem_table.py` (~40 min) writes
  `src/talos/data/ordem_<year>.npz`.
- DAS 3.2.7: `%LOCALAPPDATA%\NASA\DAS3.2.7` or `TALOS_DAS_HOME`; `talos.das` calls LIFEDWELL
  through ctypes.
- ESA MASTER 8.2.3 and DRAMA (OSCAR) are installed; the MASTER population cross-check is
  not done.
- Solar activity: ESA SOLMAG monthly F10.7 table (`scripts/build_f107_table.py`), default.
- Viewer: Cesium 1.119.0 from jsDelivr; serve the page over http (`--serve`), not `file://`.

## Inputs and their basis
| Parameter | Where | Value | Basis |
|---|---|---|---|
| η₂ (fluence efficiency) | `LaserConfig.eta2` | 0.40 (sweep 0.3–0.67) | **sourced**: backed out of L'ADROIT spot size + fluence |
| Fire threshold | `AblationParams.threshold_fluence` | 4.5 kJ/m² | **sourced** |
| φ_opt (Cm peak) | `AblationParams.phi_opt` | 8.5 kJ/m² | **sourced** |
| Cm model | `AblationParams.cm_model` | plasma: log ramp, then ∝ φ^-1/4 | **sourced** falloff; ramp shape is a guess |
| Cm peak | `AblationParams.cm` | 50 N/MW (sweep 25–100) | **sourced** range; aluminium 10–40 N/MW at ns (Keil 2023, SDC9 paper 307); no ps / UV point; rep-rated pulses may couple up to 3× less (NASA OTPS 2024) |
| Cm by material | `cm_scale_*` | 1 (off) | ordering sourced (low-density > aluminium ≥ high density), sizes not |
| η₁ | `eta1_mean/kappa`, `eta1_basis` | Beta, mean 0.3, κ = 10; `along_beam` | mean and basis **sourced** (Phipps 2012, η_c = 0.3 after Liedahl); spread a guess |
| Δv off-axis angle | `off_axis_*` | N(45°, 15°) clipped [0, 90°] | centre **sourced** (Scharring 2018: 45°, 40 % scatter); spread a guess |
| Fragmentation | `fragment_prob_per_burst` | 0 (off) | **guess**: no published probability |
| Target heating (melt gate) | `thermal_coupling`, `melt_enthalpy_j_kg`, `melt_margin`, `melt_policy` | 0 (off); 976 kJ/kg; 1; `fragment` | coupling **unsourced at 355 nm / 100 ps**: bracket 0.07–0.25 (Scharring 2018 simulation at 1064 nm: ≤ 7 % ultrashort, ≥ 20 % near onset for 100 ps and longer; measured 15–25 % typical); 0.02–0.125 at 10 ns IR (Keil 2023); enthalpy derived |
| Debris count in band | `population.kind = "ordem"` | 75.9k, 1–10 cm, 600–1000 km | **sourced** (ORDEM 3.2); inclination mix is a fit to ORDEM flux spectra |
| Areal density μ | `areal_density_model` | `sbm` (median 4.5 kg/m²) | **sourced** (NASA SBM); `hanada` variants give lighter fragments (median 2.2–3.0) |
| Debris albedo | `albedo_median/log_sigma`, `albedo_model` | 0.1, lognormal σ 0.5 (`fixed`) | pick; `proof` (fragments 0.2, NaK 0.85) **sourced**, off |
| Detection | `[atp] detection_model` | passive optical: acquire ≤ 16 mag for 10 s, track ≤ 18, +3 mag on sunlit Earth | picks |
| Target lighting / sun exclusion | `require_target_sunlit`, `sun_exclusion_deg` | true; 30° | picks |
| Pointing jitter | `ATPConfig.jitter_rad` | ~0.3 µrad | pick |
| Settle / retarget | `ATPConfig.settle_s` | 5 s, not overlapped | guess |
| Min burst | `ResourceConfig.min_burst_s` | 5 s | guess |
| Wall-plug (elec → 355 nm) | `wallplug_eff` | 0.10 (0.04–0.17) | **sourced** chain; ATLAS flight 8.2 % at 532 nm |
| Radiator net flux | `ResourceState.q_net(T)` | ε 0.9, F_E 0.22, fin 0.85, 2-sided → 445 W/m² at 280 K | **sourced** inputs, derived relation |
| Radiator area | `radiator_area_m2` | 20 m² (baseline), 60 m² (design) | design choice, dominant |
| Store → radiator ΔT | `radiator_dT_k` | 8 K | **sourced** (ATLAS, AMS-02) |
| PCM store | `pcm_melt_k`, `pcm_mass_kg`, `pcm_specific_j_kg` | 15 °C, 25 kg × 120 kJ/kg | density **sourced**; melt point unsourced |
| Sensible heat capacity / band | `heat_capacity_j_k`, `t_min_k`, `t_max_k` | 0.5 MJ/K; melt −5 K … +2 K | guesses |
| Radiator absorptivity / turndown | `radiator_absorptivity`, `radiator_turndown_min` | 0.20; 10 % | 0.16 BOL sourced; rest picks |
| Storage | `STORAGE_PRESETS` | Li-ion 5000 cycles at 100 % DoD, k 1.7, 120 Wh/kg; LTO 20k, k 1.2, 70; supercap 5e5, k 0, 4 Wh/kg, 10-yr calendar; `supercap_bosc` 4.5e6, 2.56 Wh/kg | supercap **sourced** (Maxwell datasheet; BOSC life test 4.5 M cycles at 45 °C); BOSC voltage derating and Wh/kg an assumption; the rest picks |
| Burst readiness | `ready_burst_s` | = burst_s, capped at 95 % of the feasible burst | policy |
| Bus | `bus_power_w` | 1.5 kW | guess |
| Epoch / LTAN | `[run] epoch`, `ltan_hours` | 2026-12-21, 06:00 | pick: worst-case eclipse |
| Atmosphere | `[atmosphere]` | NRLMSIS 2.1 global mean vs F10.7, ESA SOLMAG forecast | derived table; F10.7 **sourced** |
| Background source | `[dynamics] source_mode` | `replace_decays` (un-lasered band in steady state) | assumption |
| Breakups | `breakup_*` | 0.5/yr × 2000 fragments, σ 50 m/s | pick |
| Debris SRP | `[dynamics] srp`, `srp_cr` | on; Cr 1.2 | physics; Cr pick |
| Protected shells | `[risk]` | crewed 370–430 km, Starlink 530–575 km, band 600–1000 km | picks |
| Mass / cost | `[cost]` | radiator 10 kg/m², array 100 W/kg, laser 20 kg/kW, beam director 300 kg, bus 30 %; $50k/kg, $3k/kg launch, $2M/sat-yr; 5 yr, 1e10 shots | picks except radiator; laser mass has no flight anchor |
| Station keeping | `[cost] drag_*`, `sk_*` | drag make-up costed, Hall Isp 1500 s | picks; negligible (< 1 m/s) |
| Platform impacts | `[survival]` | lethal ≥ 1 cm on 10 m² bus; radiator puncture ≥ 1 mm on 10 % of area, zenith / nadir faces | 1 cm **sourced** (weak); rest picks |
| Uncertainty ranges | `talos.sweep.UNCERTAIN_INPUTS` | η₂, Cm, η₁, thrust angle, wall-plug, sun exclusion, settle | sourced ranges where the table has one, else picks |

## Validation gates
| Gate | Status |
|---|---|
| 1. Spot size / range vs L'ADROIT | passes, but not independent (η₂ is backed out of the same numbers) |
| 2. Pieters & Noomen (SDC8 paper 43) single-pass Δv | Table 4 (altitude offsets) within 15 %, mean ratio 1.01, with their inputs. Head-on case only bracketed: the paper gives no miss distance. `tests/test_gate2_pieters.py` |
| 3. Slew limit for a head-on pass | 63 km confirmed by the same paper; miss distance unstated |
| 4. Encounter rate | sim / ORDEM flux 0.95 at 800 km, 0.87 at 900 km (open, ~2σ). `scripts/validate_encounters_ordem.py` |
| 5. Lifetimes | TALOS/DAS geo-mean 0.92 (log-sd 0.34), TALOS/OSCAR 0.69–0.96; solar-cycle forecast is the main difference. orekit / STELA and the MASTER population cross-check are not done |

## What the runs have established
Use these when choosing what to build or sweep; the numbers behind them are in git history.
- **Light-tail harvest.** Removals peak in month 3 and fall ~25× by month 24 while the
  burst rate holds. Objects with μ ≥ 5 (47 % of the population) are ~1 % removed in 2 years.
  A 30-day rate is not an annual rate.
- **Scheduling is not the lever.** The commitment planner loses to greedy
  `removal_progress`, and score variants give equal removals. Orbit-knowledge errors up to
  10 km / 10 m/s do not matter to the greedy scheduler.
- **The platform is heat-limited.** Removals scale ~linearly with radiator area; wall-plug
  efficiency and average power set the rate. PCM mass only sets the longest burst.
  Bursts of ~20–35 s beat 10 s.
- **Storage life sets mission life and cost.** Li-ion wears out in months. With a
  supercapacitor bank the cycle rating decides the design ranking: at 0.5 M cycles storage
  limits life to 1.4–2.3 yr and d027 is cheapest; at 4.5 M every design reaches 5 yr and
  the highest-rate design wins.
- **Pointing is a hard requirement.** A 0.3 µrad bias on top of 0.3 µrad jitter cuts
  removals ~70 %; 1 µrad stops the system. With the design jitter the target receives at
  most ~0.3 of the commanded fluence.
- **Cm on aluminium is the largest single sensitivity.** Halving it raises cost per
  debris-year 35 %; removals are strongly nonlinear in Cm.
- **Fragmentation and melting decide the sign.** At 1 % fragmentation per burst the
  population grows. With the melt gate firing through, 7 % thermal coupling cancels the
  collision-risk benefit and ~15 % makes the laser a net debris source. Cost per
  debris-year does not see this: it does not charge for fragments.
- **Capping bursts at the melt limit works only with a jitter allowance.** A cap on
  commanded fluence costs 78–99 % of early removals. With `melt_margin = 3` it costs
  0 / 5 / 12 % of debris-years at 7 / 15 / 25 % coupling (7-day runs, one seed).
- **The μ model moves removal counts (×1.2–2.6), not cost per debris-year (±1 %).**
- **Large debris needs concentrated fire and pulse energy.** Greedy removes 0–1 objects
  ≥ 9 cm in 30 days on the depleted sky; an exclusive focus campaign removes 9–29.
- **Radiator orientation matters for survival.** It must sit edge-on to the debris stream
  (zenith / nadir faces) with tubes armoured against ≥ 2 mm particles; this conflicts with
  the thermal model's Earth view and is not reconciled.

## Known limitations
- Fixed platform orbits; no constellation coordination or hand-off between platforms.
- Detection is a hard magnitude threshold; settle time is a fixed 5 s (no slew dynamics).
- Melt gate: no cooling within a burst, no heat carried between bursts, aluminium only,
  no incidence angle; the cap margin is hand-set, not derived from the jitter.
- Cost per debris-year does not charge for ablation fragments. Debris-year lifetimes are
  drag-only (no SRP) and depend on the 200-yr cap.
- Not modelled: M² / thermal-lens transients at burst start, THG crystal detuning, array
  degradation, UV optics damage, radiator self-view, meteoroid flux.
- ORDEM population: circular orbits, uncertainty files unused, power-law size inside each
  half-decade. Background source and breakup rates are picks.
- Playback does not support runs that shed ablation fragments.
- Results are one seed per variant unless stated.

## Next steps
- Melt gate: jitter-aware cap (expected delivered fluence); `thermal_coupling` 0.07–0.25
  and the policy as a band in the Latin hypercube; a fragment charge in cost per
  debris-year; 30-day runs and seeds.
- Supercap bank: confirm the BOSC test voltage and depth from the paper itself, then set
  the specific energy for the 4.5 M rating.
- Seeds on the leading designs (control, d027, d002, d007, d012).
- Whole-mission run of d027 (`config/design_mc.toml`, 2 years, checkpointed; overnight),
  scored per debris-year; decide whether to size the mission to the light inventory (~2 yr).
- Carry `cm_scale_md` 0.5–1, fragmentation 0–1 % and the μ model in the uncertainty bands.
  Measurements to ask for: aluminium Cm and thermal coupling at 355 nm / 100 ps.
- Storage sized for life with average power; Δv per burst on μ ≥ 2 targets.
- Platform placement / reconfiguration for the late mission (retarget altitude or RAAN at
  a checkpoint before any optimiser).
- Radiator orientation and tube armour reconciled with the thermal model; meteoroid flux.
- Debris lifetimes with SRP; absolute pointing budget; PROOF albedo on by default.
- Validation: orekit / STELA propagation check, MASTER population cross-check, ODPO vs
  ESA F10.7.
- Repo: no LICENSE, CI or user-facing README yet.
- Admin: Space-Track.org registration; consider contacting WVU (Hang Woon Lee lab).
