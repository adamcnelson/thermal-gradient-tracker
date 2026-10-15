# r_analysis_RGB_Thermal — R/ggplot2 analysis of the RGB+thermal landmark pipeline

A separate, R-based plotting/modeling layer on top of the v7 RGB+thermal
landmark pipeline's outputs (`../src/landmarks/`, `../scripts/stage7_real_run.py`).
This workstream only **reads** those outputs — it doesn't regenerate or
modify anything under `landmark_outputs/` at the repo root.

Full context and rationale: `../project_brief_v8.md` §4-5. This directory is
the new-pipeline analog of `../r_analysis/` (which analyzes the legacy,
pre-RGB pipeline's `mouse_surface_temp_mean` outcome only) — same
conventions, own `renv`, extended from one outcome to three
(`dorsal_mean_c`, `warm_spot_temp_c`, `tail_delta_t_c`).

## What it reads

One **data version** at a time (`R/data_versions.R`, picked with the `DATA_VERSION` env var).
Each version names exactly which SLURM results it reads: landmark masters, Stage 7 configs (entry
times) and tracking masters (activity).

| `DATA_VERSION` | SLURM results | Landmark masters |
|---|---|---|
| `retrack_2026-10` (default) | retrack_2026-10 + recover_2026-10 + recover_lanes_2026-10 tracking; 4 Stage 7 runs; pre-v2 tracker | `../landmark_outputs/` |
| `retrack_v2_2026-10` | `slurm/run_retrack_v2.sbatch` (job 24611205, 2026-10-15): all 85 lanes, tracker v2, tracking + Stage 7 in one run | `../thermalFeatures/retrack_v2_2026-10/landmark_join/` |

A new SLURM run gets a new version entry; never repoint an existing one. The landmark masters are
already fully metadata-joined by `../scripts/join_landmark_metadata.py` (project_brief_v8.md §3.2):

```
master_landmarks_with_metadata_bout.csv    # bout-level, one row per stationary bout
master_landmarks_with_metadata_frame.csv   # frame-level, one row per Stage 7 sample
                                            # (stationary bouts + §3.1's non-stationary pass)
```

Plus the metadata LUT already local to this repo,
`../metadata/LUT_CLEAN_July6.csv`, used to join `Craniotomy` status
(`Pre-craniotomy` / `Post`) into both tables — see `R/data.R::join_craniotomy()`.
That column isn't in the landmark master files (the Python join stage
deliberately doesn't select it, matching the legacy pipeline's own
convention); this workstream joins it in on the R side using
`(mouse_id, date)` parsed from `session`, the same pattern
`../r_analysis/R/data.R::join_craniotomy()` uses on `video_file` — ported,
not shared, since this workstream has its own renv and inputs.

**Outcome applicability is NOT a single shared QC gate** — see
`R/data.R`'s `OUTCOME_COLS` / `filter_for_outcome()` and
project_brief_v8.md §1: `dorsal_mean_c` (bout) / `mouse_surface_temp_mean_c`
(frame, same measurement, different column name — not a typo) only needs a
non-NA check; `warm_spot_temp_c` only needs non-NA too (confirmed non-NA
exactly iff `posture=="extended"`); `tail_delta_t_c` additionally needs
`qc_valid` (bout) / `qc_flag=="ok"` (frame) — non-NA alone is not enough for
that one (real, verified counterexamples exist). Always go through
`filter_for_outcome()` rather than re-deriving this per script.

**Floor preference is bout-level only** (`mean_floor_temp_c`,
`gradient_zone`) — the frame-level `floor_temp_mean_c` column is a
*different* measurement (the local floor annulus next to the tail sample
point, an input to tail-ΔT), not a preference signal, despite the similar
name.

## What it produces

Figures under `output/<DATA_VERSION>/` (git-ignored — regenerate via the scripts below
rather than committing PNGs). Each version folder has:
- `INPUTS.sha256`: checksums of every input. `run_all.sh` stops if the inputs no longer match.
- `SOURCES.md`: which SLURM jobs produced the inputs.

## How to run

From `r_analysis_RGB_Thermal/` (this directory):

```r
renv::restore()   # first time only — installs the locked package versions
```

Then `DATA_VERSION=<version> scripts/run_all.sh` (whole sessions → `output/<version>/`, entry +
1750 s → `output/<version>/1750s_post_entry/`), or run scripts under `scripts/` individually;
each is self-contained and sources `R/data.R` / `R/*.R` helpers as needed (set `DATA_VERSION`, and
`ANALYSIS_WINDOW_SEC` for the windowed version).

| Script | Brief section | Produces |
|---|---|---|
| `scripts/00_sanity_check.R` | — | Verifies schema, Craniotomy join coverage, and outcome-applicability counts; no figures. |
| `scripts/01_timecourse_plots.R` | 4.1 | Per virus, per outcome, DCZ vs Vehicle mean±SEM trace over `elapsed_time_thermal_sec` (12 figures: per virus, 3 frame-level outcomes + floor pref + velocity and cumulative distance from the tracking, the last two over time since arena entry). Pre/post-craniotomy split omitted — see status note below. |
| `scripts/02_spaghetti_paired_reproduce.R` | 4.2.1 | Per-mouse DCZ-vs-Vehicle paired points + group mean±SE, one figure per virus × outcome, faceted by stationary state; plus dorsal with both states combined (`dorsal_all_states`, time-weighted), whole-session velocity (`velocity_all_states`) and `distance_travelled` from the tracking, floor pref while stationary (`floor_preference_stationary`, mean over bouts) and while moving (`floor_preference_nonstationary`, session medians). Written to `output/spaghetti/{full_dataset,with_nestlet,no_nestlet}/` (18 figures per tier). |
| `scripts/03_craniotomy_plots.R` | 4.2.2 | Per-mouse Pre- vs Post-craniotomy, Gi+Gq combined, stationary/non-stationary split, 4 outcomes (incl. velocity) + dorsal all-states + tracking velocity/distance + floor pref, two contrasts (16 figures): `craniotomy_*` Pre = habituation/Saline vs Post = experimental/Vehicle (n=8); `craniotomy_saline_*` vs Post = rehabituation/Saline (n=6). Both **confounded with time/experience**, see design notes. |
| `scripts/04_distribution_plots.R` | 4.3 | Distributions per virus, DCZ vs Vehicle, split stationary/non-stationary, per outcome (16 figures: 4 outcomes incl. velocity × 2 viruses × 2 states). Plus velocity per virus, all states, from every tracking frame (2 figures). |
| `scripts/05_bout_organization_plots.R` | 4.2.3 | Bouts per session and mean bout duration, DCZ vs Vehicle, per virus (2 figures). |
| `scripts/06_timeseries_models.R` | 4.4 | LMM per outcome × virus (time × injection on per-session 1-min bin means; floor pref per bout) + GAMM difference-smooth check, incl. velocity (tracking 1-min bins) → `output/models/timeseries_{lmm,gamm}.csv` + 5 figures; per-session distance/velocity LMM → `activity_lmm.csv`. |
| `scripts/07_all_experiments_plots.R` | — | Same figure set as 02, but one jittered dot per experiment (session × lane), colored by mouse, no connecting lines, DCZ left / Vehicle right, mean±SEM across experiments beside each cloud. `output/all_experiments/{full_dataset,with_nestlet,no_nestlet}/`. |

All paired panels (4.2.2, 4.2.3) carry a paired t-test on per-mouse means (mice with both
conditions), the same test as the Python pipeline's `paired_tests_dcz_vehicle.csv`.

## Design notes / decisions made along the way

- **Craniotomy join**: ported from `../r_analysis/R/data.R`, not sourced —
  this workstream has its own renv/inputs. Only real change: date is parsed
  from `session`, not `video_file` (same leading `MM-DD-YY` token, same
  regex).
- **No `video_file` → metadata lookup needed**: unlike `../r_analysis/`,
  both master files here are already fully metadata-joined on the Python
  side (`../scripts/join_landmark_metadata.py`), including `stationary` at
  frame level (Stage 7 already classifies it, see
  `../scripts/stage7_real_run.py`) — so `load_bouts()`/`load_frames()` are
  much thinner than their legacy equivalents.
- **Corpus (2026-10-13)**: 43 sessions × lanes = 85 Stage 7 lanes, 8 DREADD mice (Gi 4539/4540/
  4551/4552, Gq 4541/4547/4548/4550) + 1 no-virus mouse, 3–4 sessions per mouse per condition.
  The 10 lanes added on 2026-10-13 are recovered sessions (thermalFeatures/recover_2026-10/):
  4541's two habituation sessions (07-08, 07-11) and three post-craniotomy rehabituation
  sessions (Post/Saline). The rehab `.seq` names carry wrong mouse IDs (LUT note); the LUT's are used.
  4 more DCZ/Vehicle test lanes were added the same day (thermalFeatures/recover_lanes_2026-10/):
  lanes never cropped in June whose sibling lane was already in the corpus (4548 DCZ, 4547 Vehicle,
  4551 Vehicle + DCZ). Per-mouse counts: ../../metadata/experiment_overview_by_mouse.csv
  (scripts/build_experiment_overview.py).
  (Built originally on a 6-lane, 2-mouse corpus; sample-size labels are now computed from the
  data, never hard-coded.)
- **Craniotomy is confounded with phase**: every Pre-craniotomy session is habituation (Saline),
  every Post session is experimental (DCZ/Vehicle). 4.1's time courses therefore stay
  post-craniotomy only, and 4.2.2 compares Pre/Saline vs Post/Vehicle (both control
  injections), which still mixes craniotomy with phase/experience — stated on every figure.
  4.2.2 has n=8 (4541's habituation sessions were recovered 2026-10-13). A second 4.2.2 set,
  `craniotomy_saline_*`, uses the recovered rehabituation sessions as Post (Post/Saline, n=6
  mice; 4551/4552's rehab session is unrecoverable): the same injection on both sides, so only
  craniotomy + time/experience differ. No other figure uses Saline rows.
- **Velocity** (`velocity_smooth_px_s`) isn't a native Stage 7 column. `../scripts/join_landmark_metadata.py
  --velocity-from <master_tracking_with_metadata.csv>` attaches it to the frame table from the re-track
  run (compute_velocity() output), nearest same-lane timestamp within 0.5 s: 98.2% of rows; the rest
  have NaN velocity in the tracking itself (QC-flagged frames). One distribution per virus, all
  states, as in `../r_analysis/`.
- **§4.4's model** (`R/model_timeseries.R`): LMM `value ~ time_c * injection + (1 + time_c |
  mouse_id) + (1 | session_lane)`, time centred at 15 min, falling back to random intercepts
  when the slope model is singular (recorded per row). Fitted on per-session 1-min bin means,
  not raw ~1 Hz frames — consecutive frames are autocorrelated and would make every p tiny.
  GAMM (`mgcv`, ordered-factor difference smooth) as a non-linearity check on every outcome.

- **Nestlet tiers** (2026-10-14): `Nestlet` (Y/N) is a per-(mouse, date) LUT column, joined like
  Craniotomy (`R/data.R::join_nestlet()`). The DCZ/Vehicle figure sets (02, 07) are drawn three
  times: full dataset, with nestlet, no nestlet. Post-craniotomy DCZ/Vehicle sessions per virus:
  no nestlet 11–12 per condition, with nestlet 4 per condition (one per mouse). Every
  Pre-craniotomy session had a nestlet, so the craniotomy contrasts (03) can't be split this way.
  Nest quality/location (EthoVision) is a separate analysis, to be integrated later.
- **Dorsal, all states**: stationary samples are ~13 s apart, non-stationary 1 s apart (93% of
  samples), so a pooled sample mean would just be the non-stationary mean. Each session's two state
  means are instead weighted by time in that state (stationary = sum of bout durations,
  non-stationary = number of 1 Hz samples) — `R/dcz_vehicle_units.R::experiment_means_dorsal_combined()`.
- **Floor temp while moving** (2026-10-14): bout-level `mean_floor_temp_c` only covers stationary
  time. The frame table now also carries `tracking_floor_temp_c`: the tracker's per-frame floor
  ROI next to the mouse, the same signal `mean_floor_temp_c` averages over a bout (r=0.995 against
  it on stationary samples). It's attached by `../scripts/join_landmark_metadata.py --velocity-from`
  alongside velocity, and is distinct from Stage 7's `floor_temp_mean_c` (local annulus at the tail).
  The non-stationary floor figures use each session's **median** over its 1 Hz moving samples, since
  a moving mouse sweeps the gradient and the distribution is wide and skewed.
- **Velocity everywhere** (2026-10-14): `velocity_smooth_px_s` is a fifth frame-level outcome
  (`OUTCOME_COLS`, non-NA rule) for the per-state views — Stationary / Non-stationary panels and
  rest / non-rest distributions, from Stage 7 samples. Stationary velocity is near zero by
  construction (~1 px/s vs ~14 moving), so in spaghetti, all_experiments and craniotomy velocity is
  drawn as separate `*_velocity_smooth_px_s_{stationary,nonstationary}.png` files rather than one
  figure faceted by state (`SPLIT_STATE_OUTCOMES`); the temperature outcomes stay faceted. Units are thermal px/s, not cm/s; the Test_3/4/7 thermal crops (one per camera
  regime) are all 412–416 px long, so the scale looks comparable across sessions, but it hasn't
  been checked for every session.
- **Activity from the tracking, not Stage 7** (2026-10-14, `R/tracking_activity.R`): Stage 7
  samples cover only ~70% of a session — QC-invalid frames (~10%) and stationary stretches too
  short to be bouts (~13–16%) are in neither its bout samples nor its non-stationary pass — and the
  split differs between habituation and experimental sessions (45% vs 35% of time in bouts). A
  first attempt at whole-session velocity from time-weighted Stage 7 samples was biased by this
  (it gave a spurious craniotomy speed-up) and was replaced: **whole-session velocity**
  (`velocity_all_states`, velocity time course, velocity model, all-states velocity histogram) and
  **distance travelled** now come straight from every valid thermal-tracking frame (~1 Hz), from
  arena entry. Distance = sum of centroid steps between consecutive valid frames; steps < 3 px are
  rest jitter and dropped (rest steps: median 1.0 px; moving: 10.6 px; 3 → 8 px changes totals only
  8%); steps across QC dropouts ≤ 5 s are bridged, longer gaps (~5% of time) skipped. Velocity is
  `compute_velocity()`'s smoothed speed, ~25% below raw step speed, so distance ÷ time and velocity
  differ in absolute terms. Per-session totals get an LMM (`value ~ injection + (1 | mouse_id)`,
  `output/models/activity_lmm.csv`) — the repeated-measures test behind the all_experiments dots.
- **Equal-length window** (2026-10-14): sessions run 593–2398 s after arena entry, which biases
  totals like distance. `scripts/run_all.sh [WINDOW]` runs everything twice: whole sessions →
  `output/<version>/`, and every lane cut to [entry, entry + 1750 s] → `output/<version>/1750s_post_entry/`
  (`ANALYSIS_WINDOW_SEC`, `data.R::apply_window()`). 1750 s is the longest window nearly every lane
  fills: 54 of 62 DCZ/Vehicle lanes (8 lanes from 4 sessions with short RGB video, 593–1437 s, are
  dropped; 2000 s would have kept only 8 lanes). Entry times are the human-verified ones Stage 7
  used (`STAGE7_CONFIGS`, read with `jsonlite` — added to renv). Bouts straddling the cut keep their
  bout-level averages but have their end clipped (≤1 per lane). Windowed figures carry a caption
  saying so.
- **all_experiments SEM** is across experiments treated as independent (3–4 per mouse), so it
  describes the dots' spread, not an inferential error bar.
- **Mouse colors** are fixed per mouse ID (`mouse_colors()`), so a mouse keeps its color across
  tiers and figure families.

## Status

- [x] Project scaffold (renv, directory layout, `R/data.R`, sanity check)
- [x] 4.1 Time-course plots (post-craniotomy only — see design notes)
- [x] 4.2.1 Paired DCZ/Vehicle spaghetti plots (dorsal/warm-spot/tail-ΔT + floor pref)
- [x] 4.2.2 Craniotomy-effect plots (Pre/Saline vs Post/Vehicle, and vs Post/Saline rehab; both confounded with time)
- [x] 4.2.3 Bout-organization plots (bouts per session, bout duration)
- [x] 4.3 Distribution plots (dorsal/warm-spot/tail-ΔT)
- [x] 4.3 velocity distributions (velocity joined via `join_landmark_metadata.py --velocity-from`)
- [x] 4.4 Time-series models (LMM + GAMM check)
- [x] Spaghetti split per virus, dorsal all-states figure, nestlet tiers, all_experiments figures (2026-10-14)
- [x] Velocity added to every output folder; distance travelled; whole-session activity from tracking; equal-length 1750 s window (2026-10-14)
- All of the above re-run on the full 85-lane corpus, 2026-10-13. Result statistics are kept
  out of this public repo, in the git-ignored `RESULTS_NOTES.md` next to this file.
