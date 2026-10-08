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

`../landmark_outputs/`, already fully metadata-joined by
`../scripts/join_landmark_metadata.py` (project_brief_v8.md §3.2):

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

Figures under `output/` (git-ignored — regenerate via the scripts below
rather than committing PNGs).

## How to run

From `r_analysis_RGB_Thermal/` (this directory):

```r
renv::restore()   # first time only — installs the locked package versions
```

Then run scripts under `scripts/` in order; each is self-contained and
sources `R/data.R` / `R/*.R` helpers as needed.

| Script | Brief section | Produces |
|---|---|---|
| `scripts/00_sanity_check.R` | — | Verifies schema, Craniotomy join coverage, and outcome-applicability counts; no figures. |
| `scripts/01_timecourse_plots.R` | 4.1 | Per virus, per outcome, DCZ vs Vehicle mean±SEM trace over `elapsed_time_thermal_sec` (8 figures: 2 viruses × 3 frame-level outcomes + 2 viruses × floor-preference). Pre/post-craniotomy split omitted — see status note below. |
| `scripts/02_spaghetti_paired_reproduce.R` | 4.2.1 | Per-mouse DCZ-vs-Vehicle paired points + group mean±SE, faceted virus × stationary state, per outcome (4 figures: 3 frame-level outcomes + floor pref, the last faceted virus-only). |
| `scripts/03_craniotomy_plots.R` | 4.2.2 | Per-mouse Pre- vs Post-craniotomy, Gi+Gq combined, stationary/non-stationary split, 3 outcomes + floor pref (4 figures). Pre = habituation/Saline vs Post = experimental/Vehicle — **confounded with phase**, see design notes. |
| `scripts/04_distribution_plots.R` | 4.3 | Distributions per virus, DCZ vs Vehicle, split stationary/non-stationary, per outcome (12 figures: 3 outcomes × 2 viruses × 2 states). Velocity distributions deferred — see status note below. |
| `scripts/05_bout_organization_plots.R` | 4.2.3 | Bouts per session and mean bout duration, DCZ vs Vehicle, per virus (2 figures). |
| `scripts/06_timeseries_models.R` | 4.4 | LMM per outcome × virus (time × injection on per-session 1-min bin means; floor pref per bout) + GAMM difference-smooth check → `output/models/timeseries_{lmm,gamm}.csv` + 4 figures. |

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
- **Corpus (2026-10-08)**: 38 sessions × lanes = 71 Stage 7 lanes, 8 DREADD mice (Gi 4539/4540/
  4551/4552, Gq 4541/4547/4548/4550) + 1 no-virus mouse, 3–4 sessions per mouse per condition.
  (Built originally on a 6-lane, 2-mouse corpus; sample-size labels are now computed from the
  data, never hard-coded.)
- **Craniotomy is confounded with phase**: every Pre-craniotomy session is habituation (Saline),
  every Post session is experimental (DCZ/Vehicle). 4.1's time courses therefore stay
  post-craniotomy only, and 4.2.2 compares Pre/Saline vs Post/Vehicle (both control
  injections), which still mixes craniotomy with phase/experience — stated on every figure.
  Mouse 4541 has no Pre data (its habituation sessions have no homography, so no Stage 7), so
  4.2.2 has n=7.
- **Velocity distributions deferred** (Adam, 2026-09-02): the brief's 4.3
  row wants `velocity_smooth_px_s` too, but it isn't a native Stage 7
  output — it needs a separate per-timestamp join against
  `trackingOutputs/*_tracking_every10frames.csv` via
  `src/velocity.py::compute_velocity()`, a real data-prep step rather than
  just another plot. Picking this up after the full-dataset SLURM run
  regenerates the underlying tracking CSVs anyway.
- **§4.4's model** (`R/model_timeseries.R`): LMM `value ~ time_c * injection + (1 + time_c |
  mouse_id) + (1 | session_lane)`, time centred at 15 min, falling back to random intercepts
  when the slope model is singular (recorded per row). Fitted on per-session 1-min bin means,
  not raw ~1 Hz frames — consecutive frames are autocorrelated and would make every p tiny.
  GAMM (`mgcv`, ordered-factor difference smooth) as a non-linearity check on every outcome.
  First results: no robust trajectory divergence. Gi floor pref's slope difference (p=0.007)
  is driven by the first 5 min of exploration (p=0.14 dropping bouts <5 min, 0.81 dropping
  <10 min); what holds is Gq DCZ's offset (warmer floor, warmer dorsal), not a shape change.

## Status

- [x] Project scaffold (renv, directory layout, `R/data.R`, sanity check)
- [x] 4.1 Time-course plots (post-craniotomy only — see design notes)
- [x] 4.2.1 Paired DCZ/Vehicle spaghetti plots (dorsal/warm-spot/tail-ΔT + floor pref)
- [x] 4.2.2 Craniotomy-effect plots (Pre/Saline vs Post/Vehicle, confounded with phase)
- [x] 4.2.3 Bout-organization plots (bouts per session, bout duration)
- [x] 4.3 Distribution plots (dorsal/warm-spot/tail-ΔT)
- [ ] 4.3 velocity distributions (needs the per-timestamp tracking join)
- [x] 4.4 Time-series models (LMM + GAMM check)
- All of the above re-run on the full 71-lane corpus, 2026-10-08.
