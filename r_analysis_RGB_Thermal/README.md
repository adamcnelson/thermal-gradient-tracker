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
| `scripts/03_spaghetti_craniotomy_and_bouts.R` | 4.2.2, 4.2.3 | Craniotomy-effect plots (combined Gi+Gq, stationary/non-stationary split) and bout-organization plots (count, duration). |
| `scripts/04_distribution_plots.R` | 4.3 | Distributions per virus, DCZ vs Vehicle, split stationary/non-stationary, per outcome (12 figures: 3 outcomes × 2 viruses × 2 states). Velocity distributions deferred — see status note below. |
| `scripts/05_timeseries_model.R` | 4.4 | LMM (or GAMM) fit per outcome per virus, testing the time×injection interaction. |

**Not yet built**: `scripts/03_spaghetti_craniotomy_and_bouts.R` (4.2.2's
craniotomy-effect half is blocked by the all-Post-craniotomy corpus issue
below — only 4.2.3's bout-organization plots are buildable right now) and
`scripts/05_timeseries_model.R` (4.4). Built so far: `R/data.R`,
`scripts/00_sanity_check.R`, `R/plot_timecourse.R` + `scripts/01_timecourse_plots.R`,
`R/plot_spaghetti.R` + `scripts/02_spaghetti_paired_reproduce.R`,
`R/plot_distributions.R` + `scripts/04_distribution_plots.R`.

**Paused here (Adam, 2026-09-02)**: a full-dataset SLURM run is coming;
resume/optimize this R work once those fuller results exist rather than
continuing to build against the current 6-session (2-mouse) corpus.

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
- **4.1's craniotomy split is omitted for now** (Adam, 2026-09-02): every
  session in the current 6-session corpus is `craniotomy=="Post"` — no
  pre-craniotomy data exists yet to split against. `plot_timecourse_one()`
  doesn't facet on it; revisit once pre-craniotomy sessions are tracked.
- **n=1 mouse per virus×injection cell** in the current corpus (2 mice
  total, Gi=4540, Gq=4541, each seeing both DCZ and Vehicle across
  different sessions) — group mean±SEM traces are real but SEM is
  necessarily 0 throughout (see `.sem()`'s n<=1 case). Every time-course
  figure's subtitle reports the real per-injection mouse count so this
  isn't silently implied to be a larger sample.
- **Velocity distributions deferred** (Adam, 2026-09-02): the brief's 4.3
  row wants `velocity_smooth_px_s` too, but it isn't a native Stage 7
  output — it needs a separate per-timestamp join against
  `trackingOutputs/*_tracking_every10frames.csv` via
  `src/velocity.py::compute_velocity()`, a real data-prep step rather than
  just another plot. Picking this up after the full-dataset SLURM run
  regenerates the underlying tracking CSVs anyway.
- **§4.4's model**: brief allows an LMM (`outcome ~ elapsed_time_thermal_sec
  * injection + (1 + elapsed_time_thermal_sec | mouse_id)`) or a GAMM
  smooth-by-injection term if a trajectory isn't linear. Plan: start with
  the LMM (`lme4`+`lmerTest`) for every outcome, and only reach for `mgcv`'s
  GAMM on a specific outcome if its §4.1 time-course plot visibly isn't
  linear — decided from what the descriptive plots actually show, not
  committed upfront for all three outcomes.

## Status

- [x] Project scaffold (renv, directory layout, `R/data.R`, sanity check)
- [x] 4.1 Time-course plots (craniotomy split omitted, see design notes)
- [x] 4.2.1 Paired DCZ/Vehicle spaghetti plots (dorsal/warm-spot/tail-ΔT + floor pref)
- [x] 4.3 Distribution plots (dorsal/warm-spot/tail-ΔT; velocity deferred)
- [ ] **Paused** — resume after the full-dataset SLURM run
- [ ] 4.2.1 Reproduce paired DCZ/Vehicle spaghetti plots (dorsal/warm-spot/tail-ΔT + floor pref)
- [ ] 4.2.2 Craniotomy-effect spaghetti plots (stationary/non-stationary split)
- [ ] 4.2.3 Bout-organization effect plots
- [ ] 4.3 Distribution plots
- [ ] 4.4 Time-series model (LMM/GAMM)
