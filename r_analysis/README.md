# r_analysis — R/ggplot2 treatment-effect analysis

A separate, R-based plotting layer on top of the Python tracking/bout-detection
pipeline's outputs (`../src`, `../scripts`). This workstream only **reads**
those outputs — it doesn't regenerate or modify anything under `trackingOutputs/`
or `bouts/` at the repo root, and it doesn't touch the top-level `README.md`.

Full context and rationale: `../project_brief_v6.md`.

## What it reads

Raw data, one directory up from the repo root:

```
../../SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/bouts/
  master_tracking_with_metadata.csv   # frame-level, one row per sampled frame
  bout_table.csv                      # bout-level, one row per detected rest bout
```

Plus the metadata LUT already local to this repo, `../metadata/LUT_CLEAN_July6.csv`,
used to join `Craniotomy` status (`Pre-craniotomy` / `Post`) into both tables —
see `R/data.R`. That column isn't in `master_tracking_with_metadata.csv` yet
(the Python join stage doesn't select it); this workstream joins it in on the R
side using `(mouse_id, date)` parsed from `video_file`, rather than re-running
the Python pipeline. Join coverage is 100% for this run (verified in
`scripts/00_sanity_check.R`).

`master_tracking_with_metadata.csv` is filtered to `qc_flag == "ok"` everywhere
in this workstream (invalid/no-data frames are dropped explicitly, never
treated as zero).

## What it produces

Figures under `output/` (git-ignored — regenerate via the scripts below rather
than committing PNGs). Filenames match the section numbers in
`project_brief_v6.md` section 3.

## How to run

From `r_analysis/` (this directory):

```r
renv::restore()   # first time only — installs the locked package versions
```

Then run scripts under `scripts/` in order; each is self-contained and sources
`R/data.R` / `R/*.R` helpers as needed:

| Script | Brief section | Produces |
|---|---|---|
| `scripts/00_sanity_check.R` | — | Verifies schema + Craniotomy join coverage against `project_brief_v6.md`; no figures. |
| `scripts/01_timecourse_plots.R` | 3.1 | `output/timecourse/timecourse_{outcome}_{virus}_{pre\|post}.png` — 16 figures (4 outcomes × 2 viruses × pre/post-craniotomy). |
| `scripts/02_spaghetti_paired_reproduce.R` | 3.2.1 | `output/spaghetti/paired_dcz_vehicle_{outcome}.png` — 4 figures, reproducing `src/treatment_plots.py::plot_dcz_vehicle_paired()` in ggplot2. |
| `scripts/03_spaghetti_craniotomy_and_bouts.R` | 3.2.2, 3.2.3 | `output/spaghetti/craniotomy_effect_{outcome}.png` (4 figures) and `output/spaghetti/bout_{count,duration}_by_injection.png` (2 figures). |
| `scripts/04_distribution_plots.R` | 3.3 | `output/distributions/dist_{outcome}_{virus}_{rest\|nonrest}.png` (12 figures) and `output/distributions/dist_velocity_smooth_px_s_{virus}.png` (2 figures, not split by rest state). |

## Design notes / decisions made along the way

- **Craniotomy join (open question A):** done on the R side (`R/data.R::join_craniotomy()`),
  confirmed with Adam rather than extending the Python pipeline. Since
  `master_tracking_with_metadata.csv` already carries `mouse_id` per row (from
  the Python join stage), the R-side join only needs to extract the recording
  date out of `video_file` and match on `(mouse_id, date)` against the LUT —
  no need to re-derive `mouse_id` from the filename. 100% join coverage,
  verified in `scripts/00_sanity_check.R`, and the resulting `craniotomy`
  column reproduces the injection↔craniotomy mapping Adam confirmed in open
  question B exactly (`Saline` ↔ `Pre-craniotomy`, `DCZ`/`Vehicle` ↔ `Post`).
- **Rehabituation trials (open question B):** none are present in this run
  (confirmed by the sanity check — `injection x craniotomy` has zero
  Saline-and-Post rows). All post-craniotomy filtering restricts to
  `injection %in% c("DCZ", "Vehicle")` rather than special-casing rehab
  directly, which is deliberate: it produces the same result now and won't
  silently misclassify a rehab session if one is ever tracked in a future run.
- **`bout_table.csv` metadata join (open question C):** `R/data.R::load_bouts()`
  builds a deduplicated `video_file -> {mouse_id, virus, injection, phase, craniotomy}`
  lookup from the master table and left-joins it on; errors loudly (rather than
  dropping rows) if any bout fails to match. Non-rest-bout data for the 3.2.2
  craniotomy spaghetti plots comes from `master_tracking_with_metadata.csv`
  (`stationary == FALSE`), not `bout_table.csv`, which by construction only
  contains already-stationary rows.
- **Time alignment (open question D):** `R/plot_timecourse.R` bins
  `elapsed_time_sec` into 30-second intervals, takes each mouse's per-bin mean
  first, then averages those per-mouse bin means across mice per injection —
  so a mouse with more sampled frames in a bin doesn't dominate the group mean.
  SEM is 0 when only one mouse contributes to a bin.
- **3.2.2 (craniotomy effect) virus scope:** the brief says "combining Gi and
  Gq animals together" — this excludes the one `virus == "none"` control mouse
  (4549) present in this run, matching the Gi/Gq-only scope already used by
  3.2.1's reproduction of the Python reference plots.
- The 4 reference PNGs in `../bouts/analysis_plots/paired_dcz_vehicle_*.png`
  are from an earlier, much smaller run (n≈101, single mouse per virus×injection
  cell) — they were used to validate the plot *structure* (2×2 grid, per-mouse
  connected lines, black mean±SE diamond), not to reconcile exact values against
  this run's fuller 8-mouse dataset.

## Status

- [x] Project scaffold (renv, directory layout)
- [x] Data loading + Craniotomy join (`R/data.R`), verified 100% join coverage
- [x] 3.1 Time-course plots
- [x] 3.2.1 Reproduce paired DCZ/Vehicle spaghetti plots
- [x] 3.2.2 Craniotomy-effect spaghetti plots
- [x] 3.2.3 Bout-organization effect plots
- [x] 3.3 Distribution plots
