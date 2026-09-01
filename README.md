# thermal-gradient-tracker

GitHub: https://github.com/adamcnelson/thermal-gradient-tracker

Mouse tracking, temperature extraction, stationary bout detection, and treatment-effect analysis for FLIR thermography `.seq` videos from the rodent thermal-gradient assay.

**Pipeline overview:**
1. **Tracking** — read cropped single-track `.seq` files, segment the mouse, extract per-frame surface and floor temperatures → `trackingOutputs/*.csv`
2. **Bout detection** — compute centroid velocity and dispersion, detect stationary rest epochs → `bouts/`
3. **Metadata join** — link each tracking file to its animal and experimental condition via the LUT
4. **Analysis** — linear mixed models and treatment-effect plots

---

## ⚠ Before you start: point at your croppedSeqFiles folder

**The only path you ever need to supply is the `croppedSeqFiles` input location** — passed as `--input` (one file) or `--input-dir` (a folder) wherever a script needs raw thermal video. Substitute your local `croppedSeqFiles` path in place of `../croppedSeqFiles/...` in the commands below.

Every other directory — tracking outputs, QC images/plots, bout tables, analysis results — is created automatically inside the `thermal-gradient-tracker` project folder the first time it's needed, using the layout below. Nothing else needs a path edit, and the layout is identical whether the repo lives on Adam's iMac or on MedicineBow.

```
thermal-gradient-tracker/
├── bouts/
│   ├── analysis_plots/
│   ├── analysis_tables/
│   ├── qc_plots/
│   └── reference/
├── mask_preview/
├── metadata/
├── preview_tracking/
├── scripts/
├── src/
├── tests/
└── trackingOutputs/
    ├── bout_examples/
    ├── qc_images/
    │   └── <per-video subfolders>/
    ├── qc_plots/
    └── qc_summaries/
```

Every automated script computes these paths from the project root (`src/paths.py`), not from the current working directory, so it doesn't matter where you invoke the script from. `--config` and `--output-dir` (and similar) still exist as optional overrides for advanced use (e.g. the SLURM job), but you shouldn't need them for normal use.

The retired manual steps (arena masking, training-frame labeling — Steps 2 and 5 below) are out of scope for this auto-directory behavior; they still take an explicit `--input`/`--seq` path the same as before.

---

## Setup

```bash
# Create and activate a virtual environment (first time only)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

### exiftool (required for accurate temperature conversion)

```bash
brew install exiftool
```

The pipeline uses the **FLIR Planck equation** to convert raw detector values to Celsius:

```
T_celsius = B / ln(R1 / (R2 × (raw + O)) + F) − 273.15
```

Constants are read automatically from each `.seq` file's EXIF metadata via exiftool. If exiftool is not available, the pipeline falls back to a linear approximation and prints a warning — **temperatures will be incorrect without exiftool**.

---

## Part 1 — Tracking pipeline

### Step 1 — Preview the input file

```bash
python scripts/preview_tracking_input.py \
    --input "../croppedSeqFiles/07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Front.seq"
```

Check the saved images and `_info.txt` to confirm the file loaded correctly.

### Step 2 — Create the arena mask

The arena mask marks the valid floor region (excluding walls and apparatus edges). Only needs to be done once per crop geometry.

```bash
python scripts/create_arena_mask.py \
    --input "../croppedSeqFiles/07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Front.seq" \
    --config "tracking_config.json" \
    --output-dir "mask_preview"
```

If no config file exists yet:
```bash
cp tracking_config.example.json tracking_config.json
```

An interactive window opens. Click to draw a polygon around the walkable floor, then press **ENTER** to save. Press **Z** to undo the last point. Check `mask_preview/*_arena_mask_preview.png` to confirm the polygon looks right.

### Step 3 — Track one file

```bash
python scripts/track_temperatures.py \
    --input "../croppedSeqFiles/07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Front.seq" \
    --sampling-interval 10
```

Outputs:
- `trackingOutputs/<stem>_tracking_every10frames.csv`
- `trackingOutputs/qc_images/<stem>/` — overlay images showing mask, mouse ROI, floor ROI
- `trackingOutputs/qc_summaries/<stem>_qc_summary.csv`
- `trackingOutputs/qc_plots/<stem>_diagnostic.png`

### Step 4 — Inspect QC images

Open a few images in `trackingOutputs/qc_images/`. Confirm:
- **Cyan circle** = mouse surface ROI (should sit on the mouse's back/trunk)
- **Green circle** = adjacent floor ROI (should be on the empty gradient, not on the mouse)
- **Cyan contour** = mouse segmentation boundary

If the mouse ROI is misplaced, adjust `segmentation_threshold_sigma`, `mouse_roi_radius_px`, or the arena polygon in `tracking_config.json`.

### Step 5 — Optional: manual training

If automated segmentation is unreliable (e.g., mouse has low contrast near a 37 °C floor):

```bash
# Sample 50 frames for annotation
python scripts/select_training_frames.py \
    --input "../croppedSeqFiles/.../Front.seq" \
    --n-frames 50 \
    --output-dir "training_frames"

# Label the mouse in each image using labelme:
#   pip install labelme
#   labelme training_frames/images/

# Tune the detector from labeled frames
python scripts/train_mouse_detector.py \
    --seq "../croppedSeqFiles/.../Front.seq" \
    --labels "training_frames/" \
    --config "tracking_config.json"
```

The optimized `segmentation_threshold_sigma` is written back to `tracking_config.json`.

### Step 6 — Batch process all .seq files

```bash
# One session folder
python scripts/batch_track_temperatures.py --input-dir "../croppedSeqFiles/07-28-25_Test_3/"

# All subfolders at once
python scripts/batch_track_temperatures.py --input-dir "../croppedSeqFiles/" --recursive
```

Existing output CSVs are skipped unless you pass `--overwrite`.

---

## Part 2 — Bout detection and analysis

> **Note on the test subset:** results produced on the small local subset are labeled **PRELIMINARY — insufficient sample for inference**. The same commands run unchanged on the full dataset — only more data are needed for valid conclusions.

### Step 7 — Place the metadata LUT

Copy `LUT_CLEAN_July6.csv` into the `metadata/` directory:

```
metadata/LUT_CLEAN_July6.csv
```

### Step 8 — Configure bout-detection thresholds

Thresholds are set directly in `analysis_config.json`. Open it and adjust the `bouts` block as needed:

```json
{
  "bouts": {
    "position_smoothing_window_samples": 5,
    "dispersion_window_samples": 5,
    "stationary_dispersion_threshold_px": 4.0,
    "velocity_smoothing_window_samples": 5,
    "stationary_velocity_threshold_px_per_s": 5.0,
    "min_bout_duration_sec": 15.0,
    "max_gap_merge_sec": 6.0
  }
}
```

Key parameters:
- **`position_smoothing_window_samples`** — rolling-median smoothing applied to centroid-x before dispersion is computed from it, so brief per-frame jitter/dropout doesn't spuriously fragment a real rest bout.
- **`stationary_dispersion_threshold_px`** — primary criterion; lower = stricter (fewer, tighter bouts).
- **`stationary_velocity_threshold_px_per_s`** — velocity guard; lower = stricter.
- **`dispersion_window_samples`** — longer window smooths over outlier frames; shorter is more responsive.
- **`min_bout_duration_sec`** — discard short rest epochs.
- **`max_gap_merge_sec`** — merge bouts separated by brief interruptions (e.g. a short low-contrast tracking dropout). Set to 6.0 (up from 2.0) after Issue 1: real dropouts observed on local test files run 2–5s edge-to-edge, and 2.0s was fragmenting genuine rest bouts around low mouse/floor-contrast periods into several short ones. Still well below `min_bout_duration_sec`, so distinct rest bouts separated by real movement shouldn't over-merge — but this was an empirical call on a small local sample, not a value Adam specified, so revisit it once more data is available.

After changing any parameter, re-run Step 9 on one representative file and inspect the QC plot. The red dashed threshold lines on the dispersion and velocity panels show exactly where the cuts are.

### Step 9 — Compute bouts for one file (inspect first)

```bash
python scripts/compute_bouts.py \
    --input "trackingOutputs/07-28-25_4540_B_4541_F_Test3-004_Front_tracking_every10frames.csv" \
    --overwrite
```

Output: `bouts/qc_plots/<stem>_bouts_diagnostic.png` and `.pdf`.

The QC plot shows five stacked panels on a shared time axis:
1. `mouse_centroid_x` over time — detected stationary bouts shaded blue.
2. Rolling dispersion of centroid-x — red dashed line = dispersion threshold.
3. Smoothed velocity — red dashed line = velocity threshold.
4. Stationary/moving ethogram bar (green = stationary).
5. Floor and mouse surface temperatures for context.

If bouts don't look right, edit `analysis_config.json` and re-run with `--overwrite`.

### Step 10 — Batch-compute bouts for all tracking CSVs

```bash
python scripts/batch_compute_bouts.py --recursive --overwrite
```

Outputs:
- `bouts/qc_plots/<stem>_bouts_diagnostic.png` — one QC plot per tracking file.
- `bouts/bout_table.csv` — all stationary bouts across all files (one row per bout).
- `bouts/bout_summary_per_file.csv` — per-file statistics.

### Step 11 — Join metadata

```bash
python scripts/join_metadata.py
```

**Check the join report:** open `bouts/metadata_join_report.csv` and confirm:
- All files show `matched_exact` or `matched_token`.
- No `unmatched` rows (flag any to Adam if found).
- Any `seq_name_warning` entries correspond to known ambiguous filenames in the LUT.

Excluded rows (single-recording experiments) are written to `bouts/metadata_excluded.csv`.

### Step 12 — Run treatment-effect analysis

```bash
python scripts/analyze_treatment_effects.py
```

Outputs:
- `bouts/analysis_plots/` — all treatment-effect figures.
- `bouts/analysis_tables/` — group means ± SE, LMM coefficients.
- `bouts/analysis_README.txt` — which models fit vs. fell back to descriptives.

---

## Config file reference

### `tracking_config.json`

| Parameter | Default | Description |
|---|---|---|
| `sampling_interval_frames` | 10 | Sample every N frames |
| `mouse_roi_radius_px` | 8 | Radius of mouse surface ROI (pixels) |
| `floor_roi_radius_px` | 8 | Radius of the "floor"/location temperature read (pixels), same footprint as the mouse ROI |
| `max_floor_roi_shift_px` | 80 | **Unused as of Issue 1** — was for the retired live-adjacent floor ROI search |
| `floor_roi_search_step_px` | 2 | **Unused as of Issue 1** — was for the retired live-adjacent floor ROI search |
| `min_mouse_area_px` | 100 | Min mouse blob area to accept (global detection) |
| `max_mouse_area_px` | 10000 | Max mouse blob area to accept |
| `background_n_frames` | 200 | Frames used to build the temporal median background (used for both segmentation and, since Issue 1, "floor" temperature) |
| `segmentation_threshold_sigma` | 3.0 | Global foreground threshold: mean + N×std |
| `camera_fps` | 8.0 (in the shared default `tracking_config.json`) | Real camera acquisition rate; converts frame_number to elapsed_time_sec. Camera is configured for 10 fps but drops frames — 8 was the "confirmed true rate" per `.seq` EXIF `FrameRate` tag (v7 Stage 0 audit, 2026-08-12). **⚠️ 2026-08-25: this EXIF-based conclusion is now contradicted by real, human-verified cross-modal timing anchors for Test_3 and Test_4** — fitting RGB↔thermal anchor pairs spanning full sessions cleanly implies the true rate is 10.0 fps in both (not 8.0), and re-deriving session duration at 10fps matches the RGB recording's independently-verified real duration far better than at 8fps (see project memory: v7 camera_fps 8-vs-10 correction). Fixed so far ONLY for `tracking_config_test3.json` and `tracking_config_test4.json` (both now `camera_fps: 10.0`) — the shared default `tracking_config.json` is deliberately left at 8.0 until the rest of the session corpus is re-audited. Do not assume 8.0 is correct for any new session without checking. |
| `enable_local_fallback` | `true` | Enable local-fallback recovery (Issue 1) when the global threshold loses the mouse |
| `local_fallback_search_radius_px` | 25 | Symmetric window half-width/height around the last centroid for local-fallback recovery |
| `local_fallback_threshold_percentile` | 80.0 | Percentile of the local window's foreground score used as the fallback threshold |
| `local_fallback_min_area_px` | 30 | Minimum blob area accepted during local-fallback recovery |
| `arena_polygon` | null | Polygon vertices defining the valid floor; set by `create_arena_mask.py`. Still used to validate frame shape; no longer used to constrain floor placement since floor temperature now reads the same footprint as the mouse (Issue 1) |

### `analysis_config.json` — `bouts` block

| Parameter | Default | Description |
|---|---|---|
| `position_smoothing_window_samples` | 5 | Rolling-median smoothing window for centroid-x, applied before dispersion (Issue 1) |
| `dispersion_window_samples` | 5 | Rolling window width for centroid-x dispersion |
| `dispersion_metric` | `"mad"` | `mad` (recommended), `iqr`, or `sd` |
| `stationary_dispersion_threshold_px` | 4.0 | Primary stationarity threshold (px) |
| `velocity_smoothing_window_samples` | 5 | Smoothing window for velocity |
| `stationary_velocity_threshold_px_per_s` | 5.0 | Velocity guard threshold (px/s) |
| `qc_valid_only_for_dispersion` | `true` | Use only `qc_flag==ok` frames for dispersion |
| `min_bout_duration_sec` | 15.0 | Minimum length to accept a stationary bout |
| `max_gap_merge_sec` | 6.0 | Maximum gap to merge two adjacent bouts (raised from 2.0 in Issue 1 — see Step 8) |
| `frame_rate_fps` | 8.0 | Not currently consumed by the pipeline — bout timing comes from `elapsed_time_sec`, which is computed from `camera_fps` in `tracking_config.json`. Kept in sync for documentation |

### `analysis_config.json` — `analysis` block

| Parameter | Default | Description |
|---|---|---|
| `group_factors` | `["virus","injection","phase"]` | Grouping factors for all analyses |
| `primary_contrast` | `{"factor":"injection","levels":["DCZ","Vehicle"]}` | Primary treatment comparison |
| `random_effect` | `"mouse_id"` | Grouping variable for LMM random intercept |
| `trial_length_reference_sec` | 3000 | Reference trial length for rate normalization |
| `min_animals_for_lmm` | 3 | Minimum unique animals required to fit an LMM |
| `min_obs_per_group_for_lmm` | 2 | Minimum observations per group level for LMM |

---

## Output file reference

### Tracking CSV columns (`trackingOutputs/<stem>_tracking_every10frames.csv`)

| Column | Description |
|---|---|
| `video_file` | Source `.seq` filename |
| `frame_number` | Frame index (0-based) |
| `elapsed_time_sec` | Elapsed time, computed from `camera_fps` in `tracking_config.json` (8 fps — camera is configured for 10 fps but drops frames in practice; confirmed via `.seq` EXIF, v7 Stage 0 audit) |
| `mouse_surface_temp_mean/median` | Mouse surface temperature (°C), from the live frame |
| `floor_temp_mean/median` | "Floor"/location temperature (°C) — as of Issue 1, read from the historical background at the *same footprint* as the mouse ROI, not a live adjacent ROI. This is deliberate: it's robust to local gradient irregularities (warps/bubbles/arches) that would make a nearby live reading unrepresentative of the mouse's actual position |
| `mouse_minus_floor_temp_mean` | Mouse − floor temperature (°C); thermal preference proxy |
| `mouse_centroid_x/y` | Mouse centroid position (pixels) |
| `tracking_confidence` | Segmentation quality score (0–1); local-fallback recoveries are set to a fixed 0.5 |
| `mouse_roi_valid` | True if mouse ROI was successfully placed |
| `floor_roi_valid` | True if the floor/location background read succeeded (normally true whenever `mouse_roi_valid` is) |
| `detection_method` | `global`, `local_fallback`, or `none` — which stage produced this frame's mouse mask (Issue 1) |
| `qc_flag` | `ok`, `pre_entry`, `no_mouse`, `no_mouse_roi`, `no_floor_roi`, `jump` |
| `qc_notes` | Human-readable description of any QC issues |

### `bouts/` output tree

```text
bouts/
  reference/
    *_diagnostic_boutExamples.pdf  — annotated reference figures (visual reference only)
  qc_plots/
    <stem>_bouts_diagnostic.png    — 5-panel bout detection QC figure (one per tracking file)
    <stem>_bouts_diagnostic.pdf    — same, PDF version
  bout_table.csv                   — all bouts across all files (one row per bout)
  bout_summary_per_file.csv        — per-file bout statistics
  master_tracking_with_metadata.csv — all tracking frames joined to animal/condition metadata
  metadata_join_report.csv         — one row per tracking file, showing join outcome
  metadata_excluded.csv            — LUT rows dropped (single-recording experiments)
  analysis_tables/
    descriptive_*.csv              — group means ± SE
    lmm_*.csv                      — LMM coefficients (when data are sufficient)
  analysis_plots/
    bout_duration_by_*.png         — bout duration strip/box plots by group
    n_stationary_bouts_by_*.png
    bouts_per_1000s_by_*.png
    floor_temp_by_*.png            — preferred floor temperature (stationary vs non-stationary)
    mouse_surface_temp_mean_by_*.png
    mouse_minus_floor_temp_mean_by_*.png
    contrast_dcz_vehicle_*.png     — DCZ vs Vehicle paired plots
    floor_temp_dist_by_*.png       — floor temperature distributions
    coef_plot_*.png                — LMM coefficient plots (when data are sufficient)
  analysis_README.txt              — which models fit, n per analysis
```

### Additional columns in `master_tracking_with_metadata.csv`

| Column | Description |
|---|---|
| `mouse_id` | Animal identifier from LUT |
| `tail_id` | Tail marking (mid/tip/base+mid) |
| `virus` | DREADD virus type (`Gi`/`Gq`/`none`) |
| `injection` | Injection condition (`Saline`/`Vehicle`/`DCZ`) |
| `phase` | Recording phase (`habituation`/`experimental`) |
| `lane` | Track lane (`F`/`B`) |
| `track` | Track name (`Front`/`Back`) |
| `stationary` | Boolean: frame classified as stationary |
| `centroid_x_roll_dispersion` | Rolling MAD of centroid-x |
| `velocity_px_s` | Euclidean velocity (px/s) |
| `velocity_smooth_px_s` | Smoothed velocity (px/s) |

### Bout table columns (`bout_table.csv`)

| Column | Description |
|---|---|
| `bout_index` | Index within file |
| `bout_start_sec`, `bout_end_sec` | Bout time boundaries |
| `bout_duration_sec` | Duration in seconds |
| `n_samples` | Number of frames in bout |
| `floor_temp_mean_bout` | Mean floor temperature during bout |
| `mouse_surface_temp_mean_bout` | Mean mouse surface temperature during bout |
| `mouse_minus_floor_temp_mean_bout` | Mean mouse−floor temperature difference |
| `mean_centroid_x` | Mean x-position (gradient location) |

### Per-file summary columns (`bout_summary_per_file.csv`)

| Column | Description |
|---|---|
| `trial_length_sec` | Tracked trial length derived from the CSV |
| `trial_length_lut` | LUT `Arena_time` value (cross-check) |
| `n_stationary_bouts` | Count of detected bouts |
| `total_stationary_time_sec` | Total time in stationary bouts |
| `fraction_time_stationary` | Fraction of trial spent stationary |
| `mean_bout_duration_sec` | Mean bout duration |
| `bouts_per_1000s` | Rate-normalized bout count |
| `mean_preferred_floor_temp` | Mean of per-bout mean floor temperatures |

---

## Running tests

```bash
pytest tests/ -v
```

All tests use synthetic data — no `.seq` file is required.

---

## Troubleshooting

### Join misses (`unmatched` rows in `metadata_join_report.csv`)

1. Confirm the tracking filename matches the LUT `Video_name_SEQ` pattern: `MM-DD-YY_{ID}_{Lane}_{ID}_{Lane}_<suffix>`.
2. Check the `Lane` column in the LUT (`F`/`B`) and that the tracking file has `_Front` or `_Back`.
3. Manually add/correct a row in the LUT if the seq filename is genuinely incorrect — flag any such cases to Adam.

### Bouts don't match expectations

Open `bouts/qc_plots/<stem>_bouts_diagnostic.png`. The red dashed threshold lines show exactly where the cuts are. Then edit `analysis_config.json` and re-run `compute_bouts.py --overwrite`:

- **Too many bouts / short rest-blips detected:** increase `stationary_dispersion_threshold_px` or `min_bout_duration_sec`.
- **Genuine bouts being split:** increase `max_gap_merge_sec`.
- **Dispersion noisy during confirmed rest:** increase `dispersion_window_samples` (try 7 or 9).
- **Velocity guard incorrectly excluding frames:** increase `stationary_velocity_threshold_px_per_s`.

### Model fallback to descriptives (PRELIMINARY warning)

Expected on the small test subset. The script prints the specific reason:
- Too few unique animals → more recordings needed.
- A factor has only one level → that grouping variable isn't populated for the available files.

The same commands automatically fit LMMs on the full dataset — no code changes required.

### Mouse entry detection fails

If the pipeline can't find when the mouse enters the arena, pass the frame number manually:

```bash
python scripts/track_temperatures.py \
    --input "..." \
    --tracking-start-frame 450
```

---

## Project structure

```
thermal-gradient-tracker/
  README.md
  requirements.txt
  tracking_config.json         tracking + segmentation parameters + arena polygon
  tracking_config.example.json starting template
  analysis_config.json         bout detection + analysis parameters
  metadata/
    LUT_CLEAN_July6.csv        experimental metadata LUT (place here)
  scripts/
    preview_tracking_input.py    inspect .seq file, save frame previews
    create_arena_mask.py         interactive polygon mask definition
    select_training_frames.py    sample frames for manual labeling
    train_mouse_detector.py      tune segmentation from labeled frames
    track_temperatures.py        tracking pipeline (single file)
    batch_track_temperatures.py  batch version
    compute_bouts.py             velocity + bout detection (single file)
    batch_compute_bouts.py       batch version
    join_metadata.py             join tracking files to LUT, write master table
    analyze_treatment_effects.py fit models + generate all plots and tables
  src/
    paths.py                     single source of truth for all project-root-relative directories
    seq_io.py                    FLIR .seq reader
    arena_mask.py                arena mask + TrackingConfig
    mouse_segmentation.py        background model + mouse detection
    tracking.py                  frame-by-frame tracking pipeline
    roi_geometry.py              circular ROI geometry
    temperature_extraction.py    ROI statistics extraction
    qc_outputs.py                overlay images, summary CSV, diagnostic plots
    batch.py                     batch file discovery
    logging_utils.py             terminal logging + progress bar
    analysis_config.py           AnalysisConfig pydantic model
    velocity.py                  centroid velocity + rolling dispersion
    bouts.py                     RLE bout detection and per-bout metrics
    bout_qc.py                   bout diagnostic figures
    metadata.py                  LUT loading, cleaning, and join logic
    stats_models.py              linear mixed models with graceful fallback
    treatment_plots.py           all treatment-effect plots
  tests/
    test_roi_geometry.py
    test_temperature_extraction.py
    test_floor_roi_selection.py
    test_velocity.py
    test_bouts_rle.py
    test_metadata_join.py
  trackingOutputs/               tracking CSVs and QC outputs (created by tracking scripts)
  bouts/                         bout detection and analysis outputs (created by bout scripts)
```
