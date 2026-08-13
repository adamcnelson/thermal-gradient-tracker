# Stage 0 — Radiometric Data-Path Audit

Date: 2026-08-12. Scope: project_brief_v7.md §6 Stage 0 (blocking gate for M1).

## 1. Raw vs. render — resolved, no code needed

The pipeline already reads raw per-pixel radiometric data, not display-normalized
frames. `tracking.py` calls `seq_io.read_planck_constants()` (via `exiftool`) once
per file and passes the result into `seq_io.raw_to_celsius()`, which applies the
real FLIR Planck equation. This predates v7. The only gap was test coverage:
`tests/test_seq_io_radiometric.py` now pins the Planck-equation math against a
manually-computed expected value, and — against a real `.seq` file — confirms
`exiftool` actually resolves constants rather than silently hitting the linear
fallback (`raw * 0.04 - 273.15`), which `tracking.py` only flags with a log
warning that's easy to miss in a batch run.

**Caveat:** this is formula-correctness + real-file-resolution testing, not
calibration validation against a physical reference target — no such
ground-truth-temperature footage exists for this rig.

## 2. Acquisition settings — confirmed as-set, not independently verified

Read via `exiftool` from 3 real cropped session files (`07-28-25 Test_3 Front`,
`07-28-25 Test_3 Back`, `08-07-25 Test_7 Front`) — identical across all three:

| Field | Value |
|---|---|
| Emissivity | 0.95 |
| Reflected Apparent Temperature | 20.0 °C |
| Atmospheric Temperature | 20.0 °C |
| IR Window Transmission | 1.00 |
| Object Distance | 1.00 m |
| Relative Humidity | 50.0 % |

These are plausible values (0.95 is a standard rodent fur/skin emissivity
estimate) but I have no independent measurement to confirm they're correct —
this is brief §11 open parameter 3, still open. Flagging as "confirmed what was
set," per the brief's own Stage 0 wording, not "confirmed correct."

## 3. Cross-cutting finding, NOT scoped to v7 — needs Adam's decision

**`camera_fps` in `tracking_config.json` is hardcoded to `10.0`, but the real
`.seq` files' own embedded FLIR metadata reports `FrameRate: 8`** — confirmed
via `exiftool -FrameRate` on all 3 sampled files, consistent every time.

Corroborating check: `07-28-25_Test_3_..._Front.seq` has 19,141 frames.
At 8 fps that's ~39.9 min; at 10 fps, ~31.9 min. Neither matches
`analysis_config.json`'s `trial_length_reference_sec` (3000 s = 50 min)
exactly, but 8 fps is closer to a plausible real trial length than 10 fps is.

This matters because `elapsed_time_sec = frame_idx / config.camera_fps` is the
single source of truth for time throughout the *existing* pipeline —
`BoutsConfig.frame_rate_fps`'s own docstring says bout timing is derived
entirely from it. If the true rate is 8 fps:

- every already-processed session's `elapsed_time_sec` (and everything
  downstream: bout durations, `min_bout_duration_sec`/`max_gap_merge_sec`
  thresholds, velocity, all `r_analysis/` timecourse plots) is compressed by a
  factor of 0.8 — the pipeline believes 20% less time has passed than actually
  did.
- this predates v7 entirely and is not something I've touched or would
  silently fix — surfacing it here because Stage 0 is where it was found.

**Recommend:** confirm the true acquisition frame rate against ResearchIR's own
recording settings (not just this file's embedded tag) before anything —
v7 or otherwise — depends on `camera_fps` being correct.

## 4. Camouflage-band ΔT across gradient zones

Brief asked for ~6 hand-exported radiometric frames (2 per zone). Used the full
existing real corpus instead — 6 already-tracked sessions, 10,161 valid frames
(`qc_flag == "ok"`, both ROIs valid) — since that data already exists and gives
a distribution instead of 6 points. `mouse_centroid_x` (0–416 px crop width)
bins into thirds; floor temperature decreases monotonically with x, so low-x =
hot end.

| Zone | n | Floor °C (mean) | Mouse °C (mean) | ΔT mean | ΔT median | % negative |
|---|---|---|---|---|---|---|
| Hot end (low x) | 2135 | 36.4 | 31.4 | **−4.98** | −3.98 | 89.9% |
| Mid | 6528 | 23.9 | 30.1 | +6.11 | +6.30 | 0.2% |
| Cool end (high x) | 1498 | 13.4 | 28.5 | +15.04 | +14.79 | 0.0% |

The hot-end sign flip (89.9% of frames read the mouse *cooler* than floor)
directly confirms the brief's §3 predicted polarity inversion — this isn't
hypothetical, it's already in the existing corpus.

Finer resolution (10 bins) locates the crossover: floor temp ~28–33 °C is
where ΔT sign flips, and |ΔT| narrows to ~2.4–2.5 °C average in that band —
closer to the brief's "~2 °C" estimate than "~0.3 °C." It does not collapse to
near-zero at the bin level because each bin averages frames straddling the
exact crossover point.

**Caveat — likely a lower bound, not an upper bound on dead-zone width:** this
measurement is conditioned on the *old thermal-only* detector having
successfully found the mouse at all. The frames where thermal detection failed
outright (exactly the failure mode §3 exists to fix) produce no row and aren't
in this table — so the true camouflage-driven yield loss is probably worse
than what's visible here. A direct yield-by-zone check (fraction of frames
with *any* valid detection, not just ΔT among valid ones) would need
interpolated position for undetected frames and is a natural Stage 5 bake-off
input rather than something done here.

## Stage 0 gate status

Raw-vs-render: **pass** (already was). Camouflage table: **produced**, feeds
Stage 6 gating thresholds. Emissivity: **confirmed as-set**, not yet confirmed
correct (open parameter, brief §11.3). `camera_fps`: **new finding, blocking
question for Adam** — not part of Stage 0's original scope but found while
auditing the radiometric path, and it's more consequential than anything v7
introduces since it affects data already analyzed.
