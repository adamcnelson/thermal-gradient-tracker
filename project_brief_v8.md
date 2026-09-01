# Project Brief v8 — RGB+Thermal Feature Analysis

## 1. Where this picks up

v7 built a Stage 5-7 pipeline that uses RGB tracking to refine *which* thermal
pixels get measured — instead of one whole-mouse ROI (the pre-RGB, strictly-
thermal pipeline's approach), it decomposes the animal into **dorsal surface**,
**warm spot** (anterior/interscapular region), and **base-of-tail ΔT**, for
both Front and Back lanes of all three sessions (Test_3/4/7 — 6
session×lane combinations). Real, load-bearing findings from that work,
already in project memory, that this brief assumes:

- `qc_valid` gates **only** the tail-ΔT measurement — dorsal/warm-spot have
  their own applicability (posture-dependent), not the same gate.
- Warm-spot requires "extended" posture, which is architecturally fragile
  (a real skeleton-topology limit, not a bug — see
  `project_v7_extended_posture_skeleton_fragility`) — coverage is genuinely
  sparse (5-68% of aspect-qualified frames depending on session) and won't
  improve without a bigger redesign (DeepLabCut is a live, paused option —
  see `project_v7_deeplabcut_consideration`). **Treat warm-spot results as
  lower-powered than dorsal/tail; report n per cell, don't paper over it.**
- Tail-specific RGB/thermal misalignment was investigated and dropped (three
  real correction attempts, none reliable) — tail-ΔT uses the whole-body
  registration correction only, no tail-specific fix.
- Whole-body RGB→thermal registration has two corrections already applied
  per sample: a centroid translation, then a small bounded local edge
  refinement. Both are already baked into every `*_bout_output.csv` /
  `*_frame_output.csv` row — nothing further needed to use this data.

This brief's job is the **next phase: analysis**, not more tracking-pipeline
work, except for the two concrete prerequisites in §3.

## 2. Current pipeline output (what's usable today)

Per session×lane (`scripts/stage7_real_run.py`), in `landmark_outputs/`:

- **`{session}_{track}_bout_output.csv`** — one row per stationary bout.
  Key columns: `mouse_id, track, bout_id, bout_start_thermal_sec,
  bout_end_thermal_sec, gradient_zone, mean_floor_temp_c, warm_spot_temp_c,
  dorsal_mean_c, dorsal_median_c, tail_delta_t_c, qc_valid, n_frames_averaged`.
- **`{session}_{track}_frame_output.csv`** — one row per measured sample
  (currently: 3 fractional positions per stationary bout only — see §3.1).
  Key columns: `elapsed_time_thermal_sec, mouse_surface_temp_mean_c,
  floor_temp_mean_c, posture, qc_flag`.

Neither file has `virus`/`injection`/`phase` joined in yet — see §3.2.

## 3. Prerequisite code work

### 3.1 Non-stationary sampling (real gap, needed before §4 can run in full)

Stage 7 currently only samples inside stationary bouts
(`iter_stationary_bouts` + 3 fractional positions per bout). The legacy
pipeline measured every sampled frame regardless of `stationary` state, which
is what let its analysis compare rest vs. non-rest. To match that, Stage 7
needs a **non-stationary sampling mode**: run the same per-sample computation
(segmentation → registration → dorsal/warm-spot/tail) at a fixed sample rate
across `stationary == False` stretches too.

- Reuse the same per-sample function `qc_shared.compute_candidates()` /
  the inner loop of `stage7_real_run.py`'s `process_session()` — don't
  reimplement the measurement logic, just the sampling schedule.
- Start at a modest, fixed rate (e.g. the ~1 Hz stride `compute_rgb_track.py`
  already uses for full-session tracks), not exhaustive per-frame — this is
  materially more compute than today's 330 total samples across 6
  session×lanes; validate the rate is tractable locally before scaling.
- Expect posture yield to be *worse* during non-stationary frames (motion
  blur, transient postures) — don't assume today's extended/curled/ambiguous
  breakdown carries over; check it.
- Add a `stationary` boolean column to `frame_output.csv` (mirrors the legacy
  master table) so R can filter/split by it directly.

### 3.2 Metadata join for the new schema

`bout_output.csv`/`frame_output.csv` have `mouse_id`+`session`+`track` but
not `virus`/`injection`/`phase`. `src/metadata.py::join_metadata()` already
does this LUT join for the legacy pipeline's `video_file`-keyed CSVs
(`resolve_lut_row()`, `load_lut()`) — adapt it (or add a thin new entry
point) for `session`+`track` as the key instead of a raw `video_file` string;
the LUT-matching logic itself doesn't need to change, only how the join key
is derived. Produces the new-pipeline equivalent of
`master_tracking_with_metadata.csv` — call it
`master_landmarks_with_metadata.csv` (bout-level and frame-level versions),
same convention as the legacy output.

## 4. Analysis plan (centerpiece)

**Question**: does oxytocin-neuron activation (Gq) or inhibition (Gi) via
DREADDs (DCZ vs. Vehicle injection) affect (a) the mouse's own thermal
features and (b) its thermal preference (where on the gradient it rests)?

This is a direct extension of the legacy pipeline's analysis
(`SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/bouts/analysis_plots`,
`paired_dcz_vehicle_*.png`, and its retrospective spec, `project_brief_v6.md`
— not currently in the repo, only at
`/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/project_brief_v6.md`;
read it directly, don't rely on this summary). Same structure, generalized
from one outcome (`mouse_surface_temp_mean`) to three
(`dorsal_mean_c`, `warm_spot_temp_c`, `tail_delta_t_c`), each split by
stationary/non-stationary where the legacy version split by rest/non-rest.
Metadata factors are unchanged: `virus` (Gi/Gq/none), `injection`
(DCZ/Vehicle/Saline), `phase`, `craniotomy` (Pre/Post, LUT-joined — see
`r_analysis/R/data.R::join_craniotomy()` for the exact pattern to reuse),
`mouse_id`, `track` (Front/Back — analogous to legacy `lane`).

Preference outcomes (unchanged from legacy, already present in the new
schema): `mean_floor_temp_c` (bout-level — "what temperature did the mouse
choose to rest at"), `gradient_zone`.

| # | Plot family | Legacy source | New-pipeline generalization |
|---|---|---|---|
| 4.1 | Time-course | v6 §3.1 | Per virus, per outcome (dorsal/warm-spot/tail-ΔT + floor pref), pre/post-craniotomy, DCZ vs Vehicle mean±SEM trace over `elapsed_time_thermal_sec`. Needs §3.1's frame-level non-stationary data. |
| 4.2.1 | Paired spaghetti (reproduce) | v6 §3.2.1, `paired_dcz_vehicle_*.png` | Per-mouse DCZ-vs-Vehicle paired points + group mean±SE, per virus, per outcome — same visual structure, 3 outcomes instead of 1 (+ floor pref). |
| 4.2.2 | Craniotomy effect | v6 §3.2.2 | Combined Gi+Gq, stationary vs. non-stationary split, pre/post-craniotomy. |
| 4.2.3 | Bout organization | v6 §3.2.3 | Post-craniotomy only, DCZ vs Vehicle: bout count, bout duration — `bout_output.csv` already has everything needed, no new join beyond §3.2. |
| 4.3 | Distributions | v6 §3.3 | Per virus, DCZ vs Vehicle, split stationary/non-stationary. Velocity isn't a native Stage 7 output column, but `velocity_smooth_px_s` is available the same way it always was — `src/velocity.py::compute_velocity()` on the underlying `trackingOutputs/*_tracking_every10frames.csv` (already read by `stage7_real_run.py`) — join it in by timestamp rather than re-deriving it; not split by stationary/non-stationary, matching legacy. |
| 4.4 | **New: time-series model** | — | Formal statistical layer beyond descriptive time-course traces (4.1): per outcome, per virus, fit `outcome ~ elapsed_time_thermal_sec * injection + (1 + elapsed_time_thermal_sec | mouse_id)` (or a GAMM smooth-by-injection term if the trajectory isn't linear) to test whether the DCZ/Vehicle trajectories genuinely diverge over the trial, not just their marginal means. Report the time×injection interaction. |

Filtering conventions to carry over exactly: `qc_flag == "ok"` (frame-level,
legacy) → use `qc_valid` for tail-ΔT specifically, but do **not** apply it to
dorsal/warm-spot (§1 — different applicability). Exclude rehabituation the
same way the legacy analysis does (§3.2 of v6: restrict post-craniotomy to
`injection %in% c("DCZ","Vehicle")`, not a `craniotomy`/`Saline` special
case).

## 5. R analysis: new `r_analysis_RGB_Thermal/` directory

Parallel structure to `r_analysis/` (don't modify that directory — it stays
as the legacy pipeline's analysis, per §6):

```
r_analysis_RGB_Thermal/
  .Rprofile, renv.lock, renv/        # own renv, don't share r_analysis/'s
  R/data.R                           # load_master(), load_bouts()-equivalents
                                      # for master_landmarks_with_metadata.csv;
                                      # reuse join_craniotomy() logic/pattern
  R/plot_*.R                         # one file per plot family, mirroring
                                      # r_analysis/R/'s naming
  R/model_timeseries.R               # new — §4.4's LMM/GAMM fits
  scripts/NN_*.R                     # numbered, one per plot family, mirrors
                                      # r_analysis/scripts/ convention exactly
  README.md                          # what it reads/produces/how to run —
                                      # copy r_analysis/README.md's structure
  output/                            # gitignored (already added to .gitignore)
```

Read `r_analysis/R/data.R` and `r_analysis/README.md` before writing anything
— the Craniotomy-join pattern, the `qc_flag`-filtering convention, and the
per-mouse-then-per-group time-binning approach (`r_analysis/README.md`'s
"open question D" resolution) all transfer directly to the new outcomes.

## 6. Deployment: local → MedicineBow SLURM

`slurm/run_thermal_gradient.sbatch` and `slurm/DEPLOY_MEDICINEBOW.md` stay as
they are — the legacy pipeline must remain runnable exactly as documented
(4 stages: `batch_track_temperatures.py` → `batch_compute_bouts.py` →
`join_metadata.py` → `analyze_treatment_effects.py`). Add the new pipeline's
stages **alongside**, not in place of, the existing ones — e.g. new sbatch
sections invoking `stage7_real_run.py` and the §3.2 metadata join, gated so a
single script/job can run either pipeline (or both). Read the existing
sbatch script's structure first; don't restructure it to add this — append.

Standard order, same as every prior stage this project has shipped:
1. Build/test §3.1-3.2 and the R analysis locally on the 6 existing
   session×lanes.
2. QC review (same pattern as all of v7 — real images, real numbers, Adam
   reviews before scaling).
3. Extend the sbatch script, smoke-test on MedicineBow, then submit the full
   job over the whole dataset.

## 7. Best practices for launching from Claude Desktop

Real, verified differences worth planning around (verify current Desktop
capabilities early next session rather than assuming — this evolves):

- **Image-heavy review is Desktop's real advantage for this project.** Nearly
  every QC decision this session came from looking at PNGs — in the terminal
  CLI that means one `Read` tool call per image. Desktop's chat handles
  pasted/dragged images natively and inline, which should make the §4 plot
  review loop (a lot of plots, this is now a 3-outcome × multiple-plot-family
  analysis) meaningfully faster.
- **Memory and project docs load the same way regardless of interface** — the
  auto-memory system and this brief aren't terminal-specific. Start the next
  session by pointing at `project_brief_v8.md` directly rather than
  re-explaining context.
- **Verify agentic/background-task parity before relying on it.** This
  session used long-running background shell commands constantly (thermal
  `.seq` processing, batch renders taking minutes) with task notifications
  when they finished. Confirm Desktop's Claude Code integration supports the
  same background-run + notification pattern before assuming it does —
  if it doesn't, long pipeline runs (§3.1's non-stationary sampling
  especially, and anything on MedicineBow) may be better kept in the
  terminal CLI or actual `sbatch` jobs, with Desktop used for planning,
  code review, and the R/plot review loop specifically.
- **If Desktop offers a "Projects" feature**, point it at the repo root so
  file context persists across turns the way the CLI's working directory
  does — check whether it needs the same explicit file-reading discipline
  (verify before trusting) as this session, or handles it differently.
