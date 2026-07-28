# project_brief_v5 — Fix disappearing-mouse tracking (mouse≈floor temp) + auto-managed directories

## 0. Orientation for Claude Code

You are launched from inside the self-contained `thermal-gradient-tracker` folder on Adam's desktop — the same folder where all prior coding sessions took place. **Before changing anything, review the code as it currently stands:** read `README.md`, walk `scripts/` and `src/`, and build an accurate model of the real tracking pipeline (`track_temperatures.py` / `batch_track_temperatures.py`), the current bout-detection code, the config files, and how outputs are written today. Write a short summary of the current tracking + bout logic before proposing changes.

Ignore the retired interactive/manual-QC steps (arena masking, training-frame selection/labeling, manual annotation) — they are no longer in use and are out of scope.

This brief has **two independent issues**. Do Issue 1 (tracking + bouts) and Issue 2 (directories) as separate workstreams; Issue 2 can proceed without waiting on Issue 1's design sign-off.

---

## 1. Issue 1 — Mice "disappear" when their surface temperature matches the floor

### The problem

When a mouse rests on the thermal gradient at a spot whose temperature matches its own dorsal surface temperature, it blends into the background and the tracker loses it. In these moments we lose exactly the data we care about: **rest bouts at or near the animal's own surface temperature**.

Concrete example — QC plot on Adam's machine:
`bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Front_bouts_diagnostic.png`
Floor temp and mouse temp are both ~30 °C. The centroid panel shows a long, flat, horizontal run of datapoints — a clear sustained period of inactivity — yet **no bout is called** there. Fixing this means (a) not losing the animal during these low-contrast rest periods, and (b) correctly calling the resulting bout.

### Lucas's contribution — study it, treat it as the reference

Lucas (lab member) has written code that better captures these instances. His QC plots show his method recovers the mouse and the bout where the current pipeline fails. **His code is in the repo at `LucasCode/`** (i.e. `thermal-gradient-tracker/LucasCode/`) — read it there. Treat it as reference material to study and adapt, not as files to run in place; note that his scripts assume the project root is one level up (`Path(__file__).resolve().parents[1]`), so account for the `LucasCode/` nesting when reconciling his `src.seq_io` import and paths. His two files:

- **`track_blob.py`** — a background-subtraction tracker. Key ideas worth adopting:
  - **Two separate backgrounds.** A *tracking* background from the first ~100 frames (`build_initial_background`) — before the mouse is dropped in, so early frames correctly read as empty — and a *temperature* background sampled evenly across the whole video (`build_sampled_background`). Keeping these distinct is deliberate and important.
  - **Local fallback recovery** (`local_fallback_blob`). When the global threshold loses the mouse, it re-searches a small window around the last known centroid at a *lower* threshold. This is the core mechanism that recovers the animal during the mouse≈floor collapse — locally, a faint difference still exists even when the global threshold misses it. This is the heart of the fix.
  - **Continuity-aware blob choice** (`choose_blob_with_continuity`) — prefers the blob nearest the previous centroid over merely the largest, reducing jumps between candidates.
- **`analyze_rest_bouts.py`** — bout detection on *smoothed* signals: rolling-median smoothing of x-position and velocity (`rolling_median`), then a bout requires **both** low velocity **and** a bounded x-range (`x_range_threshold`) sustained for a minimum duration (`find_stationary_bouts`). This dual criterion is what catches the flat run the current pipeline misses.

### How to integrate — decision approach (do NOT commit before Adam signs off)

Per Adam's direction: **study both his current pipeline and Lucas's code, recommend the best path, defer to Lucas's approach as the reference where they diverge, and ask Adam before committing to an architecture.** Concretely:

1. **Compare the two tracking strategies** and write up, in a few paragraphs: where the current pipeline loses the mouse, why Lucas's background-subtraction + local-fallback recovers it, and what each does for temperature extraction. Note that Lucas's method computes both mouse Tb and the underlying location temperature from the same mask (`calculate_temperatures`) — check this against the current pipeline's temperature logic and flag any differences in method or units.
2. **Recommend an integration architecture** and present it to Adam as options with a clear recommendation, e.g.:
   - (a) Adopt Lucas's tracker as the new default tracking path, refactored into `src/` and wired into `batch_track_temperatures.py`; or
   - (b) Keep the current tracker as primary and graft Lucas's `local_fallback_blob` in as a recovery stage triggered on detection loss / low mouse-floor contrast; or
   - (c) A hybrid — Lucas's dual-background + continuity choice as the tracker, current pipeline's validated temperature-calibration path retained.
   Recommend one, defer to Lucas's method on genuine tracking-strategy conflicts, and **state what you'd change and why before writing it.**
3. **Fix bouts in the same workstream** (Adam confirmed tracking + bouts are one scope). Bring Lucas's smoothed-velocity + bounded-x-range bout detection into the pipeline's bout stage so the ~30 °C flat-run example is correctly called. Re-generate the diagnostic for `07-28-25_4540_B_4541_F_Test3-004_Front` and confirm the bout now appears.

### Issues in Lucas's code to raise, not silently absorb

Lucas's code is a research draft. During review, surface these to Adam rather than quietly "fixing" them, since they change behavior:

- **Continuity filter likely inverted.** In `choose_blob_with_continuity`, the "close candidates" filter uses `c["distance"] >= max_jump` (line ~328). To *prefer nearby* blobs this should almost certainly be `<= max_jump`. Flag it; don't assume.
- **Hardcoded temperature calibration.** `raw_to_celsius` hardcodes a quadratic polynomial and Lucas notes it was lifted from `temperature_extraction.py`. Reconcile against the current pipeline's calibration (and the `exiftool`/Planck path in the README). Do **not** regress temperature accuracy to adopt his tracker — keep the validated calibration.
- **Hardcoded magic numbers.** Threshold percentile (96.5), `edge_margin` (20), fallback `search_radius`/`threshold_percentile`/`min_area`, background frame counts, bout thresholds (velocity 3.0 px/frame, x-range 15 px, min duration 20 s, fps 10). These should become config parameters (see `tracking_config.json` / `analysis_config.json`), not buried constants — especially `fps`, which must match the real acquisition rate.
- **A commented-out jump-rejection block** and some obvious typos (`x1 = min(prev_x - search_radius +1, width)` looks wrong; several print/variable typos). Note them; propose corrections with the rationale.

Ask Adam questions wherever the "best" approach is genuinely uncertain rather than guessing.

---

## 2. Issue 2 — README ↔ execution mismatch on input/output directories

### The problem

The README currently tells the user to substitute paths in many places:

> Step 1 preview `--input`, Step 2 arena mask `--input`, Step 3 track one file `--input`, Step 5 manual training `--input`/`--seq`, Step 6 batch `--input-dir` … plus a note that the `batch_track_temperatures.py` docstring's `"/path/to/croppedSeqFiles"` is documentation only.

**That is not what Adam wants.** The desired behavior:

- **The only path a user ever specifies manually is the `croppedSeqFiles` input folder.**
- **Every other directory is created automatically inside the `thermal-gradient-tracker` folder**, using the existing local layout as the canonical structure.
- The retired manual steps (arena mask, training) are out of scope — don't preserve their path-substitution instructions.

### Canonical directory layout (from Adam's local machine)

All of these are auto-created relative to the project root; none should require user path edits:

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

### What to change

1. **Code:** Introduce a single source of truth for project paths — e.g. a small `paths.py` (or a `project_root` resolver) in `src/` that computes the project root from the file location (Lucas already uses this pattern: `PROJECT_ROOT = Path(__file__).resolve().parents[1]`) and defines every output directory relative to it, `mkdir(parents=True, exist_ok=True)` on use. Refactor the automated scripts (`batch_track_temperatures.py`, the bout stage, the metadata join, the analysis) to import these paths instead of taking `--output-dir` / building paths ad hoc. Keep `--input` / `--input-dir` (the croppedSeqFiles location) as the **one** user-supplied argument. Per-video subfolders under `trackingOutputs/qc_images/` should be derived automatically from each `.seq` filename.
2. **README:** Delete the path-substitution table's now-obsolete rows and the docstring note. Replace with a short, plain statement: the user sets only the `croppedSeqFiles` input path (one argument, or one config field); everything else is created automatically in the layout above. Show the resulting tree so users know what to expect.
3. **Config:** If a config field is the cleaner way to supply the input path (vs. a CLI arg), pick one convention and make it consistent across all automated scripts. Recommend which and say why.

### Portability to ARCC / MedicineBow (ties to ProjectBrief_v4)

The auto-created layout must be **relative to the project root, not absolute** — so the exact same structure appears whether the repo lives on Adam's iMac or on MedicineBow `/project`. The only thing that differs between environments is the `croppedSeqFiles` input location (a local folder on the desktop; on the cluster, the Alcova path `/cluster/alcova/bedfordlab/...`). Do not hardcode `/Users/adamnelson/...` anywhere. This keeps v5 consistent with ProjectBrief_v4's rule that inputs are read from Alcova and outputs stay on MedicineBow under the project tree.

---

## 3. Working order for this session

1. Review the current repo; write the short summary of current tracking + bout logic (Section 0). Also read Lucas's files in `LucasCode/` before designing the Issue 1 integration.
2. **Issue 2 first** (lower-risk, unblocks everyone): implement the `paths.py`/root-resolver refactor, auto-create the canonical tree, update the README and config, verify no absolute paths remain. Confirm the automated scripts still run on the local test `.seq` files with only the input path supplied.
3. **Issue 1**: write the two-strategy comparison, then present the integration recommendation (deferring to Lucas's approach on tracking-strategy conflicts) **and pause for Adam's sign-off before implementing.** After sign-off, implement tracking + bouts together and regenerate the ~30 °C diagnostic to confirm the missed bout is now called.
4. List open questions for Adam — at minimum: the continuity `>=` vs `<=` filter, which temperature calibration to keep, the correct `fps`, and CLI-arg vs config-field for the input path.

Keep changes reviewable and explained. Where Lucas's code and the current pipeline genuinely conflict on approach, favor Lucas's method (per Adam) — but surface the tradeoff rather than burying it.
