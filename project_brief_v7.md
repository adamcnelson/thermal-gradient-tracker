# Project Brief v7 — Feature-Specific Thermal Tracking

**Project:** ThermalGradient
**Author:** Adam Nelson
**Date:** 2026-08-12
**Status:** Draft — two parameters pending confirmation (see §11)

---

## 1. Objective

Extend the existing pipeline from whole-animal dorsal-surface tracking to **landmark-specific thermal measurement**, yielding two physiological readouts:

- **Warm spot** — the consistently warmest anterior surface region, spanning the top of the head and the interscapular space. A proxy for a bodily region that is reliably warmest, *not* a calibrated measure of BAT thermogenesis. This framing is deliberate; see §6.
- **Tail-base thermal index** — a vasomotor readout, reported as ΔT above local floor temperature.

Secondary landmarks (head, trunk/rump) are extracted where available but are not primary endpoints.

## 2. Non-goals

- Frame-by-frame thermal features across the whole session. Not achievable; not attempted.
- Absolute calibrated tail temperature. The tail is at or below the resolution limit (§11).
- Any modification to existing tracking or bout-detection behavior. v7 is strictly additive.
- Real-time or online processing.

## 3. Core architectural principle

**Geometry comes from RGB. Radiometry comes from thermal. The two never mix roles.**

The thermal stream is demoted from *detector* to *sampler*: it is only ever asked "what is the temperature at these coordinates," never "where is the mouse."

This is not a convenience. Thermal detection failure is a monotonic function of floor temperature — the animal is high-contrast at the cool end, vanishes at intermediate temperatures, and **inverts polarity at the hot end** (insulating fur reads cooler than a ~40 °C plate). Detection yield would therefore correlate with the independent variable, biasing every downstream comparison. Because tail vasomotor tone is most interesting precisely where thermal contrast collapses, RGB-derived geometry is a requirement, not an enhancement.

**RGB remains an optional input.** Any session that fails registration or synchronization is flagged and falls back to thermal-only whole-animal output. The pipeline must never hard-depend on the webcam stream.

## 4. Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Thermal sensor | 464 × 348 px, 10 fps | Landmarks are a few px; temporal averaging is cheap |
| Thermal data | SEQ, per-pixel temperature | **Must read raw; display colormap is dynamic and decoupled from radiance** |
| Webcam | ~20–30 fps, ~500 MB / 30 min, two tracks per frame (B top, F bottom) | Must be cropped per track before use |
| Dev hardware | Intel i7 8-core, AMD Radeon Pro 5500 XT | No CUDA, no reliable MPS → **CPU-only for local work** |
| Corpus | ~100 experiments (1 mouse × 1 track each) | ~50 GB webcam data; batch processing must be unattended |
| Ground truth budget | 75 hand-labeled frames | All allocated to validation, none to training |
| Rig | Fixed but occasionally nudged | Homography must be estimated **per session**, with within-session drift detection |

## 5. Git and reversion strategy

1. Tag current HEAD as `v6-stable` **before any v7 work begins**.
2. All work on branch `feature/v7-landmarks`.
3. New code lives in a new subpackage (`thermalgradient/landmarks/`). No edits to existing modules except a **single opt-in hook**, gated by a config flag defaulting to `false`.
4. **Golden-output regression test:** run the existing pipeline on a fixture session, store outputs, and assert byte-or-tolerance equality after the v7 merge. This is what proves standing behavior survived; absence of complaints is not proof.
5. Heavy dependencies (torch, any deep model) go in an optional extras group so the core pipeline still installs clean on ARCC without them.

Reversion is then either flipping the config flag or `git checkout v6-stable`.

## 6. Stage plan

### Stage 0 — Radiometric data-path audit *(blocking)*

Determine whether the current pipeline reads per-pixel temperature from SEQ or operates on rendered/colorized frames. FLIR's display scaling evolves with the global temperature profile, so any measurement derived from rendered frames is uninterpretable.

If the current path is render-based, build a raw SEQ reader (`flirpy`, FLIR Science File SDK, or equivalent) that returns calibrated per-pixel °C plus per-frame object parameters (emissivity, reflected apparent temperature, distance). Because raw counts are retained, **emissivity can be corrected retroactively** if acquisition settings were wrong — verify what ε was set to.

**Also under Stage 0 — measure the camouflage band.** The magnitude of the animal-vs-floor ΔT at intermediate floor temperatures is currently *unknown*. The available stills are display-normalized, so their apparent contrast is partly an artifact of FLIR's dynamic scaling and cannot answer this. Export ~6 radiometric frames — 2 cool-end, 2 mid-gradient camouflage, 2 hot-end — and measure the actual ΔT and its sign at each. This determines whether the camouflage dead zone is ~0.3 °C or ~2 °C wide, which sets the gating thresholds in Stage 6 and the expected yield floor in Stage 5. Do not tune thresholds before this number exists.

*Deliverable:* audit note + `read_seq_radiometric()` with a unit test against a frame of known temperature + a short table of measured animal-vs-floor ΔT across the three gradient zones.

### Stage 1 — Stationary-bout gating

**All landmark measurement is restricted to stationary bouts.** This single constraint:

- eliminates motion blur, which is severe in both streams;
- eliminates thermal smearing during locomotion;
- yields surface temperatures near steady state rather than mid-transient — the physiologically correct quantity;
- relaxes sync tolerance by orders of magnitude (a 500 ms error costs almost nothing mid-bout, and is fatal mid-run);
- reduces inference volume ~50–100× by sampling at 1–2 Hz instead of full rate.

Feature extraction becomes an **annotation layer on the existing bout output**, not a parallel per-frame pipeline. Reuse the current bout detector as-is.

### Stage 2 — Webcam preprocessing

Split each webcam file into per-track crops (B = upper, F = lower), joined to `LUT_CLEAN_July6.csv` for session/track/mouseID. Verify the B/F row assignment empirically on the first session rather than trusting the convention.

### Stage 2b — Nestlet handling

A nestlet is present in some but not all trials; presence is a binary column in `LUT_CLEAN_July6.csv`. The mice do not build nests with it, so it is effectively a static insulating object sitting on the plate. It must be handled explicitly in three ways.

**As an artifact source.** The nestlet is a low-conductivity, high-emissivity object thermally decoupled from the plate beneath it. Its surface temperature does not follow the gradient, so it appears as a thermal anomaly that violates the pipeline's core background assumption — that local floor temperature is a smooth function of position. Two consequences:

- **Local-floor annuli must exclude nestlet pixels.** A background annulus that straddles the nestlet edge returns a meaningless reference and silently corrupts every ΔT derived from it. This is the most dangerous nestlet failure mode because it produces plausible-looking numbers.
- **Thermal ghosts.** A nestlet an animal has rested on retains body heat and presents a warm, roughly mouse-sized patch that persists after the animal leaves. Any thermal-domain blob detection will find it. Reject by requiring coincidence with the RGB-derived animal mask — which the §3 architecture already provides for free, and is a further argument for it.

**Detection.** Segment the nestlet in RGB as a static, high-albedo, non-animal object via the temporal background model; it is compact and geometrically distinct from the plate. Use the LUT binary as ground truth for whether to expect one, and flag any disagreement between LUT and detection as a metadata error worth investigating. Track the nestlet mask over time: mice displace it occasionally, which breaks the background model and generates a transient false blob until it re-adapts. Log displacement events.

**As a registration fiducial (opportunistic).** The nestlet is visible in *both* modalities and is compact in two dimensions — which makes it a considerably better homography constraint than the track edges, since long parallel lines constrain the transform poorly along their own direction. Where a nestlet is present, add it as a correspondence to the Stage 4 fit and expect a lower reprojection residual. It is present in only a subset of sessions, so it refines the fit; it cannot be the primary basis for it.

**As a covariate.** Nestlet presence plausibly perturbs both floor temperature locally and the animal's position preference (it is a thermal refuge whether or not it is a nest). Carry `nestlet_present` through to every output table, add a per-frame `on_nestlet` state, and **check whether nestlet presence is balanced across treatment groups** before any chemogenetic comparison. If it is confounded with group, that needs to be known now rather than at review.

A mouse resting *on* the nestlet is also thermally insulated from the plate, so its dorsal surface reflects different heat exchange than the same animal at the same gradient position on bare plate. Treat `on_nestlet` as a state variable in analysis, not as a nuisance flag to filter away — those frames are interpretable, just not directly comparable.

### Stage 3 — Temporal synchronization

- Compute a motion-energy trace per stream. Use **centroid speed from the existing tracker** in preference to raw frame-difference (immune to lighting flicker and illuminator cycling).
- Compute the RGB trace on the **track-matched crop only**. Using the full frame lets the other mouse's motion contaminate the correlation and can lock onto a spurious lag.
- Cross-correlate; interpolate the correlation peak parabolically for sub-frame lag resolution (raw resolution is 100 ms at 10 fps).
- Fit lag independently in ≥5 windows across the session; regress lag on time to recover offset and clock drift jointly as an affine time map.
- Verify against 2–3 sharp motion events.
- **Acceptance (proposed, to be tuned):** residual < 1 thermal frame across all windows, and drift fit R² > 0.9 or drift consistent with zero. Failure → flag session, drop to thermal-only.

### Stage 4 — Spatial registration

Per-session homography (RGB → thermal) fitted from gradient-track plate edges and corners, which are visible in both modalities. Fit on the floor plane.

- Initialize from a canonical homography; refine per session. The rig is nudged, so a single global transform is not safe.
- **Parallax:** the mouse sits ~15–20 mm above the floor plane, and obliquity grows toward the ends of the track — exactly where measurements matter most. Quantify empirically (warm object of known height at several positions), and apply a fixed height-offset correction if the residual exceeds ~1 px.
- **Within-session nudge detection:** track plate-edge positions over time; flag discontinuities.
- **Orientation is per-session, not a fixed rig assumption.** Confirmed 2026-08-25 (Test_4, see `thermalFeatures/Track_Alignment_Test4.pptx`): the RGB↔thermal relationship is not guaranteed to be the same "vertical flip only" convention across every session — one camera can be physically repositioned between recording days while the other stays fixed, silently changing a session's true correspondence to a full 180° rotation (vertical **and** horizontal flip). Two independent, careful interactive calibration attempts both reproduced the same wrong mapping, because a human clicking "the same physical corner" under the wrong orientation assumption does so *consistently* wrong — the resulting fit's own reprojection RMSE looks fine either way, since RMSE only measures self-consistency with whatever points it was given, not agreement with physical reality. **Required gate before a session's homography is trusted downstream:** run `scripts/validate_homography_orientation.py` (wraps `src.landmarks.registration.validate_homography_orientation()`) against real, independently-tracked motion (a full-session RGB centroid track compared to the existing thermal tracking, over long/stable bouts) — not against reprojection RMSE or a canonical/trusted-reference comparison, both of which are blind to this failure mode for the reasons above. Re-run this check any time a session's camera setup could plausibly have changed (different recording day, equipment moved/serviced).

*Acceptance:* reprojection RMSE below ~1 px, reported per session in QC. Additionally, orientation validation (above) must not flag the session before its homography is used for Stage 6/7 measurement.

### Stage 5 — RGB landmark extraction

**Primary method — classical CV.** A black mouse on a white plate is near-solved: temporal background model + threshold → binary mask. Fast on CPU, no training, no labels.

From the mask:
- Medial axis / skeleton, ordered nose → tail.
- **Width profile along the medial axis** carries real anatomical structure: neck narrowing, shoulder shelf, rump, and a sharp drop at the tail base. Derive landmarks from profile features rather than fixed fractional positions.
- **Tail base** is the crispest landmark available in either modality: the junction where a thin filament meets the body outline. Define it as the medial-axis point where width drops below a threshold fraction of trunk width.
- **Tail centerline:** trace the filament outward from the tail base as an ordered curve (see §7).

**False-positive filtering is mandatory, not optional.** Fecal boli appear as small dark ellipses in RGB and warm blobs in thermal. Reject by area, aspect ratio, and track continuity.

**Ears** are visible only in near-nadir frames and disappear at oblique viewing angles toward the ends of the track. They may refine interscapular placement where present but **cannot be a required input**.

**Fallback method — supervised pose (SLEAP / DeepLabCut / Lightning Pose).** Invoked only for frames where classical CV fails QC. Single-animal mode (tracks are already cropped). Training and batch inference on ARCC GPU partitions; never a desktop dependency.

**Bake-off protocol.** Evaluate classical vs. supervised on the 75-frame validation set. Metrics:
1. Landmark pixel error in RGB;
2. Thermal-value error vs. human-drawn ROI;
3. **Yield** — fraction of bouts producing a valid measurement, *reported separately for cool / mid / hot thirds of the gradient*.

Yield stratified by position is the metric that decides this. A method with better mean accuracy but position-dependent yield is worse than one with uniform yield, because non-uniform yield reintroduces the exact bias §3 exists to eliminate.

### Stage 6 — Thermal measurement extraction

Landmarks warp from RGB into thermal coordinates via the Stage 4 homography. Then:

**Warm spot.** Maximum (or 95th percentile) within the anterior third of the warped mask. Expect this to be recoverable at the cool end and **unrecoverable at the hot end**, where the animal reads at or below floor. State this limitation in outputs rather than discovering it in QC.

**Tail base.** Do *not* use a point ROI — at the estimated scale the tail base is ~1 px wide and a point sample is mostly floor. Instead:
- warp the traced tail centerline into thermal space;
- sample max-along-curve over the proximal segment (~first 20% of tail length);
- take a high percentile of those samples;
- report as **ΔT above local floor**, measured in an annulus adjacent to the tail, excluding both the animal **and any nestlet pixels** (Stage 2b). If the annulus cannot be populated with enough clean bare-plate pixels, the measurement is invalid — do not fall back to a session-mean floor temperature, which reintroduces exactly the position-dependent bias the local reference exists to remove.

**Gating.** A measurement is valid only if: |ΔT| vs. local floor exceeds threshold; landmark confidence exceeds threshold; postural state is not curled or rearing; sync and homography QC pass.

**Record the sign of ΔT as a variable.** A mouse reading cooler than the floor at the hot end is a real physiological observation, not an error.

**Temporal averaging.** Within a bout, average across frames — ~300 frames in a 30 s bout drives sensor noise well below relevance. Note explicitly that averaging reduces *noise* but not *partial-volume bias*; bias is the limiting error term and must be handled by the curve-sampling approach above, not by more frames.

### Stage 7 — Outputs and QC

**Per-bout table (primary):** session, track, mouseID, bout ID, start/end times in both clocks, gradient zone, mean floor temperature, per-landmark ΔT and absolute value, ROI pixel counts, frames averaged, all quality flags.

**Per-frame table (diagnostic):** landmark coordinates in both spaces, ROI statistics, local background.

**Per-session QC report:** homography RMSE, sync lag and drift with residuals, nudge events, landmark yield by gradient third, fallback-invocation count, rejected-detection count.

QC is what makes this trustworthy across ~100 sessions. Build it in Stage 3, not at the end.

## 7. Key technical rationale (retain for future readers)

The tail-base approach deserves explanation because it looks unusual. At an estimated ~0.32 cm/px, the tail base spans roughly one pixel, so every pixel there is a mixture of tail and floor. Critically, the floor is a *gradient* — so partial-volume bias is a systematic function of position along the track, which is the independent variable. Naïve point sampling would manufacture an apparent position-dependent tail temperature effect out of pure measurement artifact.

Sampling along the traced centerline converts a single ambiguous pixel into a few dozen samples along a structure whose position and orientation are known from RGB. Taking the max along that known curve is substantially less biased than a mean over a blob, and expressing the result as ΔT above local floor removes the position-dependent offset.

## 8. Validation set design

75 frames, all held out, none used for training. Stratify by:
- **Gradient position** — over-sample the hot and intermediate thirds, where both modalities are hardest. Do not sample uniformly.
- **Posture** — include curled, rearing, grooming, and extended.
- **Track** — both B and F.
- **Nestlet** — include both nestlet and no-nestlet sessions, and specifically include frames where the animal is on or adjacent to the nestlet, since that is where the background-annulus logic is most likely to fail.
- **Session** — spread across ~25 sessions (≈3 frames each) rather than concentrating in few.

Label: mask outline, tail-base point, tail centerline, interscapular point, head, and a human-drawn warm-spot ROI.

## 9. Milestones

| # | Milestone | Gate |
|---|---|---|
| M0 | `v6-stable` tagged, branch created, golden regression test passing | — |
| M1 | Radiometric audit complete, raw SEQ reader tested | Blocking |
| M2 | Sync + homography working on the example session with QC output | Residuals within acceptance |
| M3 | Classical landmark extraction on one session, visual overlay QC | Eyeball check |
| M4 | 75-frame validation set labeled; bake-off metrics computed | Yield uniform across gradient thirds |
| M5 | Batch run across full corpus; per-session QC reports | — |
| M6 | Merge behind config flag; regression test re-run | Golden outputs unchanged |

## 10. Development data

- Metadata LUT: `/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/metadata/LUT_CLEAN_July6.csv`
- Starting webcam video: `/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/07-28-25_Test_3/2025-07-28_10-57-21.mp4`

Develop against a desktop subset; deploy to GitHub, then ARCC Alcova and Medicine Bow.

## 11. Open parameters

1. **Thermal pixel scale** *(Lucas — pending)*. Needed: mouse nose-to-tail-base length in px, and tail width near the base, in the *uncropped* thermal frame. Also the physical gradient track length. The §7 estimate assumes ~150 cm over 464 px → ~0.32 cm/px → ~28 px body, ~1 px tail base. **If the true scale is meaningfully finer, the tail-centerline machinery in Stage 6 may be unnecessary and a simple ROI would do.** Confirm before building it.
2. **Webcam frame rate** — read from file header at Stage 2; low risk.
3. **Acquisition emissivity setting** — determines whether a retroactive correction is needed (Stage 0).
4. **Gradient plate material** — if aluminum, check empirically for specular reflection of the animal contaminating adjacent pixels.
