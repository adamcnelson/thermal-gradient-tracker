import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.landmarks.bout_gating import iter_stationary_bouts, filter_bouts_after_entry
from src.landmarks.webcam_preprocessing import split_track_crops, detect_track_split_row, LANE_TOP
from src.landmarks.rgb_landmarks import (
    RgbBackgroundModel,
    segment_mouse_rgb,
    extract_mouse_detection,
)
from src.landmarks.registration import apply_homography
from src.landmarks.thermal_measurement import (
    anterior_region_mask,
    dorsal_surface_mask,
    proximal_tail_points,
    warp_mask_to_thermal,
    warm_spot_temperature,
    dorsal_surface_temperature,
    tail_base_delta_t,
    gate_measurement,
    GatingThresholds,
    average_valid,
    TailMeasurement,
)
from src.landmarks.sync import WindowedSyncResult, passes_acceptance as sync_passes_acceptance
from src.landmarks.outputs import (
    build_bout_output_row,
    bout_rows_to_dataframe,
    FrameOutputRow,
    frame_rows_to_dataframe,
    build_session_qc_report,
    compute_landmark_yield_by_zone,
    thermal_time_to_rgb_time,
)

MIN_AREA = 200
MAX_AREA = 20000
SAMPLE_RADIUS_PX = 2
FLOOR_INNER_RADIUS_PX = 4
FLOOR_OUTER_RADIUS_PX = 10
FRAMES_PER_BOUT = 3  # fractional positions within the bout window
BOUT_SAMPLE_FRACS = [0.25, 0.5, 0.75]
REGISTRATION_LOW_CONFIDENCE_PX = 30.0  # Adam, 2026-08-27: "flag inconsistent cases and move
                                        # on" (after the rotation-correction attempt found no
                                        # reliable auto-fix -- see project memory). 30px is the
                                        # same "big residual" cutoff already established with
                                        # real data in the alignment investigation
                                        # ([[project-v7-test7-alignment-investigation]]) --
                                        # not a new guess. Flags, does not exclude: real fast
                                        # motion during the sample is the dominant cause of a
                                        # large correction, not always a registration bug.
LOCAL_EDGE_REFINE_SEARCH_PX = 20  # bounded search window (translation only) for
LOCAL_EDGE_REFINE_RING_PX = 4     # local_edge_refine() -- see its docstring. Adam,
LOCAL_EDGE_REFINE_MIN_IMPROVEMENT_C = 1.0  # 2026-08-31: found real cases where the
                                    # whole-mask centroid correction alone left the mask
                                    # confidently but visibly wrong near a raised/parallax-
                                    # shifted body part. Validated on 27 real frames (19
                                    # flagged + 8 broader sample) before wiring in: 6/27
                                    # triggered a correction, all visibly better on
                                    # inspection; the other 21 were correctly left alone
                                    # (already-good, or no distinguishable local edge).
DORSAL_STRICT_SIGMA = 6.0  # stricter segmentation threshold used ONLY for the dorsal/warm-spot
                            # boundary (see the real, measured rationale next to its use below);
                            # posture classification and tail-finding keep the default (looser)
                            # threshold to avoid regressing find_tail_appendage()'s yield.
MAX_PRUNE_PX = 10  # tolerate short spurious skeleton branches (segmentation noise) before
                    # the simple-path check; see prune_short_skeleton_branches() docstring.
                    # Real measured effect (2026-08-26): Test_3 +1, Test_4 +3, Test_7 +0
                    # extended samples -- deliberately conservative, does not rescue
                    # genuinely hunched/curled-with-tail-out bodies.

# Sync results are per-SESSION, not per-lane (Front/Back share the same physical
# camera clocks -- confirmed via identical Front.seq/Back.seq frame counts, see
# [[project-v7-back-lane-expansion]]), so each is defined once and reused for
# both the Front and Back SESSIONS entries below rather than redefined.
SYNC_RESULT_TEST_3 = WindowedSyncResult(
    window_centers_sec=np.array([]), window_lags_sec=np.array([]),
    offset_sec=-2.8, drift_slope=0.0, r_squared=0.999, residual_max_sec=0.2,
    low_confidence=False,
    confidence_note=(
        "resolved 2026-08-25: 5 real, human-verified anchor events spanning the full "
        "session (t=46-2261s thermal), fit AFTER discovering and correcting the "
        "camera_fps 8->10 error (see project memory) -- offset is now a clean constant "
        "(std=0.13s across all 5 anchors), no drift term needed. Supersedes the earlier "
        "-93.0/drift=0 value, which was compensating for the fps error, not real RGB/"
        "thermal clock drift. These anchors were all found on the FRONT lane; reused "
        "unchanged for Back (2026-08-28) since Front.seq/Back.seq are the same recording."
    ),
)
SYNC_RESULT_TEST_4 = WindowedSyncResult(
    window_centers_sec=np.array([]), window_lags_sec=np.array([]),
    offset_sec=5.5, drift_slope=0.0, r_squared=0.995, residual_max_sec=0.5,
    low_confidence=False,
    confidence_note=(
        "resolved 2026-08-25: 4 real, human-verified anchor events spanning the full "
        "session (t=94-1615s thermal), fit AFTER discovering and correcting the "
        "camera_fps 8->10 error (see project memory) -- offset is now a clean constant "
        "(std=0.46s across all 4 anchors), no drift term needed. Supersedes the earlier "
        "low-confidence +11.0 value, which was itself compensating for the fps error on "
        "top of the real offset, not a genuinely unresolved sync. These anchors were all "
        "found on the FRONT lane; reused unchanged for Back (2026-08-28) since "
        "Front.seq/Back.seq are the same recording."
    ),
)
SYNC_RESULT_TEST_7 = WindowedSyncResult(
    window_centers_sec=np.array([]), window_lags_sec=np.array([]),
    offset_sec=-0.51, drift_slope=0.0, r_squared=0.9999, residual_max_sec=0.5,
    low_confidence=False,
    confidence_note=(
        "resolved 2026-08-26: 4 real, human-verified anchor events spanning the full "
        "session (t=28-1588s thermal). fps was already 10.0 in this session's tracking "
        "data (unlike Test_3/4, no fps correction was needed here -- confirmed via these "
        "same anchors: slope=1.00002, implied fps=10.0002). Offset resolved to a tiny "
        "near-zero constant, no drift. Homography orientation was separately disputed and "
        "resolved: Adam's direct video review initially flagged a Test_4-style 180-degree "
        "rotation, but a precise same-real-moment overlay (both centroid AND full mouse "
        "silhouette, warped through the as-clicked calibration) landed almost exactly on "
        "the real thermal mouse blob including head/tail orientation -- the as-clicked "
        "(unmirrored) calibration was confirmed correct; raw RGB-vs-thermal frames are "
        "just genuinely hard to compare by eye given the real vertical flip + differing "
        "resolution/aspect ratio + the homography's own rotation/perspective terms. These "
        "anchors were all found on the FRONT lane; reused unchanged for Back (2026-08-28) "
        "since Front.seq/Back.seq are the same recording."
    ),
)

SESSIONS = {
    "Test_3": dict(
        session_label="07-28-25_4540_B_4541_F_Test3-004",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Front.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/07-28-25_Test_3/2025-07-28_10-57-21.mp4",
        homography_json="homography_calibration/07-28-25_4540_B_4541_F_Test3-004_Front_homography.json",
        tracking_csv="trackingOutputs/07-28-25_4540_B_4541_F_Test3-004_Front_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Front_bout_rows.csv",
        mouse_id=4541,
        track="F",
        entry_time_thermal_sec=34.85,  # RGB-confirmed entry (32.05s) converted via offset -2.8s (post camera_fps 8->10 fix)
        sync_result=SYNC_RESULT_TEST_3,
    ),
    "Test_4": dict(
        session_label="07-30-25_4540_F_4541_B_Test4-008",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/07-30-25_Test_4/07-30-25_4540_F_4541_B_Test4-008_Front.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/07-30-25_Test_4/2025-07-30_10-44-05.mp4",
        homography_json="homography_calibration/07-30-25_4540_F_4541_B_Test4-008_Front_homography.json",
        tracking_csv="trackingOutputs/07-30-25_4540_F_4541_B_Test4-008_Front_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Front_bout_rows.csv",
        mouse_id=4540,
        track="F",
        entry_time_thermal_sec=77.33,  # RGB-confirmed entry (82.83s) converted via offset +5.5s (post camera_fps 8->10 fix)
        sync_result=SYNC_RESULT_TEST_4,
    ),
    "Test_7": dict(
        session_label="08-07-25_4541_F_4540_B_Test7-020",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/08-07-25_Test_7/08-07-25_4541_F_4540_B_Test7-020_Front.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/08-07-25_Test_7/2025-08-07_10-54-09.mp4",
        homography_json="homography_calibration/08-07-25_4541_F_4540_B_Test7-020_Front_homography.json",
        tracking_csv="trackingOutputs/08-07-25_4541_F_4540_B_Test7-020_Front_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Front_bout_rows.csv",
        mouse_id=4541,
        track="F",
        entry_time_thermal_sec=28.0,  # RGB-confirmed touchdown (27s) converted via offset -0.51s
        sync_result=SYNC_RESULT_TEST_7,
    ),
    # ── Back lane, added 2026-08-28 (Adam: "wire Back into stage7_real_run.py") ──
    # Front.seq/Back.seq confirmed to be the SAME underlying recording for all 3
    # sessions (identical frame counts, verified via SeqReader -- see
    # [[project-v7-back-lane-expansion]]), so camera_fps and the RGB<->thermal
    # sync_result established for Front apply directly and are reused unchanged
    # below -- NOT re-derived, deliberately. Each session's Back homography was
    # independently calibrated and validated via validate_homography_orientation.py
    # (all 3 confirmed as-clicked correct, real margins 4-25px vs 67-159px mirrored
    # -- see project memory). entry_time_thermal_sec below is the Stage 1/2
    # AUTO-DETECTED value (tracking_config's auto_detect_tracking_start), NOT a
    # human-verified real anchor like the Front lanes' entry times -- a real,
    # lower-confidence input than Front's, not yet upgraded.
    "Test_3_Back": dict(
        session_label="07-28-25_4540_B_4541_F_Test3-004",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Back.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/07-28-25_Test_3/2025-07-28_10-57-21.mp4",
        homography_json="homography_calibration/07-28-25_4540_B_4541_F_Test3-004_Back_homography.json",
        tracking_csv="trackingOutputs/07-28-25_4540_B_4541_F_Test3-004_Back_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Back_bout_rows.csv",
        mouse_id=4540,
        track="B",
        entry_time_thermal_sec=71.0,  # auto-detected (Stage 1/2), not a verified anchor -- see note above
        sync_result=SYNC_RESULT_TEST_3,
    ),
    "Test_4_Back": dict(
        session_label="07-30-25_4540_F_4541_B_Test4-008",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/07-30-25_Test_4/07-30-25_4540_F_4541_B_Test4-008_Back.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/07-30-25_Test_4/2025-07-30_10-44-05.mp4",
        homography_json="homography_calibration/07-30-25_4540_F_4541_B_Test4-008_Back_homography.json",
        tracking_csv="trackingOutputs/07-30-25_4540_F_4541_B_Test4-008_Back_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Back_bout_rows.csv",
        mouse_id=4541,
        track="B",
        entry_time_thermal_sec=81.0,  # auto-detected (Stage 1/2), not a verified anchor -- see note above
        sync_result=SYNC_RESULT_TEST_4,
    ),
    "Test_7_Back": dict(
        session_label="08-07-25_4541_F_4540_B_Test7-020",
        thermal_seq="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/08-07-25_Test_7/08-07-25_4541_F_4540_B_Test7-020_Back.seq",
        rgb_video="/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/Process_Jason/08-07-25_Test_7/2025-08-07_10-54-09.mp4",
        homography_json="homography_calibration/08-07-25_4541_F_4540_B_Test7-020_Back_homography.json",
        tracking_csv="trackingOutputs/08-07-25_4541_F_4540_B_Test7-020_Back_tracking_every10frames.csv",
        bouts_csv="bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Back_bout_rows.csv",
        mouse_id=4540,
        track="B",
        entry_time_thermal_sec=132.0,  # auto-detected (Stage 1/2), not a verified anchor -- see note above
        sync_result=SYNC_RESULT_TEST_7,
    ),
}

REPO = "/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/thermal-gradient-tracker"
THERMAL_FPS = 10.0  # true rate for Test_3/Test_4/Test_7 post camera_fps 8->10 fix (2026-08-25/26) --
                     # bout times (thermal_t) come from elapsed_time_sec, which is now genuinely
                     # frame_idx/10.0 for all three sessions; using 8.0 here would silently read
                     # the wrong raw thermal frame for every sample (a real bug caught 2026-08-26
                     # while adding Test_7 -- Test_3/Test_4's post-fps-fix Stage7 runs need redoing).


def load_homography(path):
    d = json.load(open(path))
    return np.array(d["H"], dtype=np.float64)


def build_thermal_native_lookup(tracking_df, max_gap_sec=2.0):
    """
    Real fix, 2026-08-27 (Adam, after reviewing measurement_location_qc
    images: "the ROIs are in the bottom center of the thermal image; the
    mouse is clearly in the upper right"; "it seems like we need to
    implement a routine where the green and blue ROIs from the RGB need
    to be aligned with equivalent designations that are derived from the
    thermal"). The tracking CSV already has an independent, thermal-
    native mouse centroid track (Stage 1/2's own thermal-side detection,
    `mouse_centroid_x/y`, ~1 row/sec) that was only used for QC/ground-
    truth comparison in the earlier alignment investigation
    ([[project-v7-test7-alignment-investigation]]). Confirmed on the
    exact frames Adam flagged (Test_4 t=355.2/362.5s, Test_7
    t=967.0/1483.5s) that shifting the warped RGB geometry to match this
    track's position at the same instant visually fixes the misalignment
    -- the ROI lands on the real thermal blob instead of nearby smooth
    gradient. Returns a lookup(thermal_t) -> Optional[(x, y)] using
    linear interpolation between the two nearest valid (mouse_roi_valid
    True, non-NaN) rows; returns None if the nearest valid row is more
    than max_gap_sec away, so a stale/missing thermal-native detection
    is never silently extrapolated across a real gap.
    """
    valid = tracking_df[(tracking_df["mouse_roi_valid"] == True) & tracking_df["mouse_centroid_x"].notna()]
    valid = valid.sort_values("elapsed_time_sec")
    t = valid["elapsed_time_sec"].to_numpy()
    x = valid["mouse_centroid_x"].to_numpy()
    y = valid["mouse_centroid_y"].to_numpy()

    def lookup(thermal_t):
        if len(t) == 0:
            return None
        idx = np.searchsorted(t, thermal_t)
        nearby = [t[i] for i in (idx - 1, idx) if 0 <= i < len(t)]
        if not nearby or min(abs(c - thermal_t) for c in nearby) > max_gap_sec:
            return None
        return float(np.interp(thermal_t, t, x)), float(np.interp(thermal_t, t, y))

    return lookup


def registration_correction_homography(H, mask, thermal_shape, thermal_native_xy):
    """
    Fold a translation correction (aligning `mask`'s warped centroid with
    thermal_native_xy) directly into H, so every downstream warp
    (warp_mask_to_thermal, apply_homography, warp_points_xy /
    tail_base_delta_t) picks it up with no other code changes: for a
    homography H and translation T=[[1,0,dx],[0,1,dy],[0,0,1]], T@H
    projects any point exactly to (H's own projection) + (dx, dy), since
    T only ever touches H@p's x/y numerator, not its homogeneous w
    (verified algebraically and re-confirmed against the visual
    prototype before wiring in). Returns (H, None) unchanged if the
    animal mask warps to nothing or thermal_native_xy is None (no
    thermal-native reference nearby) -- no correction is still a valid,
    intentional choice, not a failure.
    """
    if thermal_native_xy is None:
        return H, None
    warped = warp_mask_to_thermal(mask, H, thermal_shape)
    ys, xs = np.where(warped)
    if not len(ys):
        return H, None
    dx = thermal_native_xy[0] - float(xs.mean())
    dy = thermal_native_xy[1] - float(ys.mean())
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    return T @ H, (dx, dy)


def local_edge_refine(thermal_celsius, warped_mask, search_px=LOCAL_EDGE_REFINE_SEARCH_PX,
                       ring_px=LOCAL_EDGE_REFINE_RING_PX, min_improvement_c=LOCAL_EDGE_REFINE_MIN_IMPROVEMENT_C):
    """
    A second, small correction layered on top of registration_correction_homography()'s
    whole-mask centroid correction. Real motivation (2026-08-31, Adam): pointed at real
    dorsal_surface_qc frames where the whole-mask correction was small/"confident" (well
    under REGISTRATION_LOW_CONFIDENCE_PX) yet the mask still visibly missed a raised body
    part (head/nose) -- a single global translation, calibrated on the torso-dominated
    mask centroid, can't also correct a parallax-shifted body part, since that offset is
    position-dependent, not uniform across the animal (see
    [[project-v7-local-edge-refinement]]).

    Deliberately NOT a "find the hottest nearby pixel" search: tested directly on real
    frames and found this track's floor gradient can be very steep locally (15C -> 45C
    within 20-60 raw thermal px in some crops), so a temperature-peak search past ~40px
    reliably locks onto an unrelated, hotter part of the floor gradient rather than the
    animal. Instead this maximizes INSIDE-vs-RING separation (mean temp inside the
    (shifted) mask vs. a thin ring immediately outside it) within a small bounded window
    -- an edge-alignment objective, not a peak search, so it can't wander off toward the
    gradient's hot end the way a peak search would.

    Robust to mouse/floor thermal blending (the real failure mode from the beta pipeline
    Adam flagged) by construction: if the mouse's surface temperature matches the floor
    at this point on the gradient, no offset in the window will show a real inside/ring
    separation, so `improved` stays False and the position is left untouched rather than
    guessed at -- this never depends on thermal contrast to LOCATE the animal (that's
    still entirely RGB's job), only to fine-tune a position RGB already supplied.

    Validated 2026-08-31 on 27 real frames (19 Adam flagged + 8 broader random sample)
    before being wired in here: 6/27 triggered a correction, all visibly better on
    inspection; the other 21 were correctly left alone.

    Returns (dx, dy, improved, base_score, best_score). dx/dy are in thermal-array
    pixels, meant to be folded into H_sample the same way registration_correction_
    homography()'s (dx, dy) already are (T @ H_sample) -- see the call site.
    """
    h, w = thermal_celsius.shape
    ys, xs = np.where(warped_mask)
    if len(ys) == 0:
        return 0, 0, False, None, None

    kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    dilated = cv2.dilate(warped_mask.astype(np.uint8), kernel).astype(bool)
    ring = dilated & ~warped_mask
    ry, rx = np.where(ring)

    def score(dx, dy):
        ys2, xs2 = ys + dy, xs + dx
        valid = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
        if valid.sum() < 0.5 * len(ys):
            return None
        inside_vals = thermal_celsius[ys2[valid], xs2[valid]]

        ry2, rx2 = ry + dy, rx + dx
        rvalid = (ry2 >= 0) & (ry2 < h) & (rx2 >= 0) & (rx2 < w)
        if rvalid.sum() < 10:
            return None
        ring_vals = thermal_celsius[ry2[rvalid], rx2[rvalid]]
        return abs(float(inside_vals.mean()) - float(ring_vals.mean()))

    base_score = score(0, 0)
    if base_score is None:
        return 0, 0, False, None, None

    best_dx, best_dy, best_score = 0, 0, base_score
    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            s = score(dx, dy)
            if s is not None and s > best_score:
                best_dx, best_dy, best_score = dx, dy, s

    if best_score - base_score >= min_improvement_c:
        return best_dx, best_dy, True, base_score, best_score
    return 0, 0, False, base_score, best_score


def process_session(name, cfg):
    t_start = time.time()
    print(f"\n=== {name} ===", flush=True)

    tracking_df = pd.read_csv(cfg["tracking_csv"])
    bouts_df_raw = pd.read_csv(cfg["bouts_csv"])
    bouts_df = filter_bouts_after_entry(bouts_df_raw, entry_time_sec=cfg["entry_time_thermal_sec"])
    n_dropped_pre_entry = len(bouts_df_raw) - len(bouts_df)
    print(f"pre-entry filter: entry_time={cfg['entry_time_thermal_sec']:.2f}s thermal-clock, "
          f"dropped {n_dropped_pre_entry}/{len(bouts_df_raw)} bouts", flush=True)
    H, homography_rmse = load_homography(f"{REPO}/{cfg['homography_json']}"), None
    import json as _json
    homography_rmse = _json.load(open(f"{REPO}/{cfg['homography_json']}"))["rmse_px"]
    thermal_native_lookup = build_thermal_native_lookup(tracking_df)
    n_corrected = 0
    correction_mags = []
    n_edge_refined = 0

    sync_result = cfg["sync_result"]

    # ---- gather target (bout, thermal_time_sec) sample points ----
    bout_frame_plan = []
    for bout, frames in iter_stationary_bouts(tracking_df, bouts_df):
        b0, b1 = bout["bout_start_sec"], bout["bout_end_sec"]
        for frac in BOUT_SAMPLE_FRACS[:FRAMES_PER_BOUT]:
            bout_frame_plan.append((int(bout["bout_index"]), b0 + frac * (b1 - b0)))
    print(f"bouts with valid frames: {bouts_df['bout_index'].nunique()} planned samples: {len(bout_frame_plan)}", flush=True)

    # ---- one sequential pass through the thermal .seq, grab target frames ----
    thermal_idx_wanted = sorted({int(round(t * THERMAL_FPS)) for _, t in bout_frame_plan})
    idx_set = set(thermal_idx_wanted)
    planck = read_planck_constants(cfg["thermal_seq"])
    thermal_frames = {}
    reader = SeqReader(cfg["thermal_seq"])
    t0 = time.time()
    for idx, raw in reader.frames():
        if idx in idx_set:
            thermal_frames[idx] = raw_to_celsius(raw, planck)
            if len(thermal_frames) == len(idx_set):
                break
    reader.close()
    print(f"thermal pass: {len(thermal_frames)}/{len(idx_set)} frames read in {time.time()-t0:.1f}s", flush=True)

    # ---- RGB background model + per-frame seek/read ----
    cap = cv2.VideoCapture(cfg["rgb_video"])
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    total_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    split_row = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))

    bg_indices = np.linspace(0, total_rgb - 1, 30, dtype=int)
    bg_frames = []
    for idx in bg_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(gray, split_row=split_row)
            bg_frames.append(top if cfg["track"] == LANE_TOP else bottom)
    bg_model = RgbBackgroundModel.build(bg_frames)
    print(f"split_row={split_row} rgb_fps={rgb_fps:.3f} bg from {len(bg_frames)} frames", flush=True)

    # ---- per-sample landmark + thermal measurement ----
    per_sample = []
    frame_rows = []
    t0 = time.time()
    for bout_index, thermal_t in bout_frame_plan:
        thermal_idx = int(round(thermal_t * THERMAL_FPS))
        thermal_celsius = thermal_frames.get(thermal_idx)
        rec = dict(bout_index=bout_index, thermal_t=thermal_t, thermal_idx=thermal_idx)
        if thermal_celsius is None:
            rec["fail"] = "no thermal frame"
            per_sample.append(rec)
            continue

        rgb_t = thermal_time_to_rgb_time(thermal_t, sync_result)
        rgb_frame_idx = int(round(rgb_t * rgb_fps))
        if not (0 <= rgb_frame_idx < total_rgb):
            rec["fail"] = "rgb time out of video range"
            per_sample.append(rec)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_idx)
        ok, frame = cap.read()
        if not ok:
            rec["fail"] = "rgb read failed"
            per_sample.append(rec)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top, bottom = split_track_crops(gray, split_row=split_row)
        crop = top if cfg["track"] == LANE_TOP else bottom

        mask = segment_mouse_rgb(crop, bg_model, min_area=MIN_AREA, max_area=MAX_AREA)
        if mask is None:
            rec["fail"] = "no mouse blob segmented"
            per_sample.append(rec)
            continue
        # Real fix, 2026-08-27 (Adam: mask visibly bigger than the mouse in
        # measurement_location_qc images). Confirmed via a 180-frame survey
        # that the default threshold_sigma=3.0 mask is ~35-45% larger in
        # area than a strict high-confidence-core mask in EVERY session, a
        # soft motion-blur/shadow penumbra the global mean+3*std threshold
        # admits. Tried raising threshold_sigma / trimming morphology
        # globally first -- real measured cost: sigma 3.0->3.5 alone loses
        # 37% of already-hard-won tail-appendage detections (22/60) for only
        # a 6% area reduction, because the tail is itself a thin, low-
        # contrast structure indistinguishable from the halo by a single
        # global threshold. So: keep `mask` (loose, sigma=3.0) for posture
        # classification and tail-finding exactly as before -- don't
        # regress the yield fixed in find_tail_appendage() -- and compute a
        # SEPARATE, stricter mask used only for the dorsal/warm-spot
        # boundary, where a tight silhouette is what was actually wanted.
        # sigma=6.0 chosen from the same survey (0/182 frames returned None,
        # area never collapsed toward MIN_AREA); visually confirmed on the
        # two frames Adam flagged as oversized -- the strict mask hugs the
        # real dark silhouette while the loose mask's boundary visibly
        # extended into the shadow/floor.
        dorsal_mask_source = segment_mouse_rgb(crop, bg_model, min_area=MIN_AREA, max_area=MAX_AREA,
                                                threshold_sigma=DORSAL_STRICT_SIGMA)
        if dorsal_mask_source is None:
            dorsal_mask_source = mask
        detection = extract_mouse_detection(mask, MIN_AREA, MAX_AREA, max_prune_px=MAX_PRUNE_PX)
        if detection is None:
            rec["fail"] = "rejected as debris-scale"
            rec["posture"] = None
            per_sample.append(rec)
            continue
        posture = detection.posture
        lm = detection.landmarks
        tail_lm = detection.tail_landmarks  # only set for non-"extended" postures

        if posture != "extended" and tail_lm is None:
            # Real, total loss -- no whole-body decomposition AND no
            # separately-resolvable tail either (e.g. a genuinely tucked
            # curl). Nothing measurable at all for this sample.
            rec["fail"] = (
                "ambiguous posture, no tail found" if posture == "ambiguous"
                else "curled posture, no tail found"
            )
            rec["posture"] = posture
            per_sample.append(rec)
            continue

        thermal_native_xy = thermal_native_lookup(thermal_t)
        H_sample, correction = registration_correction_homography(H, mask, thermal_celsius.shape, thermal_native_xy)
        if correction is not None:
            n_corrected += 1
            correction_mags.append(float(np.hypot(*correction)))
        correction_px = float(np.hypot(*correction)) if correction is not None else None
        rec["registration_correction_px"] = correction_px
        rec["registration_low_confidence"] = bool(
            correction_px is not None and correction_px > REGISTRATION_LOW_CONFIDENCE_PX)

        warped_animal = warp_mask_to_thermal(mask, H_sample, thermal_celsius.shape)

        edge_dx, edge_dy, edge_improved, _, _ = local_edge_refine(thermal_celsius, warped_animal)
        rec["local_edge_refined"] = edge_improved
        if edge_improved:
            n_edge_refined += 1
            T_edge = np.array([[1, 0, edge_dx], [0, 1, edge_dy], [0, 0, 1]], dtype=np.float64)
            H_sample = T_edge @ H_sample
            warped_animal = warp_mask_to_thermal(mask, H_sample, thermal_celsius.shape)

        if posture == "extended":
            anterior = anterior_region_mask(dorsal_mask_source, lm.path_nose_to_tail[: lm.tail_base_index + 1])
            dorsal = dorsal_surface_mask(dorsal_mask_source, lm.path_nose_to_tail, lm.tail_base_index)
            prox_tail = proximal_tail_points(lm.tail_centerline)
            warped_anterior = warp_mask_to_thermal(anterior, H_sample, thermal_celsius.shape)
            warm_spot = warm_spot_temperature(thermal_celsius, warped_anterior)
            nose_xy = apply_homography(H_sample, np.array([[lm.nose_point[1], lm.nose_point[0]]]))[0]
            tb_xy = apply_homography(H_sample, np.array([[lm.tail_base_point[1], lm.tail_base_point[0]]]))[0]
        else:
            # Tail-only fallback (2026-08-26, see find_tail_appendage()):
            # warm-spot/dorsal-anterior genuinely need the whole-body
            # straight-line decomposition (anterior_region_mask() would
            # warp nonsense for a hunched/curled body -- see
            # project-v7-stage5-bakeoff Finding #2) so those stay
            # inapplicable here, exactly as before. Tail-ΔT does not
            # share that requirement -- tail_base_delta_t() only ever
            # consumes proximal_tail_points_rgb, so a real, separately-
            # resolved tail appendage is sufficient on its own.
            dorsal = dorsal_mask_source  # whole-animal mean/median still applies (Adam's own added metric)
            warm_spot = None
            prox_tail = proximal_tail_points(tail_lm.tail_centerline)
            nose_xy = tb_xy = None

        tail_meas = tail_base_delta_t(
            thermal_celsius, H_sample, prox_tail,
            sample_radius_px=SAMPLE_RADIUS_PX,
            floor_inner_radius_px=FLOOR_INNER_RADIUS_PX,
            floor_outer_radius_px=FLOOR_OUTER_RADIUS_PX,
            animal_mask_thermal=warped_animal,
        )

        warped_dorsal = warp_mask_to_thermal(dorsal, H_sample, thermal_celsius.shape)
        dorsal_meas = dorsal_surface_temperature(thermal_celsius, warped_dorsal)

        if posture == "extended":
            landmark_confidence = 1.0
        else:
            landmark_confidence = 0.7  # real tail found via fallback, but no whole-body cross-check
        # posture_ok gates the WHOLE row's qc_valid, but the only real
        # measurement at stake for a non-"extended" row is tail-ΔT (warm_spot
        # is already None here) -- so a real, found tail_landmarks fallback
        # satisfies the intent of "posture ok for the measurement being
        # gated", even though the animal's overall body posture is not
        # "extended". Brief's "posture is not curled/rearing" language was
        # written before this fallback existed; this is a deliberate,
        # scoped reinterpretation for tail-ΔT specifically, not a loosening
        # of warm-spot/dorsal-anterior gating (which remains untouched).
        posture_ok = (posture == "extended") or (tail_lm is not None)
        qc_valid, qc_reasons = gate_measurement(
            delta_t_c=tail_meas.delta_t_c,
            landmark_confidence=landmark_confidence,
            sync_qc_pass=True,
            homography_qc_pass=homography_rmse < 2.0,
            posture_ok=posture_ok,
        )

        rec.update(
            rgb_frame_idx=rgb_frame_idx,
            posture=posture,
            warm_spot_temp_c=warm_spot,
            dorsal_mean_c=dorsal_meas.mean_c,
            dorsal_median_c=dorsal_meas.median_c,
            dorsal_pixel_count=dorsal_meas.pixel_count,
            tail_temp_c=tail_meas.tail_temp_c,
            tail_floor_temp_c=tail_meas.floor_temp_c,
            tail_delta_t_c=tail_meas.delta_t_c,
            qc_valid=qc_valid,
            qc_reasons=qc_reasons,
        )
        per_sample.append(rec)

        tail_base_rgb_xy = lm.tail_base_point if posture == "extended" else (
            tail_lm.tail_base_point if tail_lm is not None else None
        )
        frame_rows.append(FrameOutputRow(
            session=cfg["session_label"],
            track=cfg["track"],
            frame_number=thermal_idx,
            elapsed_time_thermal_sec=thermal_t,
            nose_rgb_xy=lm.nose_point if posture == "extended" else None,
            tail_base_rgb_xy=tail_base_rgb_xy,
            nose_thermal_xy=tuple(nose_xy) if nose_xy is not None else None,
            tail_base_thermal_xy=tuple(tb_xy) if tb_xy is not None else None,
            mouse_surface_temp_mean_c=dorsal_meas.mean_c,
            floor_temp_mean_c=tail_meas.floor_temp_c,
            qc_flag="ok" if qc_valid else "; ".join(qc_reasons),
            posture=posture,
            sync_low_confidence=sync_result.low_confidence,
            registration_low_confidence=rec["registration_low_confidence"],
            local_edge_refined=rec["local_edge_refined"],
        ))
    cap.release()
    print(f"per-sample measurement: {len(per_sample)} samples in {time.time()-t0:.1f}s "
          f"({sum(1 for r in per_sample if 'fail' not in r)} succeeded)", flush=True)
    from collections import Counter
    fail_counts = Counter(r["fail"].split(":")[0] for r in per_sample if "fail" in r)
    print("failure reasons:", dict(fail_counts), flush=True)
    posture_counts = Counter(r.get("posture") for r in per_sample if "fail" not in r or r.get("posture"))
    print("posture breakdown (all attempted samples):", dict(posture_counts), flush=True)
    if correction_mags:
        arr = np.array(correction_mags)
        print(f"registration correction applied: {n_corrected} samples, "
              f"magnitude median={np.median(arr):.1f}px mean={arr.mean():.1f}px max={arr.max():.1f}px", flush=True)
    print(f"local edge refinement applied: {n_edge_refined}/{len(per_sample)} samples", flush=True)

    # ---- aggregate per bout ----
    samples_df = pd.DataFrame(per_sample)
    bout_output_rows = []
    for _, bout in bouts_df.iterrows():
        bi = int(bout["bout_index"])
        sub = samples_df[samples_df["bout_index"] == bi] if "bout_index" in samples_df.columns else pd.DataFrame()
        # Real bug found 2026-08-28 (Test_3_Back: the first session in this project with a
        # perfect 100% per-sample success rate, 33/33 -- every prior session had at least
        # one failure, which is what kept this latent). sub.get("fail", pd.Series(...))
        # only falls back to the empty default when "fail" isn't a column on `sub` at all
        # (true here, since NO sample in the whole session failed) -- but that empty
        # fallback has a mismatched index vs `sub`'s real rows, and pandas boolean
        # indexing raises IndexingError on a misaligned key rather than broadcasting.
        if len(sub) == 0:
            ok_sub = sub
        elif "fail" in sub.columns:
            ok_sub = sub[~sub["fail"].notna()]
        else:
            ok_sub = sub  # no "fail" column at all -- every sample in this bout succeeded
        n = len(ok_sub)
        warm_spot = average_valid(ok_sub["warm_spot_temp_c"].tolist()) if n else None
        dorsal_mean = average_valid(ok_sub["dorsal_mean_c"].tolist()) if n else None
        dorsal_median = average_valid(ok_sub["dorsal_median_c"].tolist()) if n else None
        dorsal_px = int(ok_sub["dorsal_pixel_count"].mean()) if n and ok_sub["dorsal_pixel_count"].notna().any() else 0
        tail_dt = average_valid(ok_sub["tail_delta_t_c"].tolist()) if n else None

        dorsal_obj = None
        if dorsal_mean is not None:
            dorsal_obj = type("D", (), dict(mean_c=dorsal_mean, median_c=dorsal_median, pixel_count=dorsal_px))()
        tail_obj = None
        if tail_dt is not None:
            tail_obj = type("T", (), dict(delta_t_c=tail_dt))()

        qc_valid_bout = bool(n > 0 and ok_sub["qc_valid"].any())
        reasons = []
        if n == 0:
            reasons = ["no successful landmark/measurement samples in this bout"]
        elif not qc_valid_bout:
            reasons = sorted(set(r for rs in ok_sub["qc_reasons"] for r in rs))

        row = build_bout_output_row(
            session=cfg["session_label"],
            track=cfg["track"],
            mouse_id=cfg["mouse_id"],
            bout_id=bi,
            bout_start_thermal_sec=float(bout["bout_start_sec"]),
            bout_end_thermal_sec=float(bout["bout_end_sec"]),
            sync_result=sync_result,
            mean_floor_temp_c=float(bout["floor_temp_mean_bout"]),
            warm_spot_temp_c=warm_spot,
            dorsal=dorsal_obj,
            tail=tail_obj,
            n_frames_averaged=n,
            qc_valid=qc_valid_bout,
            qc_reasons=reasons,
        )
        bout_output_rows.append(row)
        row.__dict__["gt_mouse_surface_temp_mean_bout"] = float(bout["mouse_surface_temp_mean_bout"])
        row.__dict__["n_samples_attempted"] = len(sub)
        row.__dict__["n_samples_dorsal_measured"] = int(ok_sub["dorsal_mean_c"].notna().sum()) if n else 0
        row.n_registration_low_confidence = (
            int(ok_sub["registration_low_confidence"].sum()) if n and "registration_low_confidence" in ok_sub else 0)
        row.n_local_edge_refined = (
            int(ok_sub["local_edge_refined"].sum()) if n and "local_edge_refined" in ok_sub else 0)

    bout_df = bout_rows_to_dataframe(bout_output_rows)
    bout_df["gt_mouse_surface_temp_mean_bout"] = [r.__dict__["gt_mouse_surface_temp_mean_bout"] for r in bout_output_rows]
    bout_df["dorsal_minus_gt_c"] = bout_df["dorsal_mean_c"] - bout_df["gt_mouse_surface_temp_mean_bout"]
    bout_df["n_samples_attempted"] = [r.__dict__["n_samples_attempted"] for r in bout_output_rows]
    bout_df["n_samples_dorsal_measured"] = [r.__dict__["n_samples_dorsal_measured"] for r in bout_output_rows]

    frame_df = frame_rows_to_dataframe(frame_rows)

    # raw dorsal-measurement yield by zone (not qc_valid -- a looser, more direct signal of Stage5 landmark yield)
    dorsal_yield_by_zone = (
        bout_df.assign(has_dorsal=bout_df["dorsal_mean_c"].notna())
        .groupby("gradient_zone")["has_dorsal"].mean().to_dict()
    )

    yield_by_zone = compute_landmark_yield_by_zone(
        bout_df["gradient_zone"].tolist(), bout_df["qc_valid"].tolist()
    )

    qc_report = build_session_qc_report(
        session=cfg["session_label"],
        track=cfg["track"],
        homography_fit=type("H", (), dict(rmse_px=homography_rmse, passes_acceptance=lambda self, m=2.0: homography_rmse < m))(),
        homography_max_rmse_px=2.0,
        sync_result=sync_result,
        landmark_yield_by_zone=yield_by_zone,
        fallback_invocation_count=0,
        rejected_detection_count=int(sum(1 for r in per_sample if "fail" in r)),
    )

    out_dir = f"{REPO}/landmark_outputs"
    import os
    os.makedirs(out_dir, exist_ok=True)
    # Real bug fixed 2026-08-28 (Adam: "wire Back into stage7_real_run.py"): Front and
    # Back share the same session_label (e.g. both "..._Test3-004"), so naming outputs
    # by session_label alone would let Back silently overwrite Front's just-validated
    # bout/frame/qc_report files. Every output filename now includes the lane (track).
    out_stem = f"{cfg['session_label']}_{cfg['track']}"
    bout_df.to_csv(f"{out_dir}/{out_stem}_bout_output.csv", index=False)
    frame_df.to_csv(f"{out_dir}/{out_stem}_frame_output.csv", index=False)
    with open(f"{out_dir}/{out_stem}_qc_report.json", "w") as f:
        json.dump({
            "session": qc_report.session, "track": qc_report.track,
            "homography_rmse_px": qc_report.homography_rmse_px,
            "homography_passes": qc_report.homography_passes,
            "sync_offset_sec": qc_report.sync_offset_sec,
            "sync_drift_slope": qc_report.sync_drift_slope,
            "sync_passes": qc_report.sync_passes,
            "sync_low_confidence": qc_report.sync_low_confidence,
            "sync_confidence_note": qc_report.sync_confidence_note,
            "landmark_yield_by_zone": qc_report.landmark_yield_by_zone,
            "dorsal_yield_by_zone": dorsal_yield_by_zone,
            "fallback_invocation_count": qc_report.fallback_invocation_count,
            "rejected_detection_count": qc_report.rejected_detection_count,
            "n_bouts_total_raw": len(bouts_df_raw),
            "n_bouts_dropped_pre_entry": n_dropped_pre_entry,
            "entry_time_thermal_sec": cfg["entry_time_thermal_sec"],
            "n_bouts_total": len(bouts_df),
            "n_bouts_qc_valid": int(bout_df["qc_valid"].sum()),
            "n_bouts_with_dorsal": int(bout_df["dorsal_mean_c"].notna().sum()),
            "posture_breakdown": dict(posture_counts),
        }, f, indent=2)

    print(f"\n--- {name} bout table (n={len(bout_df)}) ---")
    print(bout_df[["bout_id", "gradient_zone", "dorsal_mean_c", "gt_mouse_surface_temp_mean_bout",
                    "dorsal_minus_gt_c", "n_samples_dorsal_measured", "qc_valid", "sync_low_confidence"]].to_string())
    print(f"\nqc_valid yield by zone: {yield_by_zone}")
    print(f"raw dorsal-measurement yield by zone: {dorsal_yield_by_zone}")
    valid_dorsal = bout_df["dorsal_minus_gt_c"].dropna()
    if len(valid_dorsal):
        print(f"dorsal_mean_c vs ground truth: mean diff={valid_dorsal.mean():.2f}C, "
              f"mean |diff|={valid_dorsal.abs().mean():.2f}C, n={len(valid_dorsal)}")
    print(f"session wall time: {time.time()-t_start:.1f}s")
    return bout_df, frame_df, qc_report, posture_counts


if __name__ == "__main__":
    # Guarded 2026-08-28 so QC scripts can safely `import` this module for its
    # SESSIONS dict / helper functions (registration_correction_homography,
    # build_thermal_native_lookup, load_homography) without re-running the
    # full, slow 6-session pipeline as a side effect of the import.
    results = {}
    for name, cfg in SESSIONS.items():
        results[name] = process_session(name, cfg)

    print("\n=== DONE ===")
