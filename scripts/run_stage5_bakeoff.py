"""
Stage 5 bake-off runner (project_brief_v7.md §6 Stage 5 "Bake-off protocol").

Real driver for src/landmarks/bakeoff.py's metrics #1 (landmark pixel
error in RGB) and #2 (thermal-value error vs. human-drawn ROI), run
against every real, committed validation_labels/*.json file. Metric #3
(yield stratified by gradient zone) is a property of a full session's
bouts, not a single labeled frame -- it comes from
outputs.compute_landmark_yield_by_zone() against real Stage 7 batch
output (see landmark_outputs/*_qc_report.json's landmark_yield_by_zone),
reported here for context but not recomputed.

IMPORTANT: no supervised-pose fallback (SLEAP/DeepLabCut) exists yet, so
this reports the classical method's own accuracy against real human
ground truth -- not yet a "classical vs. supervised" comparison in the
brief's literal sense. Re-run this script once a fallback method exists
and/or once the validation set grows past its current n=3 real labels
(target: 75, brief §8).

Usage:
    python scripts/run_stage5_bakeoff.py
"""

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.landmarks.webcam_preprocessing import split_track_crops, detect_track_split_row
from src.landmarks.rgb_landmarks import RgbBackgroundModel, segment_mouse_rgb, extract_mouse_detection
from src.landmarks.registration import apply_homography
from src.landmarks.thermal_measurement import anterior_region_mask, warp_mask_to_thermal, warm_spot_temperature
from src.landmarks.bakeoff import compute_landmark_pixel_error, compute_thermal_value_error

REPO = Path(__file__).resolve().parent.parent
MIN_AREA = 200
MAX_AREA = 20000
OUT_DIR = REPO / "landmark_outputs" / "bakeoff"

HOMOGRAPHY_BY_SEQ = {
    "07-28-25_4540_B_4541_F_Test3-004_Front.seq": REPO / "homography_calibration" / "07-28-25_4540_B_4541_F_Test3-004_Front_homography.json",
    "07-30-25_4540_F_4541_B_Test4-008_Front.seq": REPO / "homography_calibration" / "07-30-25_4540_F_4541_B_Test4-008_Front_homography.json",
}


def load_homography(seq_path: str) -> np.ndarray:
    name = Path(seq_path).name
    d = json.load(open(HOMOGRAPHY_BY_SEQ[name]))
    return np.array(d["H"], dtype=np.float64)


def build_rgb_background(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    split_row = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
    bg_indices = np.linspace(0, total - 1, 30, dtype=int)
    bg_frames = []
    for idx in bg_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _top, bottom = split_track_crops(gray, split_row=split_row)
            bg_frames.append(bottom)
    cap.release()
    return RgbBackgroundModel.build(bg_frames), split_row


def get_rgb_frame(video_path: str, split_row: int, rgb_time_sec: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = int(round(rgb_time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _top_g, crop_gray = split_track_crops(gray, split_row=split_row)
    _top_c, crop_bgr = split_track_crops(frame, split_row=split_row)
    return crop_gray, crop_bgr


def get_thermal_frame(seq_path: str, thermal_time_sec: float, fps: float = 8.0):
    target_idx = int(round(thermal_time_sec * fps))
    planck = read_planck_constants(seq_path)
    reader = SeqReader(seq_path)
    result = None
    for idx, raw in reader.frames():
        if idx == target_idx:
            result = raw_to_celsius(raw, planck)
            break
        if idx > target_idx:
            break
    reader.close()
    return result


def save_overlay(label_name, crop_bgr, mask, lm, label_head_xy, label_tail_xy, out_dir):
    img = crop_bgr.copy()
    if mask is not None:
        overlay = img.copy()
        overlay[mask] = (60, 60, 220)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    if lm is not None:
        pts = np.array([[p[1], p[0]] for p in lm.path_nose_to_tail], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(60, 200, 60), thickness=1)
        cv2.circle(img, (int(lm.nose_point[1]), int(lm.nose_point[0])), 4, (0, 230, 255), -1)  # algo nose: yellow
        cv2.circle(img, (int(lm.tail_base_point[1]), int(lm.tail_base_point[0])), 4, (255, 60, 230), -1)  # algo tail: magenta
    if label_head_xy is not None:
        cv2.drawMarker(img, (int(label_head_xy[0]), int(label_head_xy[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)  # human head: red X
    if label_tail_xy is not None:
        cv2.drawMarker(img, (int(label_tail_xy[0]), int(label_tail_xy[1])), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)  # human tail: blue X
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, "circles=algo(yellow=nose,magenta=tail)  X=human(red=head,blue=tail)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{label_name}_rgb_compare.png"), img)


def save_thermal_overlay(label_name, thermal_celsius, warped_anterior, human_roi_mask, out_dir):
    vis = ((thermal_celsius - thermal_celsius.min()) / max(np.ptp(thermal_celsius), 1e-6) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if warped_anterior is not None:
        vis[warped_anterior] = (0, 230, 255)  # algo anterior region: yellow
    if human_roi_mask is not None:
        edge = cv2.Canny(human_roi_mask.astype(np.uint8) * 255, 50, 150)
        vis[edge > 0] = (0, 0, 255)  # human ROI outline: red
    vis = cv2.resize(vis, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis, "yellow fill=algo anterior region  red outline=human warm-spot ROI",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{label_name}_thermal_compare.png"), vis)


def _git_tracked_label_paths():
    """Only real, committed, human-confirmed labels -- validation_labels/
    also holds a handful of uncommitted, deliberately-abandoned attempts
    (wrong-sync mismatches, empty 0-point labels) that must not be
    silently included in a real bake-off. See commit a0d5663's message
    for why each committed label was kept and each other was not."""
    import subprocess
    out = subprocess.run(
        ["git", "ls-files", "validation_labels/"], cwd=REPO,
        capture_output=True, text=True, check=True,
    )
    return sorted(REPO / line for line in out.stdout.splitlines() if line.strip())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_paths = _git_tracked_label_paths()
    print(f"found {len(label_paths)} real (git-committed) label files: {[p.name for p in label_paths]}\n")

    bg_cache = {}
    pixel_error_rows = []
    thermal_error_rows = []

    for label_path in label_paths:
        label = json.load(open(label_path))
        label_name = label_path.stem
        seq_path = label["seq_path"]
        video_path = label["video_path"]
        posture = label.get("posture")
        rgb_t = label["rgb_frame_time_sec"]
        thermal_t = label["thermal_frame_time_sec"]

        print(f"=== {label_name} (posture={posture}) ===")

        if video_path not in bg_cache:
            bg_cache[video_path] = build_rgb_background(video_path)
        bg_model, split_row = bg_cache[video_path]

        crop_gray, crop_bgr = get_rgb_frame(video_path, split_row, rgb_t)
        if crop_gray is None:
            print("  FAILED to read RGB frame, skipping")
            continue

        mask = segment_mouse_rgb(crop_gray, bg_model, min_area=MIN_AREA, max_area=MAX_AREA)
        detection = extract_mouse_detection(mask, MIN_AREA, MAX_AREA) if mask is not None else None
        lm = detection.landmarks if (detection is not None and detection.posture == "extended") else None

        pe = compute_landmark_pixel_error(
            label_name, posture,
            label_head_point_xy=label.get("head_point"),
            label_nose_point_xy=label.get("nose_point"),
            label_tail_base_point_xy=label.get("tail_base_point"),
            algo_nose_point=lm.nose_point if lm else None,
            algo_tail_base_point=lm.tail_base_point if lm else None,
        )
        pixel_error_rows.append(pe)

        def _fmt(v):
            return f"{v:.1f}px" if v is not None else "None"

        msg = (f"  landmark pixel error: nose={_fmt(pe.nose_error_px)} "
               f"head={_fmt(pe.head_error_px)} tail_base={_fmt(pe.tail_base_error_px)}")
        if pe.reason:
            msg += f"  ({pe.reason})"
        if pe.caveat:
            msg += f"\n    CAVEAT: {pe.caveat}"
        print(msg)

        thermal_celsius = get_thermal_frame(seq_path, thermal_t)
        algo_warm_spot_c = None
        warped_anterior = None
        if thermal_celsius is not None and lm is not None:
            H = load_homography(seq_path)
            anterior = anterior_region_mask(mask, lm.path_nose_to_tail[: lm.tail_base_index + 1])
            warped_anterior = warp_mask_to_thermal(anterior, H, thermal_celsius.shape)
            algo_warm_spot_c = warm_spot_temperature(thermal_celsius, warped_anterior)

        if thermal_celsius is not None:
            te = compute_thermal_value_error(
                label_name, posture, thermal_celsius,
                human_roi_polygon_xy=label.get("warm_spot_roi_polygon_thermal"),
                algo_warm_spot_c=algo_warm_spot_c,
            )
            thermal_error_rows.append(te)
            print(f"  thermal value error: human_roi_p95={te.human_roi_p95_c} algo_warm_spot={te.algo_warm_spot_c} "
                  f"error={te.error_c} ({te.reason})")

            human_roi_mask = None
            if label.get("warm_spot_roi_polygon_thermal"):
                from src.landmarks.bakeoff import polygon_mask_xy
                human_roi_mask = polygon_mask_xy(label["warm_spot_roi_polygon_thermal"], thermal_celsius.shape)
            save_thermal_overlay(label_name, thermal_celsius, warped_anterior, human_roi_mask, OUT_DIR)
        else:
            print("  FAILED to read thermal frame")

        save_overlay(label_name, crop_bgr, mask, lm, label.get("head_point"), label.get("tail_base_point"), OUT_DIR)
        print()

    # ── summary ──────────────────────────────────────────────────────────
    pe_df = pd.DataFrame([vars(p) for p in pixel_error_rows])
    te_df = pd.DataFrame([vars(t) for t in thermal_error_rows])
    pe_df.to_csv(OUT_DIR / "metric1_landmark_pixel_error.csv", index=False)
    te_df.to_csv(OUT_DIR / "metric2_thermal_value_error.csv", index=False)

    print("=== SUMMARY (metric 1: landmark pixel error, RGB) ===")
    print(pe_df[["label_name", "posture", "nose_error_px", "head_error_px", "tail_base_error_px", "caveat"]].to_string())
    valid_nose = pe_df["nose_error_px"].dropna()
    valid_head = pe_df["head_error_px"].dropna()
    valid_tail = pe_df["tail_base_error_px"].dropna()
    print(f"\nnose error (real nose-tip labels only): n={len(valid_nose)} "
          f"mean={valid_nose.mean() if len(valid_nose) else float('nan'):.2f}px")
    print(f"head error (head_point, NOT the same landmark as nose): n={len(valid_head)} "
          f"mean={valid_head.mean() if len(valid_head) else float('nan'):.2f}px")
    print(f"tail_base error: n={len(valid_tail)} mean={valid_tail.mean() if len(valid_tail) else float('nan'):.2f}px")

    print("\n=== SUMMARY (metric 2: thermal value error vs. human ROI) ===")
    print(te_df[["label_name", "posture", "human_roi_p95_c", "algo_warm_spot_c", "error_c"]].to_string())
    valid_err = te_df["error_c"].dropna()
    print(f"\nerror_c: n={len(valid_err)} mean={valid_err.mean() if len(valid_err) else float('nan'):.2f}C "
          f"mean_abs={valid_err.abs().mean() if len(valid_err) else float('nan'):.2f}C")

    print(f"\nNOTE: n={len(label_paths)} real committed labels (target 75, brief §8). "
          f"No supervised-pose fallback exists -- this measures classical CV's own accuracy "
          f"against human ground truth, not yet a classical-vs-supervised comparison.")
    print(f"\nVisual overlays + CSVs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
