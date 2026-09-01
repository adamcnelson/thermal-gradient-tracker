"""
Compute a full-session RGB-derived mouse centroid track for one lane of one
session (project_brief_v7.md Stage 5, session-wide — not the sparse per-bout
sampling stage7_real_run.py does). This is real ground truth used by
scripts/validate_homography_orientation.py (matches this track against the
thermal-native track to catch a wrong RGB<->thermal orientation that
reprojection RMSE alone cannot) and by src.landmarks.bout_qc's optional RGB
panels.

Real motivation for this being a proper, permanent script (2026-08-28,
Adam): the original Front-lane rgb_track.csv files already on disk in
landmark_outputs/ were produced by an ad hoc scratchpad script during an
earlier stage of this same session (matching this project's usual pattern
of orchestration scripts living in the ephemeral per-session scratchpad,
not the repo) -- but with the Back lane now starting and the whole local
dataset coming next, this needed to stop being disposable. A thorough
repo-wide search (grep, git log -S, every plausible directory) confirmed
no version of it was ever committed.

Output columns (matches the existing Front-lane files' schema exactly):
    rgb_time_sec, rgb_centroid_x, rgb_centroid_y,
    rgb_centroid_x_thermal_aligned, rgb_centroid_y_thermal_aligned,
    rgb_velocity_px_s
The two "_thermal_aligned" columns (the raw centroid warped into thermal
space via a session's homography) are only populated if --homography-json
is given -- they're used by bout_qc.py's diagnostic panels, but NOT by
validate_homography_orientation.py's own matching (it re-warps
rgb_centroid_x/y itself against whichever homography is being tested), so
they're optional here rather than a hard requirement.

Naming: existing Front-lane files on disk predate the lane suffix used by
every other artifact in this pipeline (tracking CSVs, bout CSVs,
homography JSONs all have _Front/_Back). New output from this script
always includes the lane, for consistency going forward -- deliberately
NOT renaming the existing Front files as a side effect of adding this
script.

Usage:
    python scripts/compute_rgb_track.py \\
        --video "/path/to/Process_Jason/07-28-25_Test_3/2025-07-28_10-57-21.mp4" \\
        --lane B \\
        --session-label "07-28-25_4540_B_4541_F_Test3-004" \\
        [--homography-json homography_calibration/<session>_Back_homography.json] \\
        [--sample-hz 1.0] [--output-dir landmark_outputs] [--overwrite]

Output: <output-dir>/<session-label>_<lane>_rgb_track.csv
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.landmarks.registration import apply_homography
from src.landmarks.rgb_landmarks import RgbBackgroundModel, segment_mouse_rgb
from src.landmarks.webcam_preprocessing import LANE_TOP, detect_track_split_row, split_track_crops
from src.logging_utils import setup_logger

MIN_AREA_PX = 200
MAX_AREA_PX = 20000
N_BACKGROUND_FRAMES = 30


def compute_rgb_track(video_path: str, lane: str, sample_hz: float, log,
                       H: "np.ndarray | None" = None) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        rgb_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or rgb_fps <= 0:
            raise ValueError(f"Could not determine fps/frame count for {video_path}")
        duration_sec = total / rgb_fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
        if not ok:
            raise ValueError(f"Could not read the first frame of {video_path}")
        split_row = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
        log.info(f"  Detected track split row: {split_row} (of {first.shape[0]})  "
                 f"rgb_fps={rgb_fps:.3f}  duration={duration_sec:.0f}s")

        bg_indices = np.linspace(0, total - 1, N_BACKGROUND_FRAMES, dtype=int)
        bg_frames = []
        for idx in bg_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(gray, split_row=split_row)
            bg_frames.append(top if lane == LANE_TOP else bottom)
        if not bg_frames:
            raise ValueError(f"Could not build a background model for {video_path}")
        bg_model = RgbBackgroundModel.build(bg_frames)
        log.info(f"  Background model built from {len(bg_frames)} frames")

        sample_times = np.arange(0, duration_sec, 1.0 / sample_hz)
        rows = []
        for i, t in enumerate(sample_times):
            idx = int(round(t * rgb_fps))
            if not (0 <= idx < total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(gray, split_row=split_row)
            crop = top if lane == LANE_TOP else bottom
            mask = segment_mouse_rgb(crop, bg_model, min_area=MIN_AREA_PX, max_area=MAX_AREA_PX)
            if mask is None:
                continue
            ys, xs = np.where(mask)
            rows.append(dict(rgb_time_sec=float(t), rgb_centroid_x=float(xs.mean()), rgb_centroid_y=float(ys.mean())))
            if (i + 1) % 200 == 0:
                log.info(f"  Processed {i + 1}/{len(sample_times)} samples")
    finally:
        cap.release()

    df = pd.DataFrame(rows).sort_values("rgb_time_sec").reset_index(drop=True)
    dt = df["rgb_time_sec"].diff()
    dx = df["rgb_centroid_x"].diff()
    dy = df["rgb_centroid_y"].diff()
    df["rgb_velocity_px_s"] = np.hypot(dx, dy) / dt

    if H is not None and len(df):
        pts = df[["rgb_centroid_x", "rgb_centroid_y"]].to_numpy()
        warped = apply_homography(H, pts)
        df["rgb_centroid_x_thermal_aligned"] = warped[:, 0]
        df["rgb_centroid_y_thermal_aligned"] = warped[:, 1]
    else:
        df["rgb_centroid_x_thermal_aligned"] = np.nan
        df["rgb_centroid_y_thermal_aligned"] = np.nan

    return df[["rgb_time_sec", "rgb_centroid_x", "rgb_centroid_y",
               "rgb_centroid_x_thermal_aligned", "rgb_centroid_y_thermal_aligned",
               "rgb_velocity_px_s"]]


def main():
    parser = argparse.ArgumentParser(description="Compute a full-session RGB centroid track for one lane")
    parser.add_argument("--video", required=True, help="Path to the combined-lane RGB .mp4")
    parser.add_argument("--lane", required=True, choices=["F", "B"], help="Which lane to track")
    parser.add_argument("--session-label", required=True,
                         help="Session identifier for the output filename, e.g. 07-28-25_4540_B_4541_F_Test3-004")
    parser.add_argument("--homography-json", default=None,
                         help="Optional homography JSON -- if given, also fills the "
                              "*_thermal_aligned columns (used by bout_qc.py panels)")
    parser.add_argument("--sample-hz", type=float, default=1.0, help="Sampling rate in Hz (default: 1.0)")
    parser.add_argument("--output-dir", default=None, help="Default: landmark_outputs/ under the project root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger()

    video_path = Path(args.video)
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else paths.LANDMARK_OUTPUTS_DIR
    paths.ensure_dir(out_dir)
    out_path = out_dir / f"{args.session_label}_{args.lane}_rgb_track.csv"
    if out_path.exists() and not args.overwrite:
        log.error(f"{out_path} already exists (pass --overwrite to replace it)")
        sys.exit(1)

    H = None
    if args.homography_json:
        H = np.array(json.load(open(args.homography_json))["H"], dtype=np.float64)
        log.info(f"  Loaded homography from {args.homography_json}")

    log.info(f"Processing: {video_path.name} (lane {args.lane})")
    df = compute_rgb_track(str(video_path), args.lane, args.sample_hz, log, H=H)
    df.to_csv(out_path, index=False)
    log.info(f"Done. {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
