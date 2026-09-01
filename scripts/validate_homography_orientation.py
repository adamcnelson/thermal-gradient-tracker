"""
Validate a session's homography orientation against real, independently-
tracked motion (project_brief_v7.md §6 Stage 4 — orientation QC gate,
added 2026-08-25).

Real, confirmed bug this exists to catch automatically: Test_4's
calibration was clicked twice, independently and carefully, both times
assuming the rig's "vertical flip only" convention (documented, and
correct, for Test_3) — but Test_4's true relationship turned out to be a
full 180-degree rotation (confirmed via a direct photographic comparison,
thermalFeatures/Track_Alignment_Test4.pptx), most likely because the
thermal camera was physically repositioned between the two recording
days while the RGB webcam mounting stayed fixed. Neither calibration
attempt's own reprojection RMSE caught this — a human clicking "the same
physical corner" consistently wrong under the wrong mental model produces
a homography that fits its own (mismatched) points just fine. The only
real check is against independent ground truth: does the fitted mapping
agree with real, simultaneously-observed motion in both modalities?

This script builds that independent ground truth from real data already
on disk for a session (a full RGB track CSV — see
landmark_outputs/*_rgb_track.csv, and the corrected thermal tracking
CSV/bout table) and calls
src.landmarks.registration.validate_homography_orientation(). It does NOT
compute the RGB track itself (expensive, full-video segmentation) —
generate that first (see the RGB track computation used for
src/landmarks/bout_qc.py's RGB panels).

Usage:
    python scripts/validate_homography_orientation.py \\
        --homography-json homography_calibration/<session>_Front_homography.json \\
        --tracking-csv trackingOutputs/<session>_Front_tracking_every10frames.csv \\
        --bouts-csv bouts/qc_plots/<session>_Front_bout_rows.csv \\
        --rgb-track-csv landmark_outputs/<session>_rgb_track.csv \\
        --entry-time-thermal-sec 71.83 \\
        --sync-offset-sec 11.0 \\
        [--sync-drift-slope 0.0] \\
        [--crop-width 1920] \\
        [--fix]

--fix : if the check flags an orientation mismatch, write a corrected
    homography.json (rgb_points x-mirrored about --crop-width, refit) to
    the same path, after backing up the flagged one. Does NOT write
    anything if the check does not flag a mismatch.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.landmarks.bout_gating import filter_bouts_after_entry
from src.landmarks.outputs import thermal_time_to_rgb_time
from src.landmarks.registration import fit_homography, reprojection_error, validate_homography_orientation
from src.landmarks.sync import WindowedSyncResult

MIN_SAMPLES_PER_SIDE = 20  # per bout, for a "long, stable" validation window


def build_matched_pairs(tracking_csv, bouts_csv, rgb_track_csv, entry_time_thermal_sec, sync_result):
    bouts_raw = pd.read_csv(bouts_csv)
    bouts = filter_bouts_after_entry(bouts_raw, entry_time_sec=entry_time_thermal_sec)
    trk = pd.read_csv(tracking_csv)
    trk_ok = trk[trk["qc_flag"] == "ok"]
    rgb = pd.read_csv(rgb_track_csv)

    matched_rgb, matched_thermal, bout_mids = [], [], []
    for _, b in bouts.iterrows():
        t_sub = trk_ok[
            (trk_ok["elapsed_time_sec"] >= b["bout_start_sec"]) & (trk_ok["elapsed_time_sec"] <= b["bout_end_sec"])
        ]
        rgb_start = thermal_time_to_rgb_time(b["bout_start_sec"], sync_result)
        rgb_end = thermal_time_to_rgb_time(b["bout_end_sec"], sync_result)
        r_sub = rgb[(rgb["rgb_time_sec"] >= rgb_start) & (rgb["rgb_time_sec"] <= rgb_end)]
        if len(t_sub) < MIN_SAMPLES_PER_SIDE or len(r_sub) < MIN_SAMPLES_PER_SIDE:
            continue
        matched_thermal.append([t_sub["mouse_centroid_x"].median(), t_sub["mouse_centroid_y"].median()])
        matched_rgb.append([r_sub["rgb_centroid_x"].median(), r_sub["rgb_centroid_y"].median()])
        bout_mids.append((b["bout_start_sec"] + b["bout_end_sec"]) / 2)

    return np.array(matched_rgb), np.array(matched_thermal), bout_mids


def main():
    parser = argparse.ArgumentParser(description="Validate a session's homography orientation against real tracked motion")
    parser.add_argument("--homography-json", required=True)
    parser.add_argument("--tracking-csv", required=True)
    parser.add_argument("--bouts-csv", required=True)
    parser.add_argument("--rgb-track-csv", required=True)
    parser.add_argument("--entry-time-thermal-sec", type=float, required=True)
    parser.add_argument("--sync-offset-sec", type=float, required=True)
    parser.add_argument("--sync-drift-slope", type=float, default=0.0)
    parser.add_argument("--crop-width", type=float, default=1920.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--fix", action="store_true", help="write a corrected (x-mirrored) homography.json if flagged")
    args = parser.parse_args()

    sync_result = WindowedSyncResult(
        window_centers_sec=np.array([]), window_lags_sec=np.array([]),
        offset_sec=args.sync_offset_sec, drift_slope=args.sync_drift_slope,
        r_squared=float("nan"), residual_max_sec=float("nan"),
    )

    matched_rgb, matched_thermal, bout_mids = build_matched_pairs(
        args.tracking_csv, args.bouts_csv, args.rgb_track_csv, args.entry_time_thermal_sec, sync_result,
    )
    print(f"Built {len(matched_rgb)} independent bout-level validation pairs "
          f"(long, stable bouts, >={MIN_SAMPLES_PER_SIDE} samples/side each).")

    homography_path = Path(args.homography_json)
    d = json.loads(homography_path.read_text())
    rgb_points = np.array(d["rgb_points"], dtype=np.float64)
    thermal_points = np.array(d["thermal_points"], dtype=np.float64)

    result = validate_homography_orientation(
        rgb_points, thermal_points, matched_rgb, matched_thermal,
        crop_width=args.crop_width, min_samples=args.min_samples,
    )

    print()
    print(f"recommendation: {result.recommendation}")
    print(f"flagged: {result.flagged}")
    print(f"mean_err_as_clicked_px: {result.mean_err_as_clicked_px:.1f}")
    print(f"mean_err_x_mirrored_px: {result.mean_err_x_mirrored_px:.1f}")
    print(result.note)

    if not result.flagged:
        print("\nNo action needed.")
        return

    if not args.fix:
        print("\nRe-run with --fix to write a corrected homography.json (backs up the flagged one first).")
        return

    backup_dir = homography_path.parent / f"_{homography_path.stem}_flagged_backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / homography_path.name
    backup_path.write_text(homography_path.read_text())
    print(f"\nBacked up flagged homography to {backup_path}")

    rgb_points_fixed = rgb_points.copy()
    rgb_points_fixed[:, 0] = args.crop_width - rgb_points_fixed[:, 0]
    fit = fit_homography(rgb_points_fixed, thermal_points)
    rmse, residuals = reprojection_error(fit.H, rgb_points_fixed, thermal_points)

    d["rgb_points"] = rgb_points_fixed.tolist()
    d["H"] = fit.H.tolist()
    d["rmse_px"] = float(rmse)
    d["residuals_px"] = residuals.tolist()
    d["passes_acceptance"] = bool(rmse < 1.0)
    d["orientation_note"] = (
        f"Auto-corrected by scripts/validate_homography_orientation.py --fix: "
        f"validate_homography_orientation() found the as-clicked correspondence flagged "
        f"(mean_err_as_clicked_px={result.mean_err_as_clicked_px:.1f} vs "
        f"mean_err_x_mirrored_px={result.mean_err_x_mirrored_px:.1f} against "
        f"{len(matched_rgb)} independent bout-level validation pairs). rgb_points stored here "
        f"are PRE-MIRRORED (crop_width - raw_clicked_x); this H reproduces the corrected mapping "
        f"directly via fit_homography() on the stored points."
    )
    homography_path.write_text(json.dumps(d, indent=2))
    print(f"Wrote corrected homography to {homography_path} (RMSE={rmse:.3f}px)")


if __name__ == "__main__":
    main()
