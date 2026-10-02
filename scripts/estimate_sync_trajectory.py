"""
Estimate the RGB<->thermal sync (offset, drift, true thermal fps) for a session lane
by aligning two independent mouse trajectories in thermal pixel space -- the
thermal-native track from Stage 1 tracking vs the RGB track warped through the
lane's homography. See src.landmarks.sync.estimate_sync_from_trajectories.

Built for the sessions that have a homography (homography_calibration_inherited/)
but no human-anchored sync_result. Validated 2026-10-01 against Test_3/4/7's
human-anchored offsets (6 lanes, all within ~1s, fps=10 picked every time) using
the same SLURM tracking + Alcova-crop homography inputs as the batch run --
rerun with --validate.

Flags (provisional thresholds, set from the validation run's worst lane:
loss 0.085, peak_ratio 0.12):
    flat_minimum      peak_ratio > 0.3  -- another offset >15s away fits nearly as well
    poor_fit          loss > 0.25       -- even the best alignment disagrees often
    timing_anomaly    |drift| > 0.002   -- real clocks don't drift this much; points
                                           to a timing bug (e.g. bad video seeking)
    fps_ambiguous     other_fps_loss < 2 * loss

Usage:
    # all inherited-homography lanes (computes missing RGB tracks, --jobs in parallel)
    python scripts/estimate_sync_trajectory.py --jobs 8
    # reproduce the Test_3/4/7 validation
    python scripts/estimate_sync_trajectory.py --validate
    # one lane
    python scripts/estimate_sync_trajectory.py --tracking-csv T.csv --rgb-track-csv R.csv --homography-json H.json

Output (batch/validate): <output-dir>/sync_estimates[_validation].csv
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.landmarks.registration import apply_homography
from src.landmarks.sync import estimate_sync_from_trajectories
from src.logging_utils import setup_logger

REPO = Path(__file__).resolve().parent.parent
ALCOVA_TG = "/Volumes/alcova/bedfordlab/ThermalGradient"  # local Mac mount; on MedicineBow pass the /cluster/alcova paths
DEFAULT_TRACKING_DIR = f"{ALCOVA_TG}/SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/trackingOutputs"
DEFAULT_RGB_VIDEO_ROOT = f"{ALCOVA_TG}/Process_Jason"
MAX_VALIDATION_ERROR_SEC = 2.0
DEFAULT_OUTPUT_DIR = REPO / "thermalFeatures" / "trajectory_sync"
LANE_NAMES = {"F": "Front", "B": "Back"}
VALIDATION_SESSIONS = {  # stage7_sessions_config.json key -> homography dir in the Alcova crop convention
    "Test_3": "homography_calibration_alcova_crop", "Test_4": "homography_calibration",
    "Test_7": "homography_calibration",
}


def load_thermal_track(tracking_csv):
    d = pd.read_csv(tracking_csv)
    d = d[(d.tracking_confidence == 1.0) & d.mouse_centroid_x.notna()]
    return d.frame_number.to_numpy(float), d.mouse_centroid_x.to_numpy(float)


def load_rgb_track_warped(rgb_track_csv, H):
    r = pd.read_csv(rgb_track_csv).dropna(subset=["rgb_centroid_x"])
    warped = apply_homography(H, r[["rgb_centroid_x", "rgb_centroid_y"]].to_numpy())
    return r.rgb_time_sec.to_numpy(float), warped[:, 0]


def flags_for(res):
    flags = []
    if not res.peak_ratio <= 0.3:
        flags.append("flat_minimum")
    if res.loss > 0.25:
        flags.append("poor_fit")
    if abs(res.drift_slope) > 0.002:
        flags.append("timing_anomaly")
    if res.other_fps_loss < 2 * res.loss:
        flags.append("fps_ambiguous")
    return flags


def estimate_lane(tracking_csv, rgb_track_csv, homography_json):
    H = np.array(json.load(open(homography_json))["H"], dtype=np.float64)
    frames, thermal_x = load_thermal_track(tracking_csv)
    rgb_t, rgb_x = load_rgb_track_warped(rgb_track_csv, H)
    res = estimate_sync_from_trajectories(frames, thermal_x, rgb_t, rgb_x)
    return {**asdict(res), "flags": ";".join(flags_for(res))}


def _compute_rgb_track_job(video, lane, out_csv):
    from compute_rgb_track import compute_rgb_track
    import logging
    df = compute_rgb_track(video, lane, 1.0, logging.getLogger("rgb_track"))
    df.to_csv(out_csv, index=False)
    return out_csv


def inherited_lanes(tracking_dir, rgb_track_dir, rgb_video_root):
    lanes = []
    for hj in sorted((REPO / "homography_calibration_inherited").glob("*_homography.json")):
        meta = json.load(open(hj))
        stem = hj.name.replace("_homography.json", "")  # <session>_<Front|Back>
        session = stem.rsplit("_", 1)[0]
        # video_path was recorded on the Mac mount; keep only the part under Process_Jason/
        video_rel = meta["video_path"].split("/Process_Jason/", 1)[1]
        lanes.append(dict(
            session=session, lane=meta["lane"], homography_json=str(hj),
            video=f"{rgb_video_root}/{video_rel}",
            tracking_csv=f"{tracking_dir}/{stem}_tracking_every10frames.csv",
            rgb_track_csv=str(Path(rgb_track_dir) / f"{session}_{meta['lane']}_rgb_track.csv"),
            regime=meta.get("regime", ""),
        ))
    return lanes


def validation_lanes(tracking_dir, rgb_track_dir, rgb_video_root):
    cfg = json.load(open(REPO / "stage7_sessions_config.json"))
    lanes = []
    for key, hom_dir in VALIDATION_SESSIONS.items():
        label = cfg[key]["session_label"]
        for lane, name in LANE_NAMES.items():
            # reuse the Stage 7 track if this machine has one (landmark_outputs/ isn't in git)
            existing = REPO / "landmark_outputs" / f"{label}_{lane}_rgb_track.csv"
            lanes.append(dict(
                session=label, lane=lane, truth_offset_sec=cfg[key]["sync_result"]["offset_sec"],
                homography_json=str(REPO / hom_dir / f"{label}_{name}_homography.json"),
                video=f"{rgb_video_root}/{cfg[key]['rgb_video_relpath']}",
                tracking_csv=f"{tracking_dir}/{label}_{name}_tracking_every10frames.csv",
                rgb_track_csv=str(existing if existing.exists()
                                  else Path(rgb_track_dir) / f"{label}_{lane}_rgb_track.csv"),
            ))
    return lanes


def run_batch(lanes, jobs, log):
    missing = [l for l in lanes if not Path(l["rgb_track_csv"]).exists()]
    if missing:
        log.info(f"Computing {len(missing)} missing RGB tracks ({jobs} parallel)")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compute_rgb_track_job, l["video"], l["lane"], l["rgb_track_csv"]): l
                       for l in missing}
            for i, fut in enumerate(futures, 1):
                l = futures[fut]
                try:
                    fut.result()
                    log.info(f"  [{i}/{len(missing)}] RGB track done: {l['session']} {l['lane']}")
                except Exception as e:  # keep going; the lane is reported as failed below
                    log.error(f"  [{i}/{len(missing)}] RGB track FAILED: {l['session']} {l['lane']}: {e}")

    rows = []
    for l in lanes:
        row = dict(l)
        try:
            row.update(estimate_lane(l["tracking_csv"], l["rgb_track_csv"], l["homography_json"]))
        except Exception as e:
            row["error"] = str(e)
        if "truth_offset_sec" in l and "offset_sec" in row:
            row["error_vs_truth_sec"] = row["offset_sec"] - l["truth_offset_sec"]
        log.info(f"  {l['session']} {l['lane']}: offset={row.get('offset_sec', float('nan')):+.2f}s "
                 f"fps={row.get('camera_fps', float('nan')):.0f} drift={row.get('drift_slope', float('nan')):+.4f} "
                 f"loss={row.get('loss', float('nan')):.3f} peak_ratio={row.get('peak_ratio', float('nan')):.2f} "
                 f"{row.get('flags', '')}{row.get('error', '')}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Trajectory-alignment RGB<->thermal sync estimation")
    parser.add_argument("--validate", action="store_true", help="Run on Test_3/4/7 and compare to human truth")
    parser.add_argument("--tracking-csv")
    parser.add_argument("--rgb-track-csv")
    parser.add_argument("--homography-json")
    parser.add_argument("--tracking-dir", default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--rgb-video-root", default=DEFAULT_RGB_VIDEO_ROOT,
                        help="Process_Jason root holding the session .mp4 folders")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--jobs", type=int, default=4, help="Parallel RGB-track computations (batch mode)")
    args = parser.parse_args()
    log = setup_logger()

    if args.tracking_csv:
        print(json.dumps(estimate_lane(args.tracking_csv, args.rgb_track_csv, args.homography_json), indent=2))
        return

    out_dir = Path(args.output_dir)
    rgb_track_dir = out_dir / "rgb_tracks"
    rgb_track_dir.mkdir(parents=True, exist_ok=True)
    if args.validate:
        df = run_batch(validation_lanes(args.tracking_dir, rgb_track_dir, args.rgb_video_root), args.jobs, log)
        out = out_dir / "sync_estimates_validation.csv"
    else:
        df = run_batch(inherited_lanes(args.tracking_dir, rgb_track_dir, args.rgb_video_root), args.jobs, log)
        out = out_dir / "sync_estimates.csv"
    df.to_csv(out, index=False)
    log.info(f"Done. {len(df)} lanes -> {out}")

    if args.validate:
        err = df.get("error_vs_truth_sec", pd.Series(np.nan, index=df.index)).abs()
        bad = df[~(err <= MAX_VALIDATION_ERROR_SEC)]
        if len(bad):
            log.error(f"VALIDATION FAILED: {len(bad)} lane(s) missing or >{MAX_VALIDATION_ERROR_SEC}s from truth")
            sys.exit(1)
        log.info(f"Validation passed: max |error| {err.max():.2f}s")


if __name__ == "__main__":
    main()
