"""
Detect each lane's mouse-entry time from the RGB video (the hand/arm drop-in,
then a sustained mouse-only detection) and convert it to the thermal clock with
the session's sync -- the entry_time_thermal_sec Stage 7 needs per lane.

Detection logic is src.landmarks.entry_detection (validated by eye in v7 on
Test_3/Test_4), plus its changed-fraction intrusion rule (2026-10-05). Entry =
the mouse alone after the hands have left THAT lane's view -- a few seconds
later than v7's "mouse first appears" values when handling continues. This driver: one sequential decode pass over the first
--scan-sec of the video (frame-accurate; cv2 seeking is not -- see
src/landmarks/video_io.py), both lane crops classified at --sample-hz, entry =
first sustained (--min-sustained-sec) "mouse" run after an "intrusion".
thermal_entry = (rgb_entry - offset) / (1 + drift).

For every lane it also writes a contact sheet (1Hz thumbnails around the
detected entry) so a person can confirm the drop-in and release at a glance.

Modes:
    --validate  Test_3/4/7 (both lanes), sync from stage7_sessions_config.json;
                Front lanes compared with their human-confirmed RGB entry times
                (Back lanes have no confirmed value -- check their contact sheets).
    (default)   every session in thermalFeatures/trajectory_sync/session_sync.csv.

Outputs (in --output-dir, default thermalFeatures/entry_detection/):
    entry_times[_validation].csv, contact_sheets/<session>_<lane>_entry.png
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.landmarks.entry_detection import classify_frame_state_rgb, find_entry_index
from src.landmarks.rgb_landmarks import RgbBackgroundModel
from src.landmarks.video_io import SequentialFrameReader
from src.landmarks.webcam_preprocessing import LANE_TOP, detect_track_split_row, split_track_crops

REPO = Path(__file__).resolve().parent.parent
SYNC_DIR = REPO / "thermalFeatures" / "trajectory_sync"
DEFAULT_RGB_VIDEO_ROOT = "/Volumes/alcova/bedfordlab/ThermalGradient/Process_Jason"
DEFAULT_OUTPUT_DIR = REPO / "thermalFeatures" / "entry_detection"
MIN_AREA, MAX_AREA = 200, 20000  # same mouse-size bounds as compute_rgb_track.py / stage7_real_run.py
# Also call a frame "intrusion" when >5% of the lane differs from background (light gloves/arms escape
# the dark-blob area rule -- see classify_frame_state_rgb). Measured: mouse alone <=3.2%, hand/arm 5-62%.
# Below 0.4% changed a lane counts as empty (static artifacts read as "mouse"; empty lanes measure 0-0.2%).
MOUSE_MIN_CHANGED_FRACTION = 0.004
THUMB_SCALE = 0.25
SHEET_BEFORE_SEC, SHEET_AFTER_SEC = 8, 7


def lane_crop(gray, split_row, lane):
    top, bottom = split_track_crops(gray, split_row=split_row)
    return top if lane == LANE_TOP else bottom


def detect_session(video, lanes, scan_sec, sample_hz, min_sustained_sec, changed_fraction, min_intrusion_sec):
    """{lane: dict(rgb_entry_sec | None, n_intrusion, first_intrusion_sec, thumbs={sec: img})}."""
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    split = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
    bg = {l: [] for l in lanes}
    for i in np.linspace(0, n - 1, 30, dtype=int):  # median background: seeking is fine here
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            for l in lanes:
                bg[l].append(lane_crop(g, split, l))
    cap.release()
    models = {l: RgbBackgroundModel.build(bg[l]) for l in lanes}

    step = max(1, int(round(fps / sample_hz)))
    last = min(n, int(scan_sec * fps))
    states = {l: [] for l in lanes}
    thumbs = {l: {} for l in lanes}
    times = []
    reader = SequentialFrameReader(video)
    try:
        for idx in range(0, last, step):
            ok, f = reader.read(idx)
            if not ok:
                raise IOError(f"Video decode stopped at frame {idx} ({video})")
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            t = idx / fps
            times.append(t)
            for l in lanes:
                crop = lane_crop(g, split, l)
                states[l].append(classify_frame_state_rgb(
                    crop, models[l], MIN_AREA, MAX_AREA, intrusion_changed_fraction=changed_fraction,
                    min_changed_fraction_for_mouse=MOUSE_MIN_CHANGED_FRACTION))
                if idx % int(round(fps)) < step:  # ~1 Hz thumbnails for the contact sheet
                    thumbs[l][int(round(t))] = cv2.resize(crop, None, fx=THUMB_SCALE, fy=THUMB_SCALE)
    finally:
        reader.release()

    min_run = max(1, int(round(min_sustained_sec * sample_hz)))
    min_intr = max(1, int(round(min_intrusion_sec * sample_hz)))
    # Both mice go in during one handling episode; a faint arm in one lane's crop is covered by the other's.
    own = {l: first_qualifying_intrusion(states[l], min_intr) for l in lanes}
    episode = min((v for v in own.values() if v is not None), default=None)
    out = {}
    for l in lanes:
        k = find_entry_index(states[l], min_sustained_detections=min_run, min_intrusion_run=min_intr,
                             intrusion_seen_from=episode)
        intr = [times[i] for i, s in enumerate(states[l]) if s == "intrusion"]
        out[l] = dict(rgb_entry_sec=None if k is None else round(times[k], 2), n_intrusion=len(intr),
                      entry_via_other_lane=k is not None and (own[l] is None or own[l] > k),
                      first_intrusion_sec=round(intr[0], 2) if intr else None, thumbs=thumbs[l])
    return out


def first_qualifying_intrusion(states, min_run):
    """Index at which a run of >= min_run consecutive intrusion states completes, or None."""
    n = 0
    for i, s in enumerate(states):
        n = n + 1 if s == "intrusion" else 0
        if n >= min_run:
            return i
    return None


def contact_sheet(thumbs, entry_sec, title, path):
    """1Hz thumbnails from entry-8s to entry+7s (or the first 16s if no entry), stacked as a column."""
    center = int(round(entry_sec)) if entry_sec is not None else SHEET_BEFORE_SEC
    tiles = []
    for s in range(center - SHEET_BEFORE_SEC, center + SHEET_AFTER_SEC + 1):
        img = thumbs.get(s)
        if img is None:
            continue
        tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color = (0, 200, 0) if s == center and entry_sec is not None else (255, 255, 255)
        cv2.putText(tile, f"t={s}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if s == center and entry_sec is not None:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 200, 0), 2)
        tiles.append(tile)
    if not tiles:
        return
    sheet = np.vstack(tiles)
    header = np.zeros((22, sheet.shape[1], 3), np.uint8)
    cv2.putText(header, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, sheet]))


def validation_jobs(rgb_root):
    cfg = json.load(open(REPO / "stage7_sessions_config.json"))
    jobs = {}
    for key, e in cfg.items():
        if key.startswith("_"):
            continue
        label = e["session_label"]
        job = jobs.setdefault(label, dict(session=label, video=f"{rgb_root}/{e['rgb_video_relpath']}",
                                          offset=e["sync_result"]["offset_sec"],
                                          drift=e["sync_result"]["drift_slope"], lanes={}))
        m = re.search(r"RGB-confirmed (?:entry|touchdown) \(([\d.]+)s\)", e.get("entry_time_note", ""))
        job["lanes"][e["track"]] = dict(truth_rgb_entry_sec=float(m.group(1)) if m else None,
                                        config_entry_thermal_sec=e["entry_time_thermal_sec"])
    return list(jobs.values())


def batch_jobs(rgb_root):
    sync = pd.read_csv(SYNC_DIR / "session_sync.csv").set_index("session")
    jobs = []
    for session, s in sync.iterrows():
        lanes = {}
        video = None
        for lane, name in (("F", "Front"), ("B", "Back")):
            hj = REPO / "homography_calibration_inherited" / f"{session}_{name}_homography.json"
            if hj.exists():
                video = json.load(open(hj))["video_path"].split("/Process_Jason/", 1)[1]
                lanes[lane] = {}
        jobs.append(dict(session=session, video=f"{rgb_root}/{video}", offset=s.offset_sec,
                         drift=s.drift_slope, lanes=lanes))
    return jobs


def run_job(job, args_tuple):
    scan_sec, sample_hz, min_sustained_sec, changed_fraction, min_intrusion_sec, sheet_dir = args_tuple
    res = detect_session(job["video"], list(job["lanes"]), scan_sec, sample_hz, min_sustained_sec,
                         changed_fraction, min_intrusion_sec)
    rows = []
    for lane, extra in job["lanes"].items():
        r = res[lane]
        rgb = r["rgb_entry_sec"]
        thermal = None if rgb is None else round((rgb - job["offset"]) / (1 + job["drift"]), 2)
        contact_sheet(r["thumbs"], rgb, f"{job['session']} {lane}  RGB entry={rgb}s  thermal={thermal}s",
                      Path(sheet_dir) / f"{job['session']}_{lane}_entry.png")
        row = dict(session=job["session"], lane=lane, rgb_entry_sec=rgb, thermal_entry_sec=thermal,
                   first_intrusion_sec=r["first_intrusion_sec"], n_intrusion_samples=r["n_intrusion"],
                   entry_via_other_lane=r["entry_via_other_lane"],
                   sync_offset_sec=job["offset"], sync_drift=job["drift"], **extra)
        if extra.get("truth_rgb_entry_sec") is not None and rgb is not None:
            row["error_vs_truth_sec"] = round(rgb - extra["truth_rgb_entry_sec"], 2)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="RGB-side per-lane entry-time detection")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--rgb-video-root", default=DEFAULT_RGB_VIDEO_ROOT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scan-sec", type=float, default=600.0, help="Seconds of video to scan from the start")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--min-sustained-sec", type=float, default=2.0)
    parser.add_argument("--intrusion-changed-fraction", type=float, default=0.05)
    parser.add_argument("--min-intrusion-sec", type=float, default=1.5,
                        help="Intrusions shorter than this (shadows, exposure blips) are ignored")
    parser.add_argument("--jobs", type=int, default=1, help="Sessions decoded in parallel")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    jobs = validation_jobs(args.rgb_video_root) if args.validate else batch_jobs(args.rgb_video_root)
    params = (args.scan_sec, args.sample_hz, args.min_sustained_sec, args.intrusion_changed_fraction,
              args.min_intrusion_sec, str(sheet_dir))

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_job, j, params): j for j in jobs}
        for fut in futures:
            j = futures[fut]
            try:
                for r in fut.result():
                    rows.append(r)
                    print(f"{r['session']} {r['lane']}: rgb_entry={r['rgb_entry_sec']} "
                          f"thermal_entry={r['thermal_entry_sec']} first_intrusion={r['first_intrusion_sec']}"
                          + (f" truth_rgb={r.get('truth_rgb_entry_sec')} err={r.get('error_vs_truth_sec')}"
                             if args.validate else ""), flush=True)
            except Exception as e:
                print(f"{j['session']}: FAILED {e}", flush=True)
                rows.append(dict(session=j["session"], error=str(e)))
    out = out_dir / ("entry_times_validation.csv" if args.validate else "entry_times.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Done. {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
