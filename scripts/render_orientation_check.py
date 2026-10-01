"""
RGB<->thermal orientation check (flip classification), project scoping for
"automate parts of the per-session calibration" (2026-09-29).

For a session with an already-known entry_time_thermal_sec + sync_result
(currently: Test_3/4/7, from stage7_real_run.py's SESSIONS dict), grabs the
thermal frame at entry time and the RGB frame at the corresponding
sync-mapped time, and renders them side by side -- same layout as Adam's own
manual reference files (thermalFeatures/Track_Alignment_*), so the flip
relationship (none / horizontal / vertical / both) can be read off by eye
the same way those were built, just without the manual frame-hunting.

This is the VALIDATION step against the 3 known-answer sessions before
building the (harder) version for the other 46 sessions, which don't have a
precomputed entry_time_thermal_sec/sync_result yet.

Usage:
    python scripts/render_orientation_check.py [SESSION ...]  (default: Test_3 Test_4 Test_7)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.landmarks.outputs import thermal_time_to_rgb_time
from stage7_real_run import SESSIONS, THERMAL_FPS

OUT_DIR = f"{Path(__file__).resolve().parent.parent}/thermalFeatures/orientation_check"

# project_v8: cfg["thermal_seq"] is Stage 7's LOCAL, ALREADY-CROPPED single-lane .seq
# (e.g. 60x416, one long thin strip -- confirmed 2026-09-29) -- the crop rectangle
# itself was chosen assuming the orientation we're trying to independently verify, so
# comparing it against the full, unsplit RGB frame would be circular. Use the RAW,
# uncropped thermal .seq on Alcova instead (348x464, both lanes, matching what Adam's
# own Track_Alignment_*.pdf reference images show) for a genuine from-scratch check.
ALCOVA_PROCESS_JASON = "/Volumes/cluster/alcova/bedfordlab/ThermalGradient/Process_Jason"


def find_raw_thermal_seq(session_label):
    matches = sorted(Path(ALCOVA_PROCESS_JASON).glob(f"*/{session_label}.seq"))
    if not matches:
        raise FileNotFoundError(f"No raw .seq found under {ALCOVA_PROCESS_JASON}/*/{session_label}.seq")
    if len(matches) > 1:
        print(f"  WARNING: multiple raw .seq matches for {session_label}, using first: {matches}", flush=True)
    return str(matches[0])


def grab_thermal_frame(thermal_seq_path, thermal_idx):
    planck = read_planck_constants(thermal_seq_path)
    reader = SeqReader(thermal_seq_path)
    frame = None
    for idx, raw in reader.frames():
        if idx == thermal_idx:
            frame = raw_to_celsius(raw, planck)
            break
        if idx > thermal_idx:
            break
    reader.close()
    return frame


def grab_rgb_frame(rgb_video_path, rgb_t):
    cap = cv2.VideoCapture(rgb_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(round(rgb_t * fps))
    frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_session(name, cfg):
    entry_t = cfg["entry_time_thermal_sec"]
    thermal_idx = int(round(entry_t * THERMAL_FPS))
    rgb_t = thermal_time_to_rgb_time(entry_t, cfg["sync_result"])
    raw_thermal_seq = find_raw_thermal_seq(cfg["session_label"])

    print(f"{name}: entry_time_thermal_sec={entry_t:.2f}s (thermal_idx={thermal_idx}), "
          f"rgb_t={rgb_t:.2f}s, raw_thermal_seq={raw_thermal_seq}", flush=True)

    thermal_frame = grab_thermal_frame(raw_thermal_seq, thermal_idx)
    rgb_frame = grab_rgb_frame(cfg["rgb_video"], rgb_t)

    if thermal_frame is None or rgb_frame is None:
        print(f"  SKIP: thermal_frame={'ok' if thermal_frame is not None else 'MISSING'} "
              f"rgb_frame={'ok' if rgb_frame is not None else 'MISSING'}", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    axL.imshow(rgb_frame)
    axL.set_title(f"RGB  t={rgb_t:.1f}s\n{Path(cfg['rgb_video']).name}", fontsize=9)
    axL.axis("off")

    im = axR.imshow(thermal_frame, cmap="jet")
    plt.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="°C")
    axR.set_title(f"Thermal (RAW, uncropped)  t={entry_t:.1f}s (idx={thermal_idx})\n{Path(raw_thermal_seq).name}", fontsize=9)
    axR.axis("off")

    fig.suptitle(f"{name} -- orientation check (compare landmark positions left/right, top/bottom)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = f"{OUT_DIR}/{name}_orientation_check.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  saved {fname}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or ["Test_3", "Test_4", "Test_7"]
    for name in names:
        render_session(name, SESSIONS[name])
    print("\n=== DONE ===")
