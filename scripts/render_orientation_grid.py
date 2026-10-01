"""
Semi-manual RGB<->thermal orientation check (project_v8 scoping, round 2:
after three rounds of automated blob/motion detection ran into real,
non-obvious confounds -- see detect_orientation.py's module docstring and
project memory -- switched to this simpler approach).

Key realization: the double-rail channel housing (the physical arena
structure) is a STATIC landmark visible in nearly every frame, in both
modalities -- exactly the corners Adam clicked by hand in
thermalFeatures/Track_Alignment_*.pdf. No need to find "the right moment"
(entry, intrusion, peak motion) at all: any reasonably early, unobstructed
frame works, which sidesteps every confound the motion-based approach hit
(auto-exposure shifts, thermal gradient formation, peak-motion landing on
the symmetric release instant).

Renders a session's RGB/thermal frame pair with a coordinate grid (0.0-1.0
fraction on each axis) so the rail corners' relative positions can be read
off precisely by eye -- the flip classification itself stays a visual
judgment call (same as Adam's own method), not an automated decision.

Usage:
    python scripts/render_orientation_grid.py [SESSION_LABEL ...]   (default: Test_3/4/7)
    python scripts/render_orientation_grid.py --all [--limit N]
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.metadata import load_lut, filter_excluded
from scripts.detect_orientation import find_session_files, ALCOVA_PROCESS_JASON  # reuse, don't reimplement

OUT_DIR = f"{Path(__file__).resolve().parent.parent}/thermalFeatures/orientation_check_grid"

THERMAL_FRAME_IDX = 450   # ~45-56s at 8-10fps: past any startup transient, well before
                          # the gradient has fully developed, arena structure fully in frame
RGB_TIME_SEC = 45.0       # same rough target on the RGB side (sync offset is unknown for
                          # most of these 46 sessions, so this is a rough match, not exact --
                          # doesn't matter, we only need the STATIC rail structure, not a
                          # synchronized moment)


def grab_thermal_frame(seq_path, target_idx=THERMAL_FRAME_IDX):
    planck = read_planck_constants(seq_path)
    reader = SeqReader(seq_path)
    frame = None
    for idx, raw in reader.frames():
        if idx >= target_idx:
            frame = raw_to_celsius(raw, planck)
            break
    reader.close()
    return frame


def grab_rgb_frame(video_path, t=RGB_TIME_SEC):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(int(round(t * fps)), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _add_grid(ax, img, n=10):
    h, w = img.shape[:2]
    xticks = np.linspace(0, w, n + 1)
    yticks = np.linspace(0, h, n + 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / w:.1f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y / h:.1f}"))
    ax.grid(True, color="white", alpha=0.4, linewidth=0.6)
    ax.tick_params(colors="black", labelsize=7)


def render_session(name, seq_path, rgb_path):
    print(f"\n=== {name} ===", flush=True)
    thermal_frame = grab_thermal_frame(seq_path)
    rgb_frame = grab_rgb_frame(rgb_path)
    if thermal_frame is None or rgb_frame is None:
        print(f"  SKIP: thermal={'ok' if thermal_frame is not None else 'MISSING'} "
              f"rgb={'ok' if rgb_frame is not None else 'MISSING'}", flush=True)
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    axL.imshow(rgb_frame)
    _add_grid(axL, rgb_frame)
    axL.set_title(f"RGB\n{Path(rgb_path).name}", fontsize=9)

    im = axR.imshow(thermal_frame, cmap="jet")
    plt.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="°C")
    _add_grid(axR, thermal_frame)
    axR.set_title(f"Thermal (raw, uncropped)\n{Path(seq_path).name}", fontsize=9)

    fig.suptitle(f"{name}  (grid = fraction of width/height, 0.0-1.0)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = f"{OUT_DIR}/{name}_grid.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    KNOWN_STEMS = {
        "07-28-25_4540_B_4541_F_Test3-004": "Test_3",
        "07-30-25_4540_F_4541_B_Test4-008": "Test_4",
        "08-07-25_4541_F_4540_B_Test7-020": "Test_7",
    }

    lut = load_lut(str(Path(__file__).resolve().parent.parent / "metadata" / "LUT_CLEAN_July6.csv"))
    kept, _ = filter_excluded(lut)
    sessions = kept[["Video_name_SEQ", "Video_name"]].drop_duplicates().dropna()

    if args.all:
        targets = list(sessions.itertuples(index=False))
        if args.limit:
            targets = targets[: args.limit]
    elif args.sessions:
        targets = [row for row in sessions.itertuples(index=False) if row.Video_name_SEQ in args.sessions]
    else:
        targets = [row for row in sessions.itertuples(index=False) if row.Video_name_SEQ in KNOWN_STEMS]

    rows = []
    for row in targets:
        seq_path, mp4_path, n_seq, n_mp4 = find_session_files(row.Video_name_SEQ, row.Video_name)
        if seq_path is None or mp4_path is None:
            print(f"\n=== {row.Video_name_SEQ} ===\n  FILE NOT FOUND: seq_matches={n_seq} mp4_matches={n_mp4}", flush=True)
            rows.append(dict(session=row.Video_name_SEQ, status="file_not_found"))
            continue
        out = render_session(row.Video_name_SEQ, seq_path, mp4_path)
        rows.append(dict(session=row.Video_name_SEQ, status="ok" if out else "no_frame", image=out))

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/render_manifest.csv", index=False)
    print(f"\n=== DONE === {len(rows)} sessions, manifest -> {OUT_DIR}/render_manifest.csv")
