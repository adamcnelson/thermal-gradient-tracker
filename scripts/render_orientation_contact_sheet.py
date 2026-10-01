"""
Semi-manual RGB<->thermal orientation check, round 3: contact sheet.

Round 1 (detect_orientation.py) tried to automatically pick "the" intrusion
frame via motion detection -- ran into real confounds (auto-exposure shifts,
thermal gradient formation, peak-motion landing on the symmetric release
instant) across three iterations. Round 2 (render_orientation_grid.py) tried
reading the static rail structure alone -- too left/right symmetric to
independently disambiguate without an explicit marked point.

This sidesteps both: render several candidate timestamps per modality as a
contact sheet (thumbnails), and let a human (or Claude, visually) pick out
whichever pair clearly shows the handler's arm/hand -- the same judgment
call that worked cleanly for Test_3/4/7's very first (known-entry-time)
renders. No detection algorithm, no "peak" to get wrong -- just a faster way
to find a good frame than scrubbing the full video by hand.

Usage:
    python scripts/render_orientation_contact_sheet.py [SESSION_LABEL ...]  (default: Test_3/4/7)
    python scripts/render_orientation_contact_sheet.py --all [--limit N]
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
import numpy as np
import pandas as pd

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.metadata import load_lut, filter_excluded
from scripts.detect_orientation import find_session_files

OUT_DIR = f"{Path(__file__).resolve().parent.parent}/thermalFeatures/orientation_contact_sheets"

# Spread across ~10-100s: known entry times for Test_3/4/7 ranged 24-95s, so
# this window should comfortably bracket the entry moment for most sessions.
THERMAL_CANDIDATE_IDXS = [80, 160, 240, 320, 400, 480, 560, 650, 750, 850, 950, 1050]  # ~8-130s at 8-10fps
# 2026-09-29: 8 candidates at ~15s spacing missed the real Test_4 entry moment
# (82.8s fell between the 75s and 90s samples, and the arm-visible window is
# apparently only a few real seconds wide) -- denser spacing costs almost
# nothing (RGB seeks are the only real per-candidate cost, ~0.28s each).
RGB_CANDIDATE_SEC = [5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110]


def grab_thermal_candidates(seq_path, target_idxs):
    planck = read_planck_constants(seq_path)
    reader = SeqReader(seq_path)
    targets = sorted(set(target_idxs))
    found = {}
    for idx, raw in reader.frames():
        while targets and idx >= targets[0]:
            t = targets.pop(0)
            found[t] = raw_to_celsius(raw, planck)
        if not targets:
            break
    reader.close()
    return found  # {requested_idx: frame} -- frame is the first one seen at/after that idx


def grab_rgb_candidates(video_path, target_secs):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = {}
    for t in target_secs:
        frame_idx = max(0, min(int(round(t * fps)), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            found[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return found


def render_contact_sheet(name, seq_path, rgb_path):
    print(f"\n=== {name} ===", flush=True)
    thermal_frames = grab_thermal_candidates(seq_path, THERMAL_CANDIDATE_IDXS)
    rgb_frames = grab_rgb_candidates(rgb_path, RGB_CANDIDATE_SEC)
    print(f"  thermal: {len(thermal_frames)}/{len(THERMAL_CANDIDATE_IDXS)} candidates read; "
          f"rgb: {len(rgb_frames)}/{len(RGB_CANDIDATE_SEC)}", flush=True)

    n = max(len(THERMAL_CANDIDATE_IDXS), len(RGB_CANDIDATE_SEC))
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.2))
    for ax in axes.flat:
        ax.axis("off")

    for col, idx in enumerate(THERMAL_CANDIDATE_IDXS):
        ax = axes[0, col]
        if idx in thermal_frames:
            ax.imshow(thermal_frames[idx], cmap="jet")
        ax.set_title(f"idx={idx}", fontsize=8)
        ax.axis("off")

    for col, t in enumerate(RGB_CANDIDATE_SEC):
        ax = axes[1, col]
        if t in rgb_frames:
            ax.imshow(rgb_frames[t])
        ax.set_title(f"t={t}s", fontsize=8)
        ax.axis("off")

    axes[0, 0].set_ylabel("Thermal", fontsize=9)
    axes[1, 0].set_ylabel("RGB", fontsize=9)
    fig.suptitle(f"{name} -- contact sheet (pick the clearest arm/hand frame from each row)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = f"{OUT_DIR}/{name}_contact_sheet.png"
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
        "07-28-25_4540_B_4541_F_Test3-004",
        "07-30-25_4540_F_4541_B_Test4-008",
        "08-07-25_4541_F_4540_B_Test7-020",
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
        try:
            out = render_contact_sheet(row.Video_name_SEQ, seq_path, mp4_path)
            rows.append(dict(session=row.Video_name_SEQ, status="ok", image=out))
        except Exception as exc:
            # A corrupt/truncated/oversized source file (real, confirmed cases exist
            # in this corpus -- 2026-09-29) must not abort the whole batch.
            print(f"  ERROR reading {row.Video_name_SEQ}: {exc}", flush=True)
            rows.append(dict(session=row.Video_name_SEQ, status="error", image=str(exc)))

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    print(f"\n=== DONE === {len(rows)} sessions, manifest -> {OUT_DIR}/manifest.csv")
