"""
Interactively label one validation-set frame (project_brief_v7.md §8):
mask outline, tail-base point, tail centerline, interscapular point, head
point, nose-tip point, and a human-drawn warm-spot ROI.

Two panels: RGB (left) for the mask/tail/interscapular/head/nose labels,
thermal (right) for the warm-spot ROI (a thermal judgment, not an RGB one).
Switch label mode with a key, click to add points for that mode, ENTER to
finish and save. Frames are specified by an already-selected row from
select_validation_frames() (session/track/frame) — this script labels one
row at a time, run once per selected frame.

nose_point vs. head_point (added 2026-08-25): these are deliberately TWO
SEPARATE landmarks, not one. src/landmarks/rgb_landmarks.py's own
"nose_point" (the classical-CV skeleton's anterior-most endpoint) is
specifically the literal tip of the animal's silhouette — that's what
nose_point here should match, for a real, well-defined bake-off
comparison (see src/landmarks/bakeoff.py). head_point is a looser,
brief-§1-secondary-landmark click ("head" generally, e.g. center of the
head mass) and is NOT the same thing — a real comparison run against the
project's first 3 labels (which only had head_point) found the
classical algorithm's nose point consistently sitting right at the
silhouette tip while the human head_point click landed further back
toward the head's center, on frames where the head is a rounded, not
sharply-pointed, shape. Label BOTH going forward; don't conflate them.

Usage:
    python scripts/label_validation_frame.py \\
        --seq "/path/to/cropped_..._Front.seq" \\
        --video "/path/to/webcam.mp4" \\
        --lane F \\
        --rgb-frame-time 62 \\
        --thermal-frame-time 62 \\
        --output-dir validation_labels

Modes (press the key to switch, click to add points in that mode):
    m  mask outline polygon (RGB)         t  tail centerline (RGB, base->tip)
    i  interscapular point (RGB)          h  head point (RGB, general)
    n  nose-tip point (RGB, literal tip)  w  warm-spot ROI polygon (thermal)
    z  undo last point in current mode    ENTER  finish and save
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader, adu_to_display
from src.landmarks.webcam_preprocessing import detect_track_split_row, split_track_crops, LANE_TOP

MODE_KEYS = {"m": "mask", "t": "tail", "i": "interscapular", "h": "head", "n": "nose", "w": "warm_spot_roi"}
POINT_MODES = {"interscapular", "head", "nose"}  # single point, not a polygon/path
MODE_COLORS = {"mask": "lime", "tail": "yellow", "interscapular": "cyan", "head": "magenta", "nose": "orange", "warm_spot_roi": "red"}


def get_rgb_frame(video_path: str, lane: str, time_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Could not read first frame of {video_path}")
    split_row = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))

    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame at t={time_sec}s from {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top, bottom = split_track_crops(gray, split_row=split_row)
    return top if lane == LANE_TOP else bottom


def get_thermal_frame(seq_path: str, time_sec: float, fps: float = 8.0) -> np.ndarray:
    target_idx = int(time_sec * fps)
    with SeqReader(seq_path) as reader:
        for idx, px in reader.frames():
            if idx == target_idx:
                return adu_to_display(px)
            if idx > target_idx:
                break
    raise ValueError(f"Could not find frame index {target_idx} in {seq_path}")


def label_frame_interactive(rgb_img: np.ndarray, thermal_img: np.ndarray) -> dict:
    fig, (ax_rgb, ax_thermal) = plt.subplots(1, 2, figsize=(16, 6))
    ax_rgb.imshow(rgb_img, cmap="gray", aspect="equal")
    thermal_cmap = plt.get_cmap("inferno")
    thermal_rgb = (thermal_cmap(thermal_img / 255.0)[:, :, :3] * 255).astype(np.uint8)
    ax_thermal.imshow(thermal_rgb, aspect="equal")

    state = {"mode": "mask"}
    labels: dict = {"mask": [], "tail": [], "interscapular": None, "head": None, "nose": None, "warm_spot_roi": []}
    artists: list = []

    def mode_axis(mode):
        return ax_thermal if mode == "warm_spot_roi" else ax_rgb

    def title():
        return (
            f"Mode: {state['mode'].upper()}  (m=mask t=tail i=interscapular h=head n=nose w=warm-spot-ROI)\n"
            "Click to add point(s) for this mode  |  Z undo  |  ENTER finish+save"
        )

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        fig.suptitle(title(), fontsize=10)

        if labels["mask"]:
            xs, ys = zip(*labels["mask"])
            artists.append(ax_rgb.scatter(xs, ys, c=MODE_COLORS["mask"], s=20, zorder=5))
            (line,) = ax_rgb.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color=MODE_COLORS["mask"], lw=1)
            artists.append(line)
        if labels["tail"]:
            xs, ys = zip(*labels["tail"])
            artists.append(ax_rgb.scatter(xs, ys, c=MODE_COLORS["tail"], s=20, zorder=5))
            (line,) = ax_rgb.plot(xs, ys, color=MODE_COLORS["tail"], lw=1.5)
            artists.append(line)
        for key in ("interscapular", "head", "nose"):
            if labels[key] is not None:
                x, y = labels[key]
                artists.append(ax_rgb.scatter([x], [y], c=MODE_COLORS[key], s=60, marker="x", zorder=6))
        if labels["warm_spot_roi"]:
            xs, ys = zip(*labels["warm_spot_roi"])
            artists.append(ax_thermal.scatter(xs, ys, c=MODE_COLORS["warm_spot_roi"], s=20, zorder=5))
            (line,) = ax_thermal.plot(
                list(xs) + [xs[0]], list(ys) + [ys[0]], color=MODE_COLORS["warm_spot_roi"], lw=1
            )
            artists.append(line)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.button != 1:
            return
        mode = state["mode"]
        if event.inaxes != mode_axis(mode):
            return
        pt = [int(event.xdata), int(event.ydata)]
        if mode in POINT_MODES:
            labels[mode] = pt
        else:
            labels[mode].append(pt)
        redraw()

    def on_key(event):
        if event.key in MODE_KEYS:
            state["mode"] = MODE_KEYS[event.key]
            redraw()
        elif event.key == "enter":
            plt.close(fig)
        elif event.key in ("z", "Z"):
            mode = state["mode"]
            if mode in POINT_MODES:
                labels[mode] = None
            elif labels[mode]:
                labels[mode].pop()
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    plt.show()

    return labels


def main():
    parser = argparse.ArgumentParser(description="Interactively label one validation-set frame")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--lane", required=True, choices=["B", "F"])
    parser.add_argument("--rgb-frame-time", type=float, required=True, help="RGB clock time (sec)")
    parser.add_argument("--thermal-frame-time", type=float, required=True, help="Thermal clock time (sec)")
    parser.add_argument(
        "--posture",
        default=None,
        help=(
            "Free-text posture note (e.g. curled, rearing, grooming, extended) — brief §8 wants "
            "posture diversity, but there's no automated classifier to derive this from, so it's "
            "a human judgment call at labeling time."
        ),
    )
    parser.add_argument("--output-dir", default="validation_labels")
    args = parser.parse_args()

    seq_path = Path(args.seq)
    video_path = Path(args.video)
    for p in (seq_path, video_path):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Reading RGB frame at t={args.rgb_frame_time}s (lane {args.lane})…")
    rgb_img = get_rgb_frame(str(video_path), args.lane, args.rgb_frame_time)
    print(f"Reading thermal frame at t={args.thermal_frame_time}s…")
    thermal_img = get_thermal_frame(str(seq_path), args.thermal_frame_time)

    print()
    print("Label window opening. Modes: m=mask t=tail i=interscapular h=head n=nose w=warm-spot-ROI")
    print("Click to add points for the active mode. Z undoes. ENTER finishes and saves.")

    labels = label_frame_interactive(rgb_img, thermal_img)

    if len(labels["mask"]) < 3:
        print("WARNING: mask polygon has fewer than 3 points — not a valid outline.", file=sys.stderr)
    if len(labels["tail"]) < 2:
        print("WARNING: tail centerline has fewer than 2 points.", file=sys.stderr)
    if labels["interscapular"] is None:
        print("WARNING: interscapular point not set.", file=sys.stderr)
    if labels["head"] is None:
        print("WARNING: head point not set.", file=sys.stderr)
    if labels["nose"] is None:
        print("WARNING: nose-tip point not set.", file=sys.stderr)
    if len(labels["warm_spot_roi"]) < 3:
        print("WARNING: warm-spot ROI has fewer than 3 points — not a valid outline.", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{seq_path.stem}_rgb{args.rgb_frame_time:g}_thermal{args.thermal_frame_time:g}_label.json"
    payload = {
        "seq_path": str(seq_path),
        "video_path": str(video_path),
        "lane": args.lane,
        "rgb_frame_time_sec": args.rgb_frame_time,
        "thermal_frame_time_sec": args.thermal_frame_time,
        "posture": args.posture,
        "tail_base_point": labels["tail"][0] if labels["tail"] else None,
        "mask_polygon": labels["mask"],
        "tail_centerline": labels["tail"],
        "interscapular_point": labels["interscapular"],
        "head_point": labels["head"],
        "nose_point": labels["nose"],
        "warm_spot_roi_polygon_thermal": labels["warm_spot_roi"],
    }
    out_path = out_dir / out_name
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
