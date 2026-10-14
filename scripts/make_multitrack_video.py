"""Render a multi-track video around one stationary bout.

Layout (top to bottom): the lane's thermal video, velocity, mouse surface
temperature (tracking ROI mean) and floor temperature. Traces are drawn up to
a sweeping cursor, in the style of Shimatani et al. 2021 (PLOS ONE e0252447,
S1 Movie), but stacked vertically because the lanes are wide.

All signals share one clock: thermal seconds since .seq start
(frame_number / THERMAL_FPS), the same axis as the bouts_diagnostic QC plots.
Tracking signals are 1 Hz; they are linearly interpolated to the cursor.

Example:
    .venv/bin/python scripts/make_multitrack_video.py \\
        --video-file 08-20-25_4550_F_4539_B_Test9-002_Front.seq --bout-index 12 \\
        --results-dir /Volumes/alcova/.../retrack_2026-10/retrack \\
        --seq-root /Volumes/alcova/bedfordlab/ThermalGradient/croppedSeqFiles \\
        --out out.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import (  # noqa: E402
    BLOCK_ENTRY_SIZE, BLOCK_TABLE_START, FRAME_MAGIC, TAG_RAW_DATA,
    SeqReader, _parse_block_table, raw_to_celsius, read_planck_constants,
)

THERMAL_FPS = 10.0
STATIONARY_VEL_THRESH = 4.0  # px/s, same threshold line as the bouts_diagnostic QC plot

TRACKS = [  # (column, y label, color)
    ("velocity_smooth_px_s", "Velocity\n(px/s)", "#7b2d8e"),
    ("mouse_surface_temp_mean", "Mouse surface\ntemp (°C)", "#d6452f"),
    # Floor temp is read from the tracker's background model at the mouse ROI
    # footprint ("same_footprint"), not from a separate live floor patch.
    ("floor_temp_mean", "Floor temp at\nmouse pos. (°C)", "#2f62d6"),
]


def _frame_byte_size(fh, offset):
    fh.seek(offset)
    head = fh.read(BLOCK_TABLE_START + 4 * BLOCK_ENTRY_SIZE)
    raw = next((b for b in _parse_block_table(head) if b[0] == TAG_RAW_DATA), None)
    if head[:4] != FRAME_MAGIC or raw is None:
        raise ValueError(f"No FFF frame header at byte {offset}")
    return raw[1] + raw[2]


def frame_offset(index, frame0_size, frame_size):
    """Byte offset of frame `index`; frame 0 can differ in size from the rest."""
    return 0 if index == 0 else frame0_size + (index - 1) * frame_size


def read_frame_range(seq_path, start, stop):
    """Raw frames [start, stop) via one seek, so only the window crosses the network."""
    reader = SeqReader(seq_path)
    fh = reader._fh
    frame0 = _frame_byte_size(fh, 0)
    size = _frame_byte_size(fh, frame0)
    fh.seek(frame_offset(start, frame0, size))
    if fh.read(4) != FRAME_MAGIC:
        raise ValueError(f"Frame {start} is not at the expected offset; frames are not fixed-size")
    fh.seek(frame_offset(start, frame0, size))
    frames = []
    for _ in range(stop - start):
        result = reader._read_one_frame()
        if result is None:
            break
        frames.append(result[1].copy())
    reader.close()
    if len(frames) != stop - start:
        raise IOError(f"Read {len(frames)}/{stop - start} frames from {seq_path} (truncated or mount dropped?)")
    return frames


def trace_up_to(t, y, t_now):
    """Points with t <= t_now plus one interpolated point at t_now (keeps NaN gaps)."""
    keep = t <= t_now
    tt, yy = t[keep], y[keep]
    if len(tt) and t_now > tt[-1] and t_now <= t[-1]:
        tt = np.append(tt, t_now)
        yy = np.append(yy, np.interp(t_now, t, y))
    return tt, yy


def load_inputs(results_dir, video_file, bout_index, pad_sec):
    bouts = pd.read_csv(Path(results_dir) / "bout_table.csv")
    bout = bouts[(bouts.video_file == video_file) & (bouts.bout_index == bout_index)]
    if bout.empty:
        raise SystemExit(f"No bout {bout_index} for {video_file} in bout_table.csv")
    b0, b1 = float(bout.bout_start_sec.iloc[0]), float(bout.bout_end_sec.iloc[0])
    t0, t1 = b0 - pad_sec, b1 + pad_sec
    cols = ["video_file", "elapsed_time_sec", "mouse_roi_center_x", "mouse_roi_center_y",
            "mouse_roi_radius_px", "mouse_id", "virus", "injection"] + [c for c, _, _ in TRACKS]
    chunks = pd.read_csv(Path(results_dir) / "master_tracking_with_metadata.csv",
                         usecols=cols, chunksize=200_000)
    trk = pd.concat(c[c.video_file == video_file] for c in chunks)
    trk = trk[(trk.elapsed_time_sec >= t0 - 1) & (trk.elapsed_time_sec <= t1 + 1)]
    return trk.sort_values("elapsed_time_sec").reset_index(drop=True), (b0, b1), (t0, t1)


def roi_at(trk, prefix, t_now):
    """ROI center/radius at the nearest 1 Hz sample, or None if invalid there."""
    row = trk.iloc[(trk.elapsed_time_sec - t_now).abs().argmin()]
    x, y, r = row[f"{prefix}_center_x"], row[f"{prefix}_center_y"], row[f"{prefix}_radius_px"]
    return None if np.isnan(x) or np.isnan(y) else (x, y, r)


def build_figure(frame_c, trk, bout, window, title, cmap, vlim):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = fig.add_gridspec(len(TRACKS) + 1, 1, height_ratios=[1.1] + [1] * len(TRACKS),
                          left=0.08, right=0.97, top=0.93, bottom=0.07, hspace=0.25)
    ax_img = fig.add_subplot(gs[0])
    im = ax_img.imshow(frame_c, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       aspect="equal", interpolation="nearest")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    cbar = fig.colorbar(im, ax=ax_img, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("°C")
    roi_circle = Circle((0, 0), 1, fill=False, ec="cyan", lw=1.5)
    ax_img.add_patch(roi_circle)
    fig.suptitle(title, fontsize=16)
    clock = fig.text(0.97, 0.955, "", ha="right", fontsize=14, family="monospace")

    t = trk.elapsed_time_sec.to_numpy()
    lines, cursors, axes = [], [], []
    for i, (col, label, color) in enumerate(TRACKS):
        ax = fig.add_subplot(gs[i + 1], sharex=axes[0] if axes else None)
        y = trk[col].to_numpy()
        ax.axvspan(*bout, color="#b9cfe3", alpha=0.6, lw=0,
                   label="stationary bout" if i == 0 else None)
        ax.plot(t, y, color=color, alpha=0.12, lw=1)  # faint full trace for orientation
        (line,) = ax.plot([], [], color=color, lw=2)
        if col == "velocity_smooth_px_s":
            ax.axhline(STATIONARY_VEL_THRESH, color="red", ls="--", lw=1,
                       label=f"thresh={STATIONARY_VEL_THRESH:g} px/s")
            ax.legend(loc="upper right", fontsize=10)
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = 0.08 * (hi - lo or 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(label, fontsize=12)
        if i < len(TRACKS) - 1:
            ax.tick_params(labelbottom=False)
        cursors.append(ax.axvline(window[0], color="red", lw=1.5))
        lines.append((line, y))
        axes.append(ax)
    axes[0].set_xlim(*window)
    axes[-1].set_xlabel("Time since thermal recording start (s)", fontsize=12)
    return fig, im, roi_circle, clock, lines, cursors


def render(args):
    trk, bout, window = load_inputs(args.results_dir, args.video_file, args.bout_index, args.pad_sec)
    seq_path = Path(args.seq_root) / args.seq_subdir / args.video_file
    if not seq_path.exists():
        matches = list(Path(args.seq_root).glob(f"*/{args.video_file}"))
        if len(matches) != 1:
            raise SystemExit(f"Could not find a unique {args.video_file} under {args.seq_root}")
        seq_path = matches[0]
    f_start, f_stop = int(round(window[0] * THERMAL_FPS)), int(round(window[1] * THERMAL_FPS))
    print(f"bout {bout[0]:.0f}-{bout[1]:.0f} s; window {window[0]:.0f}-{window[1]:.0f} s; "
          f"frames {f_start}-{f_stop} from {seq_path}", flush=True)
    planck = read_planck_constants(str(seq_path))
    if planck is None:
        raise SystemExit("No Planck constants (exiftool missing?); refusing the linear fallback")
    frames = [raw_to_celsius(f, planck) for f in read_frame_range(str(seq_path), f_start, f_stop)]
    vlim = args.vlim or tuple(np.percentile(np.stack(frames), [1, 99.8]).round(1))

    meta = trk.iloc[0]
    session = args.video_file.replace(".seq", "")
    title = (f"{session}  |  mouse {meta.mouse_id}, {meta.virus}, {meta.injection}  |  "
             f"stationary bout {args.bout_index} ({bout[0]:.0f}-{bout[1]:.0f} s)  |  {args.speed:g}x speed")
    fig, im, roi_circle, clock, lines, cursors = build_figure(
        frames[0], trk, bout, window, title, args.cmap, vlim)

    out_fps = THERMAL_FPS * args.speed
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.canvas.get_width_height()
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), out_fps, (w, h))
        if writer.isOpened():
            print(f"encoding {fourcc} at {out_fps:g} fps, {w}x{h}", flush=True)
            break
    if not writer.isOpened():
        raise SystemExit("cv2.VideoWriter could not open an mp4 encoder")

    t_trk = trk.elapsed_time_sec.to_numpy()
    for i, frame in enumerate(frames):
        t_now = (f_start + i) / THERMAL_FPS
        im.set_data(frame)
        roi = roi_at(trk, "mouse_roi", t_now)
        roi_circle.set_visible(roi is not None)
        if roi is not None:
            roi_circle.center, roi_circle.radius = roi[:2], roi[2]
        for (line, y), cursor in zip(lines, cursors):
            line.set_data(*trace_up_to(t_trk, y, t_now))
            cursor.set_xdata([t_now, t_now])
        clock.set_text(f"t = {t_now:7.1f} s")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
    writer.release()
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / out_fps:.1f} s)", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-file", required=True, help="video_file value in the master CSV, e.g. ..._Front.seq")
    p.add_argument("--bout-index", type=int, required=True, help="0-based bout_index in bout_table.csv")
    p.add_argument("--results-dir", required=True,
                   help="dir with master_tracking_with_metadata.csv and bout_table.csv")
    p.add_argument("--seq-root", required=True, help="croppedSeqFiles root (date subdirs inside)")
    p.add_argument("--seq-subdir", default="", help="date subdir; searched for if omitted")
    p.add_argument("--pad-sec", type=float, default=30.0)
    p.add_argument("--speed", type=float, default=5.0, help="playback speed; output fps = 10 * speed")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--vlim", type=float, nargs=2, help="fixed thermal color limits (°C); default 1-99.8th pct")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
