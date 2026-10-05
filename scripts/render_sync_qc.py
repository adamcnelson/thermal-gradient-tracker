"""
Visual QC for trajectory-alignment sync estimates (estimate_sync_trajectory.py +
consolidate_sync_estimates.py). One PNG per session:

  1. per lane: thermal-native mouse x(t) vs the RGB track warped into thermal
     space and shifted by the session's consolidated sync;
  2. loss vs offset (at the session's fps/drift) for the frame-overlay lane,
     marking the estimate and the best competing offset >15s away;
  3. six real thermal frames at moments the mouse is moving, with the RGB mouse
     silhouette warped on top at the estimated sync (green) and at the
     competing offset (red). Right sync => green sits on the thermal mouse.

Panel 3 is the one to trust: numeric fit metrics have misled this project
before, while a wrong offset on a moving mouse is obvious by eye.

Usage:
    python scripts/render_sync_qc.py --sessions 08-20-25_4550_F_4539_B_Test9-002 ...
    python scripts/render_sync_qc.py --review   # every session with low_confidence or review_reasons
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.landmarks.rgb_landmarks import RgbBackgroundModel, segment_mouse_rgb
from src.landmarks.sync import _trajectory_loss
from src.landmarks.video_io import SequentialFrameReader
from src.landmarks.webcam_preprocessing import LANE_TOP, detect_track_split_row, split_track_crops
from src.seq_io import SeqReader, raw_to_celsius, read_planck_constants

REPO = Path(__file__).resolve().parent.parent
SYNC_DIR = REPO / "thermalFeatures" / "trajectory_sync"
ALCOVA_TG = "/Volumes/alcova/bedfordlab/ThermalGradient"
DEFAULT_TRACKING_DIR = f"{ALCOVA_TG}/SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/trackingOutputs"
LANE_NAMES = {"F": "Front", "B": "Back"}
N_MOMENTS = 6
MOVING_PX = 15.0  # thermal x change between consecutive 1s samples that counts as "moving"


def alcova_path(recorded: str, marker: str, root: str) -> str:
    """Re-root a path recorded on an old mount onto `root`, keeping the part after `marker`."""
    return f"{root}/{recorded.split(marker, 1)[1]}"


def lane_inputs(session, lane, tracking_dir):
    hj = REPO / "homography_calibration_inherited" / f"{session}_{LANE_NAMES[lane]}_homography.json"
    meta = json.load(open(hj))
    return dict(
        H=np.array(meta["H"], dtype=np.float64),
        seq=alcova_path(meta["seq_path"], "/croppedSeqFiles/", f"{ALCOVA_TG}/croppedSeqFiles"),
        video=alcova_path(meta["video_path"], "/Process_Jason/", f"{ALCOVA_TG}/Process_Jason"),
        tracking=f"{tracking_dir}/{session}_{LANE_NAMES[lane]}_tracking_every10frames.csv",
        rgb_track=SYNC_DIR / "rgb_tracks" / f"{session}_{lane}_rgb_track.csv",
    )


def load_tracks(inp):
    d = pd.read_csv(inp["tracking"])
    d = d[(d.tracking_confidence == 1.0) & d.mouse_centroid_x.notna()]
    r = pd.read_csv(inp["rgb_track"]).dropna(subset=["rgb_centroid_x"]).sort_values("rgb_time_sec")
    pts = r[["rgb_centroid_x", "rgb_centroid_y"]].to_numpy(np.float64).reshape(-1, 1, 2)
    rx = cv2.perspectiveTransform(pts, inp["H"])[:, 0, 0]
    return d.frame_number.to_numpy(float), d.mouse_centroid_x.to_numpy(float), r.rgb_time_sec.to_numpy(float), rx


def competing_offset(offsets, curve, best, min_sep=15.0, basin_halfwidth=5.0):
    """Best separate local minimum >min_sep from `best` (not just the shoulder of best's own basin)."""
    k = max(1, int(round(basin_halfwidth / (offsets[1] - offsets[0]))))
    c = np.where(np.isnan(curve), np.inf, curve)
    is_min = np.array([c[i] == c[max(0, i - k):i + k + 1].min() for i in range(len(c))])
    cand = is_min & (np.abs(offsets - best) > min_sep) & np.isfinite(c)
    if not cand.any():
        cand = (np.abs(offsets - best) > min_sep) & np.isfinite(c)
    return float(offsets[cand][np.argmin(c[cand])])


def x_bias_px(frames, tx, rt, rx, fps, drift, offset):
    """Median (RGB-warped x - thermal x) at the sync: a constant homography offset, not timing."""
    q = frames / fps * (1 + drift) + offset
    ok = (q >= rt[0]) & (q <= rt[-1])
    return float(np.median(np.interp(q[ok], rt, rx) - tx[ok]))


def pick_moments(frames, tx, fps, n=N_MOMENTS):
    """Most-moving thermal sample in each of n equal chunks of the session."""
    moving = np.abs(np.diff(tx, prepend=tx[0])) > MOVING_PX
    picks = []
    for chunk in np.array_split(np.arange(len(frames)), n):
        cand = chunk[moving[chunk]]
        if len(cand):
            picks.append(int(cand[np.argmax(np.abs(np.diff(tx, prepend=tx[0]))[cand])]))
    return [int(frames[i]) for i in picks]


def read_thermal_frames(seq_path, wanted):
    planck = read_planck_constants(seq_path)
    out, reader = {}, SeqReader(seq_path)
    try:
        for idx, raw in reader.frames():
            if idx in wanted:
                out[idx] = raw_to_celsius(raw, planck)
                if len(out) == len(wanted):
                    break
    finally:
        reader.close()
    return out


def read_rgb_masks(video, lane, frame_idxs):
    """{rgb frame index: lane-crop mouse mask or None}, frames reached by sequential decode."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    split = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
    bg = []
    for i in np.linspace(0, n - 1, 30, dtype=int):  # median background: seeking is fine here
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            top, bot = split_track_crops(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), split_row=split)
            bg.append(top if lane == LANE_TOP else bot)
    cap.release()
    model = RgbBackgroundModel.build(bg)
    reader, masks = SequentialFrameReader(video), {}
    try:
        for idx in sorted(set(frame_idxs)):
            ok, f = reader.read(idx)
            if not ok:
                masks[idx] = None
                continue
            top, bot = split_track_crops(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), split_row=split)
            masks[idx] = segment_mouse_rgb(top if lane == LANE_TOP else bot, model, min_area=200, max_area=20000)
    finally:
        reader.release()
    return masks, reader.fps


def draw_mask_outline(ax, mask, H, shape, color, ls="-"):
    if mask is None:
        return
    warped = cv2.warpPerspective(mask.astype(np.uint8), H, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = np.vstack([c[:, 0, :], c[:1, 0, :]])
        ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, lw=1.4)


def render_session(session, sess_row, lane_rows, tracking_dir, out_dir):
    offset, drift, fps = sess_row.offset_sec, sess_row.drift_slope, sess_row.camera_fps
    lanes = list(lane_rows.lane)
    clean = [l for l, f in zip(lane_rows.lane, lane_rows["flags"].fillna("")) if not f]
    qc_lane = clean[0] if clean else lanes[0]
    inputs = {l: lane_inputs(session, l, tracking_dir) for l in lanes}
    tracks = {l: load_tracks(inputs[l]) for l in lanes}

    frames, tx, rt, rx = tracks[qc_lane]
    offsets = np.arange(-300, 300, 0.5)
    curve, _ = _trajectory_loss(frames, tx, rt, rx, fps, drift, offsets, 15.0, 3.0, 200)
    runner = competing_offset(offsets, curve, offset)

    moments = pick_moments(frames, tx, fps)
    to_rgb_t = lambda f, off: f / fps * (1 + drift) + off
    thermal = read_thermal_frames(inputs[qc_lane]["seq"], set(moments))
    rgb_fps = 60.0
    want = {m: (int(round(to_rgb_t(m, offset) * rgb_fps)), int(round(to_rgb_t(m, runner) * rgb_fps))) for m in moments}
    masks, rgb_fps_actual = read_rgb_masks(inputs[qc_lane]["video"], qc_lane, [i for p in want.values() for i in p])
    assert abs(rgb_fps_actual - rgb_fps) < 0.01, f"unexpected RGB fps {rgb_fps_actual}"

    n_rows = len(lanes) + 1 + len(moments)
    fig = plt.figure(figsize=(14, 2.2 * (len(lanes) + 1) + 1.0 * len(moments)))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[2.2] * (len(lanes) + 1) + [1.0] * len(moments))
    for k, l in enumerate(lanes):
        f, x, t, r = tracks[l]
        ax = fig.add_subplot(gs[k])
        ax.plot((t - offset) / (1 + drift), r, ".", ms=1.5, color="tab:orange", label="RGB (warped, shifted by sync)")
        ax.plot(f / fps, x, ".", ms=1.5, color="tab:blue", label="thermal-native")
        for m in moments if l == qc_lane else []:
            ax.axvline(m / fps, color="green", lw=0.6, alpha=0.6)
        ax.set_ylabel(f"{LANE_NAMES[l]} x (px)")
        ax.set_title(f"{LANE_NAMES[l]}: RGB-thermal x bias at sync = "
                     f"{x_bias_px(f, x, t, r, fps, drift, offset):+.1f}px", fontsize=8, loc="left", pad=2)
        ax.set_xlim(0, f.max() / fps)
        if k == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=6)
    ax = fig.add_subplot(gs[len(lanes)])
    ax.plot(offsets, curve, color="k", lw=1)
    ax.axvline(offset, color="green", label=f"estimate {offset:+.2f}s")
    ax.axvline(runner, color="red", ls="--", label=f"best competitor {runner:+.1f}s")
    ax.set_xlabel("offset (s)")
    ax.set_ylabel(f"loss ({LANE_NAMES[qc_lane]})")
    ax.legend(loc="upper right", fontsize=7)
    for k, m in enumerate(moments):
        ax = fig.add_subplot(gs[len(lanes) + 1 + k])
        img = thermal.get(m)
        if img is not None:
            ax.imshow(img, cmap="inferno", vmin=np.percentile(img, 2), vmax=np.percentile(img, 99.5),
                      aspect="equal", interpolation="nearest")
            draw_mask_outline(ax, masks.get(want[m][1]), inputs[qc_lane]["H"], img.shape, "red", "--")
            draw_mask_outline(ax, masks.get(want[m][0]), inputs[qc_lane]["H"], img.shape, "lime")
        ax.set_title(f"t_thermal={m / fps:.0f}s  (green = estimate, red dashed = {runner:+.1f}s)",
                     fontsize=7, loc="left", pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    flags = "; ".join(x for x in (sess_row.low_confidence_reasons, sess_row.review_reasons) if isinstance(x, str) and x)
    fig.suptitle(f"{session}  sync offset={offset:+.2f}s drift={drift:+.4f} fps={fps:.0f}  "
                 f"[{flags or 'no flags'}]  frames: {LANE_NAMES[qc_lane]} lane", fontsize=10)
    fig.tight_layout()
    out = out_dir / f"{session}_sync_qc.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Render sync QC figures")
    parser.add_argument("--sessions", nargs="*", default=[])
    parser.add_argument("--review", action="store_true", help="All low-confidence / needs-review sessions")
    parser.add_argument("--tracking-dir", default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--output-dir", default=str(SYNC_DIR / "qc"))
    args = parser.parse_args()

    sess = pd.read_csv(SYNC_DIR / "session_sync.csv").set_index("session")
    lanes = pd.read_csv(SYNC_DIR / "sync_estimates.csv")
    names = list(args.sessions)
    if args.review:
        names += [s for s, r in sess.iterrows()
                  if r.low_confidence or (isinstance(r.review_reasons, str) and r.review_reasons)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in dict.fromkeys(names):
        out = render_session(s, sess.loc[s], lanes[lanes.session == s].sort_values("lane", ascending=False),
                             args.tracking_dir, out_dir)
        print(f"{s} -> {out}", flush=True)


if __name__ == "__main__":
    main()
