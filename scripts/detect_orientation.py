"""
Automated RGB<->thermal orientation detection (project_v8 scoping,
"automate parts of the per-session calibration", 2026-09-29).

For any session in the metadata LUT (not just the 3 already Stage7-
calibrated ones), locates the raw thermal .seq and RGB .mp4 on Alcova,
automatically finds a good "intrusion" frame (largest hand/mouse-scale
blob relative to a quick per-session background) in each modality within
an early time window, and reports the horizontal/vertical flip by
comparing which half of the frame that blob sits in -- extending the same
"mouse in hand" read that independently reproduced Test_4's already-known
horizontal-flip finding (see render_orientation_check.py's validation
pass: mouse-in-hand at the LEFT in RGB, RIGHT in thermal).

Deliberately does NOT reuse src/landmarks/entry_detection.py's
classify_frame_state_*()/find_entry_frame_index_*() -- those need a full
BackgroundModel built by sampling frames spread across the WHOLE video
(mouse_segmentation.BackgroundModel.build() calls reader.count_frames()
first), which for a ~2000s thermal .seq at ~38ms/frame (confirmed
2026-09-29 timing over the Alcova mount) means a ~12-minute read PER
SESSION just to count frames -- not viable across 49 sessions. This uses a
much cheaper single-pass, bounded-window approach instead: background =
the first few frames only (session start is reliably empty, per
entry_detection.py's own docstring), scanned forward only ~90-120s of
real time.

Usage:
    python scripts/detect_orientation.py [SESSION_LABEL ...]   (default: Test_3/4/7, known-answer validation)
    python scripts/detect_orientation.py --all                 (all 49 LUT sessions)
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

# The Alcova mount point has been observed at more than one path across
# sessions (2026-09-30: /Volumes/cluster/alcova/..., 2026-10-01:
# /Volumes/alcova/... -- same share, no "cluster" prefix) -- try known
# candidates rather than hardcoding one that silently stops matching.
_ALCOVA_CANDIDATES = [
    "/Volumes/cluster/alcova/bedfordlab/ThermalGradient/Process_Jason",
    "/Volumes/alcova/bedfordlab/ThermalGradient/Process_Jason",
]
ALCOVA_PROCESS_JASON = next((p for p in _ALCOVA_CANDIDATES if Path(p).is_dir()), _ALCOVA_CANDIDATES[0])
OUT_DIR = f"{Path(__file__).resolve().parent.parent}/thermalFeatures/orientation_check"

THERMAL_MAX_FRAMES = 1200  # ~90-150s at 8-10fps, bounded sequential read
THERMAL_DIFF_THRESH_C = 3.0  # body-heat-scale signal above a quiet LOCAL background

RGB_WINDOW_SEC = 120
RGB_SAMPLE_HZ = 2  # NOTE: RGB seeks cost ~0.28s each (confirmed 2026-09-29) --
                    # raising this multiplies real per-session cost directly
                    # (720 samples/session at 6Hz over a 120s window would be
                    # ~3.4min/session x 49 sessions ~= 2.75hr just for RGB).
                    # The wide-motion-trail problem is fixed below instead by
                    # tightening exclude_recent, which costs nothing extra.
RGB_DIFF_THRESH = 100.0  # empirical (2026-09-29, post median-correction): sum-of-abs-diff across RGB channels

MIN_BLOB_AREA_FRAC = 0.01  # blob must cover at least 1% of the frame to count


ALCOVA_THERMALGRADIENT_ROOT = str(Path(ALCOVA_PROCESS_JASON).parent)


def find_session_files(video_name_seq, video_name):
    """
    Locate the raw .seq and RGB .mp4 for one LUT row's session.

    Most sessions live under Process_Jason/*/ (date-folders, flat files -- no
    further nesting). Rehabituation sessions (2026-09-29: confirmed) instead
    live directly under ThermalGradient/<date>_Rehabituation/, a sibling of
    Process_Jason -- search both.
    """
    search_globs = [
        Path(ALCOVA_PROCESS_JASON).glob(f"*/{video_name_seq}.seq"),
        Path(ALCOVA_THERMALGRADIENT_ROOT).glob(f"*_Rehabituation/{video_name_seq}.seq"),
    ]
    seq_matches = sorted({p for g in search_globs for p in g})
    search_globs_mp4 = [
        Path(ALCOVA_PROCESS_JASON).glob(f"*/{video_name}"),
        Path(ALCOVA_THERMALGRADIENT_ROOT).glob(f"*_Rehabituation/{video_name}"),
    ]
    mp4_matches = sorted({p for g in search_globs_mp4 for p in g})
    seq = str(seq_matches[0]) if seq_matches else None
    mp4 = str(mp4_matches[0]) if mp4_matches else None
    return seq, mp4, len(seq_matches), len(mp4_matches)


def _largest_blob(diff, thresh, min_area):
    """
    Largest connected above-threshold region, with an intensity-WEIGHTED centroid
    rather than the binary mask's plain geometric centroid. Added 2026-09-29: a
    moving hand can leave a long, thin, low-and-uneven-intensity motion trail
    along whatever it passes over (confirmed: a bright reflective floor strip),
    which connected-components correctly merges into one region, but a plain
    geometric centroid then lands near the middle of that whole trail rather than
    on the actual object -- weighting by diff magnitude pulls it back toward
    wherever the change was strongest, which is reliably the object itself, not
    a faint trailing edge.
    """
    mask = (diff > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    component_mask = labels == i
    weights = np.clip(diff, 0, None) * component_mask
    total_w = float(weights.sum())
    if total_w > 0:
        ys, xs = np.indices(diff.shape)
        cx = float((xs * weights).sum() / total_w)
        cy = float((ys * weights).sum() / total_w)
    else:
        cx, cy = (float(v) for v in centroids[i])
    bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
    return dict(area=area, cx=cx, cy=cy, bbox=bbox)


def _find_motion_peak(frame_iter, diff_fn, motion_pixel_thresh, max_frames,
                       history_len=8, min_bg_frames=4, exclude_recent=2):
    """
    Single-pass, memory-bounded motion-peak finder shared by both modalities.

    2026-09-29: the first version of this tool diffed every frame against a
    FIXED early-session background, and got fooled by two different real, large,
    slow-but-expected signals: the thermal gradient still forming over the first
    ~90s (the cool end keeps cooling -- nothing to do with the intrusion), and an
    unexplained RGB brightness/shadow shift over a similar timescale. Both are
    slow relative to how fast an arm sweeps through frame, so tracking motion
    between NEARBY frames (not a frame from session start) sidesteps both: a
    slow drift barely shows up frame-to-frame, while a real intrusion still
    produces a sharp spike.

    frame_iter yields (key, frame) pairs in chronological order. Keeps a bounded
    rolling history (deque semantics via list-trim); whenever the current frame's
    motion energy (pixel count above motion_pixel_thresh, vs the immediately
    previous frame) sets a new high, snapshots (frame, local background = median
    of history excluding the most recent `exclude_recent` frames, key) as the
    running best candidate -- so the "background" used to localize the blob is
    always recent, not from session start.
    """
    history = []
    prev_frame = None
    best = None
    n = 0
    for key, frame in frame_iter:
        n += 1
        if prev_frame is not None:
            motion_energy = float((diff_fn(frame, prev_frame) > motion_pixel_thresh).sum())
            if len(history) >= min_bg_frames and (best is None or motion_energy > best["motion"]):
                bg_frames = history[:-exclude_recent] if len(history) > exclude_recent else history
                if len(bg_frames) >= min_bg_frames:
                    local_bg = np.median(np.stack(bg_frames), axis=0)
                    best = dict(key=key, frame=frame, local_bg=local_bg, motion=motion_energy)
        history.append(frame)
        if len(history) > history_len:
            history.pop(0)
        prev_frame = frame
        if max_frames and n >= max_frames:
            break
    return best


def find_intrusion_thermal(seq_path, max_frames=THERMAL_MAX_FRAMES, thresh_c=THERMAL_DIFF_THRESH_C,
                            motion_thresh_c=1.0):
    planck = read_planck_constants(seq_path)
    reader = SeqReader(seq_path)

    def gen():
        for idx, raw in reader.frames():
            yield idx, raw_to_celsius(raw, planck)

    best = _find_motion_peak(gen(), diff_fn=lambda a, b: np.abs(a - b),
                              motion_pixel_thresh=motion_thresh_c, max_frames=max_frames)
    reader.close()
    if best is None:
        return None
    diff = np.abs(best["frame"] - best["local_bg"])
    diff = diff - np.median(diff)  # extra safety net; see module docstring
    min_area = MIN_BLOB_AREA_FRAC * best["frame"].size
    blob = _largest_blob(diff, thresh_c, min_area)
    if blob is None:
        return None
    h, w = best["frame"].shape
    return dict(idx=best["key"], frame=best["frame"], cx_frac=blob["cx"] / w, cy_frac=blob["cy"] / h,
                bbox=blob["bbox"], area=blob["area"])


def _rgb_diff(a, b):
    return np.abs(a - b).sum(axis=2)


def find_intrusion_rgb(video_path, window_sec=RGB_WINDOW_SEC, sample_hz=RGB_SAMPLE_HZ,
                        thresh=RGB_DIFF_THRESH, motion_thresh=20.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / sample_hz)))
    n_samples = int(window_sec * sample_hz)

    def gen():
        for i in range(n_samples):
            frame_idx = i * stride
            if frame_idx >= total:
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            yield frame_idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

    # exclude_recent=1 (vs the default 2): at 2Hz sampling, a background excluding
    # only the immediately-preceding sample (0.5s gap to "current") keeps the
    # object's real displacement between background and current small, instead of
    # the ~1-1.5s gap the default would give -- directly shrinks the motion-trail
    # length that was pulling the RGB centroid off-target (2026-09-29), at no
    # extra I/O cost (same samples, just a different background window).
    best = _find_motion_peak(gen(), diff_fn=_rgb_diff, motion_pixel_thresh=motion_thresh,
                              max_frames=n_samples, exclude_recent=1)
    cap.release()
    if best is None:
        return None
    diff = _rgb_diff(best["frame"], best["local_bg"])
    diff = diff - np.median(diff)  # extra safety net; see module docstring
    min_area = MIN_BLOB_AREA_FRAC * (best["frame"].shape[0] * best["frame"].shape[1])
    blob = _largest_blob(diff, thresh, min_area)
    if blob is None:
        return None
    h, w = best["frame"].shape[:2]
    return dict(t=best["key"], frame=best["frame"].astype(np.uint8), cx_frac=blob["cx"] / w, cy_frac=blob["cy"] / h,
                bbox=blob["bbox"], area=blob["area"])


def classify_side(frac, tol=0.15):
    """left/right (or top/bottom) if clearly off-center; "center" (ambiguous) if within tol of 0.5."""
    if abs(frac - 0.5) < tol:
        return "center"
    return "low" if frac < 0.5 else "high"


def classify_flip(rgb_result, thermal_result):
    h_rgb, h_th = classify_side(rgb_result["cx_frac"]), classify_side(thermal_result["cx_frac"])
    v_rgb, v_th = classify_side(rgb_result["cy_frac"]), classify_side(thermal_result["cy_frac"])

    def axis_call(a, b):
        if "center" in (a, b):
            return "ambiguous"
        return "flip" if a != b else "no_flip"

    h_call = axis_call(h_rgb, h_th)
    v_call = axis_call(v_rgb, v_th)
    label_parts = []
    if h_call == "flip":
        label_parts.append("horizontal")
    if v_call == "flip":
        label_parts.append("vertical")
    if h_call == "ambiguous" or v_call == "ambiguous":
        label = "AMBIGUOUS (" + ",".join(label_parts + ["?"]) + ")"
    elif not label_parts:
        label = "none"
    else:
        label = "+".join(label_parts)
    return label, h_call, v_call


def render_pair(name, rgb_result, thermal_result, label, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    axL.imshow(rgb_result["frame"])
    x, y, w, h = rgb_result["bbox"]
    axL.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
    axL.plot(rgb_result["cx_frac"] * rgb_result["frame"].shape[1],
              rgb_result["cy_frac"] * rgb_result["frame"].shape[0], "r+", markersize=15, markeredgewidth=3)
    axL.set_title(f"RGB  t={rgb_result['t']:.1f}s  blob@({rgb_result['cx_frac']:.2f},{rgb_result['cy_frac']:.2f})", fontsize=9)
    axL.axis("off")

    im = axR.imshow(thermal_result["frame"], cmap="jet")
    plt.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="°C")
    x, y, w, h = thermal_result["bbox"]
    axR.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
    axR.plot(thermal_result["cx_frac"] * thermal_result["frame"].shape[1],
              thermal_result["cy_frac"] * thermal_result["frame"].shape[0], "r+", markersize=15, markeredgewidth=3)
    axR.set_title(f"Thermal  idx={thermal_result['idx']}  blob@({thermal_result['cx_frac']:.2f},{thermal_result['cy_frac']:.2f})", fontsize=9)
    axR.axis("off")

    fig.suptitle(f"{name}  ->  detected flip: {label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def process_session(name, seq_path, rgb_path):
    print(f"\n=== {name} ===", flush=True)
    print(f"  seq={seq_path}\n  rgb={rgb_path}", flush=True)
    thermal_result = find_intrusion_thermal(seq_path)
    rgb_result = find_intrusion_rgb(rgb_path)
    if thermal_result is None or rgb_result is None:
        print(f"  NO DETECTION: thermal={'ok' if thermal_result else 'MISSING'} "
              f"rgb={'ok' if rgb_result else 'MISSING'}", flush=True)
        return dict(session=name, status="no_detection", flip=None)

    label, h_call, v_call = classify_flip(rgb_result, thermal_result)
    print(f"  thermal blob: idx={thermal_result['idx']} area={thermal_result['area']} "
          f"pos=({thermal_result['cx_frac']:.2f},{thermal_result['cy_frac']:.2f})", flush=True)
    print(f"  rgb blob:     t={rgb_result['t']:.1f}s area={rgb_result['area']} "
          f"pos=({rgb_result['cx_frac']:.2f},{rgb_result['cy_frac']:.2f})", flush=True)
    print(f"  ==> flip: {label}  (h={h_call}, v={v_call})", flush=True)

    out_path = f"{OUT_DIR}/{name}_auto_orientation.png"
    render_pair(name, rgb_result, thermal_result, label, out_path)
    print(f"  saved {out_path}", flush=True)
    return dict(session=name, status="ok", flip=label, h_call=h_call, v_call=v_call,
                thermal_idx=thermal_result["idx"], thermal_area=thermal_result["area"],
                rgb_t=rgb_result["t"], rgb_area=rgb_result["area"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="*", help="Video_name_SEQ stems to process")
    parser.add_argument("--all", action="store_true", help="process every session in the LUT")
    args = parser.parse_args()

    # Known-answer validation targets (default when no args given)
    KNOWN = {
        "07-28-25_4540_B_4541_F_Test3-004": "vertical",
        "07-30-25_4540_F_4541_B_Test4-008": "horizontal+vertical",
        "08-07-25_4541_F_4540_B_Test7-020": "horizontal+vertical",
    }

    lut = load_lut(str(Path(__file__).resolve().parent.parent / "metadata" / "LUT_CLEAN_July6.csv"))
    kept, _ = filter_excluded(lut)
    sessions = kept[["Video_name_SEQ", "Video_name"]].drop_duplicates().dropna()

    if args.all:
        targets = list(sessions.itertuples(index=False))
    elif args.sessions:
        targets = [row for row in sessions.itertuples(index=False) if row.Video_name_SEQ in args.sessions]
    else:
        targets = [row for row in sessions.itertuples(index=False) if row.Video_name_SEQ in KNOWN]

    results = []
    for row in targets:
        seq_path, mp4_path, n_seq, n_mp4 = find_session_files(row.Video_name_SEQ, row.Video_name)
        if seq_path is None or mp4_path is None:
            print(f"\n=== {row.Video_name_SEQ} ===\n  FILE NOT FOUND: seq_matches={n_seq} mp4_matches={n_mp4}", flush=True)
            results.append(dict(session=row.Video_name_SEQ, status="file_not_found", flip=None))
            continue
        r = process_session(row.Video_name_SEQ, seq_path, mp4_path)
        if r["session"] in KNOWN:
            expected = KNOWN[r["session"]]
            match = "MATCH" if r.get("flip") == expected else "MISMATCH"
            print(f"  known-answer check: expected={expected} got={r.get('flip')} -> {match}", flush=True)
            r["expected"] = expected
        results.append(r)

    out_csv = f"{OUT_DIR}/orientation_report.csv"
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n=== DONE === report written to {out_csv}")
