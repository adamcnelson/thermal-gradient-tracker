"""
Interactively pick RGB<->thermal point correspondences for one session/track
and fit+save the Stage 4 spatial-registration homography (project_brief_v7.md
§6 Stage 4).

Click corresponding points ALTERNATING panels: click a point in the RGB
panel (left), then click its matching physical point in the thermal panel
(right) — e.g. the same plate corner in both. The tool enforces this
alternation so pairing can't get scrambled. Click at least 4 pairs (plate
corners plus edge midpoints recommended for a better-conditioned fit).

*** ORIENTATION WARNING (confirmed 2026-08-13, lane F): the RGB and
thermal cameras do NOT share the same orientation — the apparatus's front
edge sits at the BOTTOM of the RGB crop but the TOP of the thermal crop (a
~180-degree relative rotation, not just a vertical flip). A same-screen-
position clicking habit (e.g. "top-left of each panel") silently pairs the
WRONG physical corners — this happened for real on the first calibration
attempt and produced a low-RMSE but physically wrong homography. For each
pair, identify the actual physical feature (corner/edge) in BOTH images
before clicking — don't rely on relative on-screen position. ***

Both panels show a temporal MEDIAN frame (not a single frame), so a mouse
sitting on a corner at some random instant doesn't block it.

Usage:
    python scripts/calibrate_homography.py \\
        --seq "/path/to/croppedSeqFiles/.../07-28-25_..._Front.seq" \\
        --video "/path/to/Process_Jason/07-28-25_Test_3/2025-07-28_10-57-21.mp4" \\
        --lane F \\
        --output-dir homography_calibration

Press ENTER when done (needs >=4 pairs). Press Z to undo the last click.
Output: <output-dir>/<seq_stem>_homography.json (points, H, rmse_px) and
a preview PNG showing reprojected points against the thermal frame.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("MacOSX")  # native interactive backend for macOS
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader, adu_to_display
from src.mouse_segmentation import BackgroundModel
from src.landmarks.webcam_preprocessing import LANE_TOP, detect_track_split_row, split_track_crops
from src.landmarks.rgb_landmarks import RgbBackgroundModel
from src.landmarks.registration import fit_homography


def thermal_background_frame(seq_path: str, n_frames: int = 100, seed: int = 123) -> np.ndarray:
    with SeqReader(seq_path) as reader:
        bg = BackgroundModel.build(reader, n_frames, seed)
    return adu_to_display(bg.background.astype(np.uint16))


def rgb_background_frame(video_path: str, lane: str, n_samples: int = 30) -> np.ndarray:
    """
    Temporal median RGB background, sampled evenly across the FULL video
    duration via direct seeking — matching thermal_background_frame's
    BackgroundModel.build(), which samples evenly across the whole .seq.

    An earlier version of this function sampled a fixed stride from frame
    0 (~first 100s of a ~40min session, whatever n_samples*stride worked
    out to). That's a real inconsistency with the thermal side, not just a
    style choice: confirmed during real-session calibration (2026-08-13)
    that a nestlet visible early was faint-to-invisible in the (properly
    full-session) thermal background — nestlets move at least a bit over
    the course of every session, so an RGB background built from only the
    first couple of minutes isn't a fair comparison to a thermal background
    that reflects the whole session, and the two could end up disagreeing
    about where (or whether) a moved object shows up.

    Also uses detect_track_split_row() rather than split_track_crops()'s
    naive height//2 default — confirmed WRONG for the real rig (2026-08-13):
    the true gap between tracks sits well off-center (row ~725 of 1080, not
    540), so the naive split let one track bleed into the other's crop.
    Detected once from the first frame and reused for every sample, so all
    sampled frames are cropped consistently.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not determine frame count for {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first_frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read the first frame of {video_path}")
        split_row = detect_track_split_row(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
        print(f"  Detected track split row: {split_row} (of {first_frame.shape[0]})")

        indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(gray, split_row=split_row)
            frames.append(top if lane == LANE_TOP else bottom)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Could not read any frames from {video_path}")
    model = RgbBackgroundModel.build(frames)
    return model.background.clip(0, 255).astype(np.uint8)


def pick_point_pairs_interactive(rgb_img: np.ndarray, thermal_img: np.ndarray):
    """Open a two-panel window; alternate clicks: RGB point, then its thermal match."""
    fig, (ax_rgb, ax_thermal) = plt.subplots(1, 2, figsize=(16, 6))
    ax_rgb.imshow(rgb_img, cmap="gray", aspect="equal")
    ax_rgb.set_title("RGB (click point N here first)")
    thermal_cmap = plt.get_cmap("inferno")
    thermal_rgb = (thermal_cmap(thermal_img / 255.0)[:, :, :3] * 255).astype(np.uint8)
    ax_thermal.imshow(thermal_rgb, aspect="equal")
    ax_thermal.set_title("Thermal (then click matching point N here)")
    fig.suptitle(
        "Alternate clicks: RGB point -> matching thermal point -> repeat. "
        "ENTER when done (>=4 pairs)  |  Z to undo last click\n"
        "WARNING: RGB and thermal are NOT the same orientation (confirmed ~180-deg rotated, "
        "lane F, 2026-08-13) — match physical features, not screen position.",
        fontsize=10,
        color="red",
    )

    rgb_points: list = []
    thermal_points: list = []
    expecting = {"axis": "rgb"}  # alternation state

    def redraw():
        for ax, pts in ((ax_rgb, rgb_points), (ax_thermal, thermal_points)):
            for artist in list(ax.texts) + [c for c in ax.collections]:
                artist.remove()
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, c="lime", s=40, zorder=5)
                for i, (x, y) in enumerate(pts):
                    ax.text(x + 4, y - 4, str(i + 1), color="lime", fontsize=11, zorder=6)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button != 1:
            return
        if expecting["axis"] == "rgb" and event.inaxes == ax_rgb:
            rgb_points.append([int(event.xdata), int(event.ydata)])
            expecting["axis"] = "thermal"
            redraw()
        elif expecting["axis"] == "thermal" and event.inaxes == ax_thermal:
            thermal_points.append([int(event.xdata), int(event.ydata)])
            expecting["axis"] = "rgb"
            redraw()
        # Clicks in the "wrong" panel for the current expected step are ignored —
        # this is what enforces correct pairing rather than relying on the user
        # to alternate correctly on their own.

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key in ("z", "Z"):
            if expecting["axis"] == "rgb" and thermal_points:
                thermal_points.pop()
                expecting["axis"] = "thermal"
            elif expecting["axis"] == "thermal" and rgb_points:
                rgb_points.pop()
                expecting["axis"] = "rgb"
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return rgb_points, thermal_points


def main():
    parser = argparse.ArgumentParser(description="Interactively calibrate an RGB->thermal homography")
    parser.add_argument("--seq", required=True, help="Path to cropped .seq file (defines the thermal side)")
    parser.add_argument("--video", required=True, help="Path to the combined-frame webcam .mp4")
    parser.add_argument("--lane", required=True, choices=["B", "F"], help="Track lane: B (top) or F (bottom)")
    parser.add_argument("--output-dir", default="homography_calibration")
    parser.add_argument("--max-rmse-px", type=float, default=1.0, help="Stage 4 acceptance threshold")
    args = parser.parse_args()

    seq_path = Path(args.seq)
    video_path = Path(args.video)
    if not seq_path.exists():
        print(f"ERROR: .seq file not found: {seq_path}", file=sys.stderr)
        sys.exit(1)
    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building thermal background frame from {seq_path.name}…")
    thermal_img = thermal_background_frame(str(seq_path))
    print(f"Building RGB background frame from {video_path.name} (lane {args.lane})…")
    rgb_img = rgb_background_frame(str(video_path), args.lane)

    print()
    print("A matplotlib window will open with two panels.")
    print("Click a point in RGB, then its matching point in Thermal, repeat for >=4 pairs.")
    print("Press ENTER when done, Z to undo the last click.")
    print()
    print("*** WARNING: RGB and thermal are confirmed NOT the same orientation (~180-deg")
    print("    rotated for lane F, 2026-08-13) — identify the actual physical feature in")
    print("    BOTH images before clicking. Do not assume matching screen position means")
    print("    matching physical point. ***")

    rgb_points, thermal_points = pick_point_pairs_interactive(rgb_img, thermal_img)

    if len(rgb_points) != len(thermal_points):
        print(
            f"ERROR: unequal point counts (rgb={len(rgb_points)}, thermal={len(thermal_points)}) "
            "— an unmatched click was left dangling. Re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(rgb_points) < 4:
        print(f"ERROR: need >=4 point pairs, got {len(rgb_points)}.", file=sys.stderr)
        sys.exit(1)

    rgb_points_arr = np.array(rgb_points, dtype=float)
    thermal_points_arr = np.array(thermal_points, dtype=float)
    fit = fit_homography(rgb_points_arr, thermal_points_arr)

    print()
    print(f"Fitted homography from {len(rgb_points)} point pairs.")
    print(f"Reprojection RMSE: {fit.rmse_px:.3f} px  (per-point: {np.round(fit.residuals_px, 2).tolist()})")
    if fit.passes_acceptance(args.max_rmse_px):
        print(f"PASS — below {args.max_rmse_px} px acceptance threshold.")
    else:
        print(
            f"WARNING: RMSE exceeds the {args.max_rmse_px} px acceptance threshold. "
            "Consider re-clicking with more/better-spread points before trusting this fit."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_path.stem}_homography.json"
    payload = {
        "seq_path": str(seq_path),
        "video_path": str(video_path),
        "lane": args.lane,
        "rgb_points": rgb_points,
        "thermal_points": thermal_points,
        "H": fit.H.tolist(),
        "rmse_px": fit.rmse_px,
        "residuals_px": fit.residuals_px.tolist(),
        "passes_acceptance": fit.passes_acceptance(args.max_rmse_px),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")

    # Preview: reproject the RGB points through H and overlay on the thermal image,
    # alongside the human-clicked thermal points, for a visual sanity check.
    from src.landmarks.registration import apply_homography

    reprojected = apply_homography(fit.H, rgb_points_arr)
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 6))
    thermal_cmap = plt.get_cmap("inferno")
    thermal_rgb = (thermal_cmap(thermal_img / 255.0)[:, :, :3] * 255).astype(np.uint8)
    ax.imshow(thermal_rgb)
    ax.scatter(thermal_points_arr[:, 0], thermal_points_arr[:, 1], c="lime", s=50, label="clicked (truth)")
    ax.scatter(reprojected[:, 0], reprojected[:, 1], c="cyan", marker="x", s=60, label="reprojected from RGB")
    for i in range(len(thermal_points_arr)):
        ax.plot(
            [thermal_points_arr[i, 0], reprojected[i, 0]],
            [thermal_points_arr[i, 1], reprojected[i, 1]],
            "r-",
            lw=1,
        )
    ax.set_title(f"Reprojection check — RMSE {fit.rmse_px:.3f} px")
    ax.legend(loc="upper right", fontsize=8)
    preview_path = out_dir / f"{seq_path.stem}_homography_preview.png"
    fig.savefig(str(preview_path), dpi=120)
    print(f"Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
