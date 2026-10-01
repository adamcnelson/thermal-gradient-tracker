"""
Roll out the Hybrid calibration (project_v8, 2026-10-01): apply one base
homography per orientation regime to every session beyond Test_3-004/
Test_4-008/Test_7-020, which already have real individually-fit homographies.

Regime assignment reuses orientation_classification_final.csv (already
validated with zero exceptions against real reads, see project memory
project_v8_orientation_detection.md) rather than re-deriving the date
cutoff here -- single source of truth.

Base homographies:
  vertical            -> Test_3-004 (re-fit 2026-09-30 against Alcova's
                          crop convention; see project memory
                          project_v8_homography_hybrid_calibration.md)
  horizontal+vertical  -> Test_4-008 (existing production fit; already in
                          Alcova's crop convention, no re-fit needed)

For every inherited session, also builds a visual QC overlay: this
session's own RGB background warped through the inherited H, thresholded
to a bright-rail silhouette, drawn over this session's own real thermal
background (same construction as validate_homography_transfer.py's
deeper check). Tiles of all sessions (grouped by regime, paginated) get
assembled into contact-sheet PNGs for a quick human eyeball pass.

NOTE (2026-10-01): an earlier version of this script tried an automated
numeric QC gate instead (sub-pixel dark-line-position offset vs. the base
session's own self-warp). It produced a confident-looking ~25px "offset"
on a real session (07-08-25_4539_B_4540_F) that turned out to be a false
alarm -- that session's warped frame has a sharp LOCAL dark pixel in every
column (so a naive contrast/sharpness guard doesn't catch it), but it's
unrelated dust/debris, not the same reflective rail feature the base
image's dark line tracks. Matching unrelated-but-individually-sharp
features produced a large, meaningless offset. This is the same class of
failure this project hit three times automating orientation detection
(project_v8_orientation_detection.md) -- dropped in favor of the same fix
that worked there: a human eyeballing a rendered image, not a numeric
threshold.

Output:
  homography_calibration_inherited/<Video_name_SEQ>_{Front,Back}_homography.json
  thermalFeatures/hybrid_calibration_qc/overlays/<Video_name_SEQ>_{Front,Back}.png
  thermalFeatures/hybrid_calibration_qc/contact_sheet_<regime>_pageN.png
  thermalFeatures/hybrid_calibration_qc/run_log.csv

Usage:
    python scripts/apply_hybrid_homography.py [--limit N] [--lanes F,B]
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from calibrate_homography import rgb_background_frame, thermal_background_frame
from detect_orientation import find_session_files, ALCOVA_THERMALGRADIENT_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIENTATION_CSV = REPO_ROOT / "thermalFeatures" / "orientation_check_grid" / "orientation_classification_final.csv"
OUT_DIR = REPO_ROOT / "homography_calibration_inherited"
QC_DIR = REPO_ROOT / "thermalFeatures" / "hybrid_calibration_qc"
CROPPED_SEQ_ROOT = Path(ALCOVA_THERMALGRADIENT_ROOT) / "croppedSeqFiles"

LANE_WORD = {"F": "Front", "B": "Back"}

BASE_HOMOGRAPHY = {
    "vertical": {
        "F": REPO_ROOT / "homography_calibration_alcova_crop" / "07-28-25_4540_B_4541_F_Test3-004_Front_homography.json",
        "B": REPO_ROOT / "homography_calibration_alcova_crop" / "07-28-25_4540_B_4541_F_Test3-004_Back_homography.json",
    },
    "horizontal+vertical": {
        "F": REPO_ROOT / "homography_calibration" / "07-30-25_4540_F_4541_B_Test4-008_Front_homography.json",
        "B": REPO_ROOT / "homography_calibration" / "07-30-25_4540_F_4541_B_Test4-008_Back_homography.json",
    },
}

# Already individually fit -- never overwrite these with an inherited calibration.
ALREADY_CALIBRATED = {
    "07-28-25_4540_B_4541_F_Test3-004",
    "07-30-25_4540_F_4541_B_Test4-008",
    "08-07-25_4541_F_4540_B_Test7-020",
}


def find_cropped_seq(video_name_seq: str, lane: str):
    matches = sorted(CROPPED_SEQ_ROOT.glob(f"*/{video_name_seq}_{LANE_WORD[lane]}.seq"))
    return str(matches[0]) if matches else None


def render_overlay(tgt_thermal: np.ndarray, warped_rgb: np.ndarray) -> np.ndarray:
    """Target thermal background (colorized) + bright-rail silhouette from
    the warped target RGB (lime contour) -- same construction as
    validate_homography_transfer.py's deeper check, but returned as a small
    BGR array (not a full matplotlib figure) so many of these can be tiled
    cheaply into a contact sheet."""
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2GRAY) if warped_rgb.ndim == 3 else warped_rgb
    mask = gray > np.percentile(gray[gray > 0], 60) if (gray > 0).any() else np.zeros_like(gray, bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thermal_u8 = cv2.normalize(tgt_thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
        if tgt_thermal.dtype != np.uint8 else tgt_thermal
    color_bg = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)
    big_contours = [c for c in contours if cv2.contourArea(c) > 50]
    cv2.drawContours(color_bg, big_contours, -1, (0, 255, 0), 1)
    return color_bg


def build_contact_sheet(tiles, out_path: Path, tile_scale: int = 3, label_height: int = 18, page_size: int = 10):
    """tiles: list of (label, BGR array). Paginates into out_path with
    '_page{N}' inserted before the suffix."""
    for page_i in range(0, len(tiles), page_size):
        page_tiles = tiles[page_i:page_i + page_size]
        strips = []
        for label, arr in page_tiles:
            h, w = arr.shape[:2]
            big = cv2.resize(arr, (w * tile_scale, h * tile_scale), interpolation=cv2.INTER_NEAREST)
            strip = np.full((label_height, big.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(strip, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            strips.append(np.vstack([strip, big]))
        sheet = np.vstack(strips)
        page_path = out_path.with_name(f"{out_path.stem}_page{page_i // page_size + 1}{out_path.suffix}")
        cv2.imwrite(str(page_path), sheet)
        print(f"  wrote {page_path} ({len(page_tiles)} tiles)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most N sessions (for a quick test run)")
    parser.add_argument("--lanes", default="F,B")
    args = parser.parse_args()
    lanes = args.lanes.split(",")

    orient = pd.read_csv(ORIENTATION_CSV)
    orient["DateParsed"] = pd.to_datetime(orient["Date"], format="%m/%d/%y")
    orient = orient.sort_values("DateParsed")  # Test_3-onward first within each regime, oldest first

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = QC_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tiles_by_regime = {}
    n_done = 0
    for _, row in orient.iterrows():
        if args.limit and n_done >= args.limit:
            break
        stem = row["Video_name_SEQ"]
        regime = row["flip"]
        if stem in ALREADY_CALIBRATED:
            continue
        n_done += 1
        if row["data_status"] == "CORRUPTED":
            rows.append(dict(session=stem, lane="-", status="skip_corrupted", note=row["data_note"]))
            continue
        if regime not in BASE_HOMOGRAPHY:
            rows.append(dict(session=stem, lane="-", status=f"skip_regime_{regime}", note=""))
            continue

        video_name = row["Video_name"]
        _, mp4_path, _, n_mp4 = find_session_files(stem, video_name)
        if not mp4_path:
            rows.append(dict(session=stem, lane="-", status="skip_no_rgb_video", note=""))
            continue

        for lane in lanes:
            cropped_seq = find_cropped_seq(stem, lane)
            if not cropped_seq:
                rows.append(dict(session=stem, lane=lane, status="skip_no_cropped_seq", note=""))
                continue

            base_path = BASE_HOMOGRAPHY[regime][lane]
            base_json = json.load(open(base_path))
            H = np.array(base_json["H"], dtype=np.float64)

            out_json = dict(base_json)
            out_json.update({
                "seq_path": cropped_seq,
                "video_path": mp4_path,
                "lane": lane,
                "source": "inherited_base_regime",
                "regime": regime,
                "base_homography": base_path.name,
            })
            out_path = OUT_DIR / f"{stem}_{LANE_WORD[lane]}_homography.json"
            with open(out_path, "w") as f:
                json.dump(out_json, f, indent=2)

            # Visual QC: this session's own RGB background, warped through
            # the inherited H, overlaid on this session's own real thermal
            # background.
            try:
                tgt_rgb = rgb_background_frame(mp4_path, lane)
                tgt_thermal = thermal_background_frame(cropped_seq)
                th, tw = tgt_thermal.shape[:2]
                tgt_warped = cv2.warpPerspective(tgt_rgb, H, (tw, th))
                overlay = render_overlay(tgt_thermal, tgt_warped)
            except Exception as exc:
                rows.append(dict(session=stem, lane=lane, status="qc_error", note=str(exc)))
                print(f"  {stem} [{lane}] regime={regime} -> qc_error: {exc}", flush=True)
                continue

            cv2.imwrite(str(overlay_dir / f"{stem}_{LANE_WORD[lane]}.png"), overlay)
            label = f"{stem} [{LANE_WORD[lane]}] ({regime})"
            tiles_by_regime.setdefault(regime, []).append((label, overlay))

            rows.append(dict(session=stem, lane=lane, status="rendered",
                              base_homography=base_path.name, regime=regime))
            print(f"  {stem} [{lane}] regime={regime} -> rendered", flush=True)

    summary = pd.DataFrame(rows)
    out_csv = QC_DIR / "run_log.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n{len(summary)} session-lane rows -> {out_csv}")
    print(summary["status"].value_counts())

    for regime, tiles in tiles_by_regime.items():
        safe_regime = regime.replace("+", "_")
        build_contact_sheet(tiles, QC_DIR / f"contact_sheet_{safe_regime}.png")


if __name__ == "__main__":
    main()
