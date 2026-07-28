"""
Tune the mouse detector from manually labeled frames.

This script reads labelme JSON annotations (or a folder of binary mask PNGs),
pairs each annotation with its raw .seq frame, and finds the foreground
threshold sigma that best separates mouse from background.

The optimized sigma value is written back to tracking_config.json.
If the automated thresholder still performs poorly, the script prints
suggestions for next steps.

Usage:
    python scripts/train_mouse_detector.py \\
        --seq "/path/to/cropped_file_Front.seq" \\
        --labels "training_frames/" \\
        --config "tracking_config.json"

Label format (choose one):
    A) labelme JSON files in training_frames/labels/ (one per image)
    B) Binary mask PNGs in training_frames/masks/
       Mask pixels = 255 where the mouse is, 0 elsewhere.
       Filenames must match the image names: frame_000123_mask.png

Output:
    Prints the best-fit sigma and updates tracking_config.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader
from src.mouse_segmentation import BackgroundModel, segment_mouse
from src.arena_mask import TrackingConfig


def load_labelme_mask(json_path: Path, image_shape) -> Optional[np.ndarray]:
    """Convert a labelme JSON annotation to a binary mask."""
    import cv2
    with open(json_path) as f:
        data = json.load(f)
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def load_labels(labels_dir: Path, image_shape) -> Dict[int, np.ndarray]:
    """
    Return {frame_index: binary_mask} from either labelme JSONs or mask PNGs.
    """
    from PIL import Image

    masks = {}
    labels_path = labels_dir

    # Try labelme JSON files first
    json_files = list(labels_path.glob("**/*.json")) + list(labels_path.glob("*.json"))
    png_files = list(labels_path.glob("**/*_mask.png")) + list(labels_path.glob("*_mask.png"))

    if json_files:
        print(f"  Found {len(json_files)} labelme JSON annotations")
        for jf in json_files:
            # Extract frame index from filename like "frame_000123.json"
            import re
            m = re.search(r"frame_(\d+)", jf.stem)
            if m:
                idx = int(m.group(1))
                masks[idx] = load_labelme_mask(jf, image_shape)
    elif png_files:
        print(f"  Found {len(png_files)} mask PNG files")
        for pf in png_files:
            import re
            m = re.search(r"frame_(\d+)", pf.stem)
            if m:
                idx = int(m.group(1))
                arr = np.array(Image.open(pf).convert("L"))
                masks[idx] = arr > 127
    else:
        print("  No label files found. Expected either:")
        print("    - labelme JSON files (frame_NNNNNN.json)")
        print("    - binary mask PNGs  (frame_NNNNNN_mask.png)")

    return masks


def iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection-over-union between two boolean masks."""
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(inter) / max(float(union), 1)


def main():
    parser = argparse.ArgumentParser(description="Tune mouse detector from labeled frames")
    parser.add_argument("--seq", required=True, help="Path to cropped .seq file")
    parser.add_argument("--labels", required=True, help="Directory containing label files")
    parser.add_argument("--config", default="tracking_config.json", help="Config file to update")
    parser.add_argument("--n-background", type=int, default=200,
                        help="Background frames for model (default: 200)")
    args = parser.parse_args()

    seq_path = Path(args.seq)
    labels_dir = Path(args.labels)

    if not seq_path.exists():
        print(f"ERROR: .seq file not found: {seq_path}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if config_path.exists():
        config = TrackingConfig.load(str(config_path))
    else:
        config = TrackingConfig()

    print(f"Building background model from {seq_path.name}…")
    with SeqReader(str(seq_path)) as reader:
        image_shape = reader.frame_shape
        bg_model = BackgroundModel.build(reader, args.n_background, config.random_seed)

    print(f"Loading labels from {labels_dir}…")
    label_masks = load_labels(labels_dir, image_shape)

    if not label_masks:
        print("No labels loaded. Cannot tune detector.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(label_masks)} labeled frames")

    # Read the labeled frames from the .seq file
    print("Reading labeled frames…")
    frames = {}
    with SeqReader(str(seq_path)) as reader:
        for frame_idx, pixel_data in reader.frames():
            if frame_idx in label_masks:
                frames[frame_idx] = pixel_data
            if len(frames) == len(label_masks):
                break

    # Grid search over sigma values
    sigmas = np.arange(1.0, 6.1, 0.5)
    print(f"\nSearching over sigma values: {sigmas.tolist()}")

    best_sigma = config.segmentation_threshold_sigma
    best_mean_iou = -1.0
    results = []

    for sigma in sigmas:
        ious = []
        for frame_idx, pixel_data in frames.items():
            true_mask = label_masks[frame_idx]
            pred_mask, _, _, _ = segment_mouse(
                pixel_data,
                bg_model,
                min_area=config.min_mouse_area_px,
                max_area=config.max_mouse_area_px,
                threshold_sigma=float(sigma),
            )
            if pred_mask is None:
                pred_mask = np.zeros(image_shape, dtype=bool)
            ious.append(iou(pred_mask, true_mask))

        mean_iou = float(np.mean(ious))
        results.append((float(sigma), mean_iou))
        print(f"  sigma={sigma:.1f}  mean IoU={mean_iou:.3f}")

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_sigma = float(sigma)

    print(f"\nBest sigma: {best_sigma:.1f}  (mean IoU={best_mean_iou:.3f})")

    if best_mean_iou < 0.5:
        print(
            "\nWARNING: Best mean IoU < 0.5. Background subtraction may not be reliable\n"
            "for this video. Possible causes:\n"
            "  - Mouse is often near its own body temperature (low thermal contrast)\n"
            "  - Background model was built while mouse was stationary\n"
            "Consider increasing the number of labeled frames, or try a different\n"
            "segmentation approach (edge detection, supervised classifier)."
        )

    # Update config
    config.segmentation_threshold_sigma = best_sigma
    config.save(str(config_path))
    print(f"\nUpdated segmentation_threshold_sigma={best_sigma} in {config_path}")


if __name__ == "__main__":
    main()
