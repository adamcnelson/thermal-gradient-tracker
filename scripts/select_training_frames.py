"""
Sample frames from a .seq file for manual mouse annotation.

Saves N randomly-selected frames as PNG images so the user can label the
mouse region using any annotation tool (e.g., labelme, napari, or even
a drawing app). Labeled masks can then be used with train_mouse_detector.py.

Usage:
    python scripts/select_training_frames.py \\
        --input "/path/to/cropped_file_Front.seq" \\
        --n-frames 50 \\
        --output-dir "training_frames"

Output structure:
    training_frames/
        images/
            frame_000123.png
            ...
        frame_indices.txt    list of frame numbers sampled
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader, adu_to_display
from src.arena_mask import TrackingConfig


def main():
    parser = argparse.ArgumentParser(description="Sample frames for manual annotation")
    parser.add_argument("--input", required=True, help="Path to cropped .seq file")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="Number of frames to sample (default: 50)")
    parser.add_argument("--output-dir", default="training_frames", help="Output directory")
    parser.add_argument("--config", default=None,
                        help="Optional tracking_config.json (reads random_seed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    seq_path = Path(args.input)
    if not seq_path.exists():
        print(f"ERROR: File not found: {seq_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve random seed
    seed = args.seed
    if seed is None and args.config:
        try:
            cfg = TrackingConfig.load(args.config)
            seed = cfg.random_seed
        except Exception:
            pass
    if seed is None:
        seed = 123

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening: {seq_path.name}")

    with SeqReader(str(seq_path)) as reader:
        total = reader.count_frames()
        print(f"  Total frames: ~{total}")

        rng = np.random.default_rng(seed)
        n = min(args.n_frames, total)
        selected = sorted(rng.choice(total, size=n, replace=False).tolist())
        selected_set = set(selected)

        print(f"  Sampling {n} frames (seed={seed})…")
        cmap = plt.get_cmap("inferno")
        saved = 0

        for frame_idx, pixel_data in reader.frames():
            if frame_idx not in selected_set:
                continue
            gray = adu_to_display(pixel_data)
            rgb = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
            from PIL import Image
            img_path = img_dir / f"frame_{frame_idx:06d}.png"
            Image.fromarray(rgb).save(str(img_path))
            saved += 1
            if saved % 10 == 0:
                print(f"  Saved {saved}/{n}…")
            if saved >= n:
                break

    # Save the list of frame indices
    index_path = out_dir / "frame_indices.txt"
    with open(index_path, "w") as f:
        f.write(f"# Sampled from: {seq_path.name}\n")
        f.write(f"# Random seed: {seed}\n")
        f.write(f"# Total frames in file: {total}\n")
        for idx in selected:
            f.write(f"{idx}\n")

    print(f"\nSaved {saved} frame images to: {img_dir}/")
    print(f"Frame index list: {index_path}")
    print()
    print("Next steps:")
    print("  1. Use labelme or napari to draw a polygon around the mouse in each image.")
    print("     labelme: pip install labelme && labelme training_frames/images/")
    print("  2. Once labeled, run scripts/train_mouse_detector.py to tune thresholds.")


if __name__ == "__main__":
    main()
