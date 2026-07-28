"""
Inspect a cropped .seq file and save preview images for manual inspection.

Usage:
    python scripts/preview_tracking_input.py --input "/path/to/cropped_file_Front.seq"

--input is the only path you need to supply. --output-dir defaults to
preview_tracking/ under the project root (src/paths.py).

Outputs:
    preview_tracking/
        <stem>_frame0000.png         first frame
        <stem>_frame<mid>.png        middle frame
        <stem>_frame<last>.png       last frame
        <stem>_background.png        temporal median of sampled frames (empty gradient)
        <stem>_variance.png          temporal variance (highlights movement)
        <stem>_info.txt              frame count, size, approx frame rate
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.seq_io import SeqReader, adu_to_display
from src.mouse_segmentation import BackgroundModel


def save_frame_image(frame: np.ndarray, path: str, title: str = "") -> None:
    gray = adu_to_display(frame)
    cmap = plt.get_cmap("inferno")
    rgb = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(rgb)
    img.save(path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Preview a cropped .seq file")
    parser.add_argument("--input", required=True, help="Path to cropped .seq file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: preview_tracking/ under the project root)")
    parser.add_argument("--n-background", type=int, default=100,
                        help="Frames to use for background model (default: 100)")
    args = parser.parse_args()

    seq_path = Path(args.input)
    if not seq_path.exists():
        print(f"ERROR: File not found: {seq_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else paths.PREVIEW_TRACKING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = seq_path.stem

    print(f"Opening: {seq_path.name}")

    with SeqReader(str(seq_path)) as reader:
        total = reader.count_frames()
        h, w = reader.frame_shape
        print(f"  Dimensions  : {w} x {h} pixels")
        print(f"  Frame count : ~{total}")
        print(f"  File size   : {seq_path.stat().st_size / 1e9:.2f} GB")

        # Collect a few representative frames
        target_indices = {0, total // 4, total // 2, 3 * total // 4, max(0, total - 1)}
        collected = {}
        all_frames_sample = []

        for frame_idx, pixel_data in reader.frames():
            if frame_idx in target_indices:
                collected[frame_idx] = pixel_data.copy()
            if frame_idx % max(1, total // args.n_background) == 0:
                all_frames_sample.append(pixel_data.copy())
            if frame_idx >= total - 1:
                break

        # Save representative frames
        for idx in sorted(collected):
            save_frame_image(
                collected[idx],
                str(out_dir / f"{stem}_frame{idx:06d}.png"),
                title=f"Frame {idx}",
            )

        # Background (temporal median)
        if all_frames_sample:
            print(f"  Building background from {len(all_frames_sample)} frames…")
            bg = BackgroundModel.build(reader, n_frames=args.n_background, random_seed=123)
            bg_uint16 = bg.background.clip(0, 65535).astype(np.uint16)
            save_frame_image(bg_uint16, str(out_dir / f"{stem}_background.png"), "Background")

            # Variance image (shows where the mouse moves)
            stack = np.stack(all_frames_sample, axis=0).astype(np.float32)
            variance = np.var(stack, axis=0)
            var_norm = ((variance - variance.min()) / max(variance.max() - variance.min(), 1) * 65535)
            save_frame_image(var_norm.astype(np.uint16), str(out_dir / f"{stem}_variance.png"),
                             "Variance (movement)")

    # Write info text
    info_path = out_dir / f"{stem}_info.txt"
    with open(info_path, "w") as f:
        f.write(f"File: {seq_path.name}\n")
        f.write(f"Size: {seq_path.stat().st_size / 1e9:.3f} GB\n")
        f.write(f"Dimensions: {w} x {h} pixels\n")
        f.write(f"Estimated frame count: {total}\n")
        f.write(f"Pixel format: uint16 radiometric (0.04 K per unit)\n")
        f.write(
            "Temperature calibration: T_celsius = raw_uint16 * 0.04 - 273.15\n"
            "Source: lab's seq_process pipeline (j-landen/seq_process, tiff2temp).\n"
        )
    print(f"  Info saved: {info_path}")
    print(f"\nPreview complete. Check {out_dir}/")


if __name__ == "__main__":
    main()
