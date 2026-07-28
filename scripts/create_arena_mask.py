"""
Interactively define the valid floor/arena region and save it to tracking_config.json.

Click polygon vertices around the usable thermal-gradient surface (the walkable
floor, excluding walls, apparatus edges, and background).  The polygon is saved
to tracking_config.json alongside the processing parameters.

Usage:
    python scripts/create_arena_mask.py \\
        --input "/path/to/cropped_file_Front.seq" \\
        --config "tracking_config.json" \\
        --output-dir "mask_preview"

If tracking_config.json already exists, the polygon is updated in place.
If it does not exist, a new config is created with default parameters.
Press ENTER when done clicking to close the window and save.
Press Z to undo the last point.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("MacOSX")  # native interactive backend for macOS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPoly
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader, adu_to_display
from src.arena_mask import ArenaMask, TrackingConfig


def pick_polygon_interactive(frame: np.ndarray) -> list:
    """Open a matplotlib window and collect polygon vertices by clicking."""
    gray = adu_to_display(frame)
    cmap = plt.get_cmap("inferno")
    rgb = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb, aspect="equal")
    ax.set_title(
        "Click to define the valid floor polygon (exclude walls & edges)\n"
        "Press ENTER when done  |  Press Z to undo last point",
        fontsize=10,
    )

    points = []
    scatter = ax.scatter([], [], c="lime", s=40, zorder=5)
    poly_line, = ax.plot([], [], "lime", lw=1.5)

    def update_display():
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            scatter.set_offsets(np.column_stack([xs, ys]))
            # Close the loop for display
            poly_line.set_data(xs + [xs[0]], ys + [ys[0]])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            poly_line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            points.append([int(event.xdata), int(event.ydata)])
            update_display()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key in ("z", "Z") and points:
            points.pop()
            update_display()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return points


def main():
    parser = argparse.ArgumentParser(description="Interactively define arena mask polygon")
    parser.add_argument("--input", required=True, help="Path to cropped .seq file")
    parser.add_argument("--config", default="tracking_config.json", help="Config file path")
    parser.add_argument("--output-dir", default="mask_preview", help="Mask preview output dir")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to display (default: 0)")
    args = parser.parse_args()

    seq_path = Path(args.input)
    if not seq_path.exists():
        print(f"ERROR: File not found: {seq_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)

    # Load or create config
    if config_path.exists():
        config = TrackingConfig.load(str(config_path))
        print(f"Loaded existing config: {config_path}")
    else:
        config = TrackingConfig()
        print(f"Creating new config: {config_path}")

    # Read the selected frame
    print(f"Reading frame {args.frame} from {seq_path.name}…")
    with SeqReader(str(seq_path)) as reader:
        h, w = reader.frame_shape
        frame = reader.read_frame(args.frame)

    print(f"Frame shape: {w}x{h}")
    print()
    print("A matplotlib window will open. Click to define the polygon around the valid floor area.")
    print("Press ENTER to finish, Z to undo last point.")

    polygon_xy = pick_polygon_interactive(frame)

    if len(polygon_xy) < 3:
        print("ERROR: Need at least 3 points to define a polygon.", file=sys.stderr)
        sys.exit(1)

    # Save the polygon to config
    config.update_polygon(str(config_path), polygon_xy, h, w)
    print(f"Polygon with {len(polygon_xy)} vertices saved to {config_path}")

    # Save a preview image showing the mask
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt2
    from PIL import Image

    mask_obj = ArenaMask.from_polygon(polygon_xy, (h, w))
    gray = adu_to_display(frame)
    cmap = plt2.get_cmap("inferno")
    rgb = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)

    # Dim pixels outside the mask
    outside = ~mask_obj.mask
    rgb[outside] = (rgb[outside] * 0.4).astype(np.uint8)

    # Draw polygon outline
    import cv2
    pts = np.array(polygon_xy, dtype=np.int32)
    cv2.polylines(rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    preview_path = out_dir / f"{seq_path.stem}_arena_mask_preview.png"
    Image.fromarray(rgb).save(str(preview_path))
    print(f"Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
