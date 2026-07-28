"""
Track the mouse and extract temperature measurements from a single cropped .seq file.

Usage:
    python scripts/track_temperatures.py \\
        --input "/path/to/cropped_file_Front.seq" \\
        [--sampling-interval 10] \\
        [--mouse-roi-radius 8] \\
        [--overwrite]

--input is the only path you need to supply. Config and output directory
default to tracking_config.json and trackingOutputs/ under the project root
(src/paths.py) and can be overridden with --config / --output-dir if needed.

Output:
    trackingOutputs/
        <stem>_tracking_every10frames.csv
        qc_images/<stem>/frame_NNNNNN.png   (overlay previews)
        qc_summaries/<stem>_qc_summary.csv
        qc_plots/<stem>_diagnostic.png

Temperature values in the CSV are in degrees Celsius.
Calibration: T_celsius = raw_uint16 * 0.04 - 273.15 (confirmed from lab's seq_process pipeline).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.arena_mask import ArenaMask, TrackingConfig
from src.logging_utils import setup_logger
from src.qc_outputs import save_qc_summary, save_diagnostic_plots
from src.seq_io import SeqReader
from src.tracking import track_file


def main():
    parser = argparse.ArgumentParser(description="Track mouse temperatures in a cropped .seq file")
    parser.add_argument("--input", required=True, help="Path to cropped .seq file")
    parser.add_argument("--config", default=None,
                        help="Path to tracking_config.json (default: tracking_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: trackingOutputs/ under the project root)")
    parser.add_argument("--sampling-interval", type=int, default=None,
                        help="Override sampling interval from config (frames)")
    parser.add_argument("--mouse-roi-radius", type=int, default=None,
                        help="Override mouse ROI radius from config (pixels)")
    parser.add_argument("--floor-roi-radius", type=int, default=None,
                        help="Override floor ROI radius from config (pixels)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--no-qc-images", action="store_true",
                        help="Skip saving QC overlay images")
    parser.add_argument("--n-qc-images", type=int, default=20,
                        help="Number of QC images to save (default: 20)")
    parser.add_argument("--tracking-start-frame", type=int, default=None,
                        help=(
                            "Manually set the first frame to track, skipping auto-detection. "
                            "Useful when entry detection fails or the mouse is placed "
                            "at a known frame."
                        ))
    args = parser.parse_args()

    log = setup_logger()

    seq_path = Path(args.input)
    if not seq_path.exists():
        log.error(f"Input file not found: {seq_path}")
        sys.exit(1)

    # Load config
    config_path = args.config or str(paths.DEFAULT_TRACKING_CONFIG)
    try:
        config = TrackingConfig.load(config_path)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    # Apply CLI overrides
    if args.sampling_interval is not None:
        config.sampling_interval_frames = args.sampling_interval
    if args.mouse_roi_radius is not None:
        config.mouse_roi_radius_px = args.mouse_roi_radius
    if args.floor_roi_radius is not None:
        config.floor_roi_radius_px = args.floor_roi_radius
    if args.tracking_start_frame is not None:
        config.manual_tracking_start_frame = args.tracking_start_frame

    # Build arena mask from config
    with SeqReader(str(seq_path)) as reader:
        frame_shape = reader.frame_shape

    arena_mask = ArenaMask.from_config(config, frame_shape)

    if config.arena_polygon is None:
        log.warning(
            "No arena polygon defined in config — using full image minus border.\n"
            "For best results, run scripts/create_arena_mask.py first."
        )

    # Run tracking
    out_dir = Path(args.output_dir) if args.output_dir else paths.TRACKING_OUTPUTS_DIR
    try:
        df, csv_path, entry_info = track_file(
            seq_path=str(seq_path),
            config=config,
            arena_mask=arena_mask,
            output_dir=str(out_dir),
            overwrite=args.overwrite,
            save_qc_images=not args.no_qc_images,
            n_qc_images=args.n_qc_images,
        )
    except FileExistsError as e:
        log.error(str(e))
        sys.exit(1)

    # Save QC summary and diagnostic plot

    summary_dir = out_dir / "qc_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_qc_summary(
        df,
        str(summary_dir / f"{seq_path.stem}_qc_summary.csv"),
        entry_info=entry_info,
    )

    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostic_plots(df, str(plot_dir / seq_path.stem))

    log.info(f"\nDone. Output CSV: {csv_path}")


if __name__ == "__main__":
    main()
