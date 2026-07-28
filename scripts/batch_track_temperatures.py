"""
Batch-process all cropped .seq files in a directory.

Usage:
    # All .seq files in one folder
    python scripts/batch_track_temperatures.py --input-dir "/path/to/croppedSeqFiles"

    # Recurse into subfolders
    python scripts/batch_track_temperatures.py --input-dir "/path/to/croppedSeqFiles" --recursive

--input-dir is the only path you need to supply. Config and output directory
default to tracking_config.json and trackingOutputs/ under the project root
(src/paths.py) and can be overridden with --config / --output-dir if needed.

Existing output CSV files are skipped unless --overwrite is passed.

Output:
    trackingOutputs/
        <stem>_tracking_every10frames.csv   (one per input file)
        qc_images/<stem>/                   (overlay previews)
        qc_summaries/<stem>_qc_summary.csv
        qc_plots/<stem>_diagnostic.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.arena_mask import TrackingConfig
from src.batch import find_seq_files, process_batch
from src.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Batch-track temperatures in all cropped .seq files"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing cropped .seq files (the one path you supply — "
                             "everything else is created automatically under the project root)")
    parser.add_argument("--config", default=None,
                        help="Path to tracking_config.json (default: tracking_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: trackingOutputs/ under the project root)")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for .seq files recursively in subdirectories")
    parser.add_argument("--sampling-interval", type=int, default=None,
                        help="Override sampling interval from config")
    parser.add_argument("--mouse-roi-radius", type=int, default=None,
                        help="Override mouse ROI radius from config")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--no-qc-images", action="store_true",
                        help="Skip QC overlay images")
    parser.add_argument("--n-qc-images", type=int, default=20,
                        help="QC images per file (default: 20)")
    args = parser.parse_args()

    log = setup_logger()

    config_path = args.config or str(paths.DEFAULT_TRACKING_CONFIG)
    try:
        config = TrackingConfig.load(config_path)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    if args.sampling_interval is not None:
        config.sampling_interval_frames = args.sampling_interval
    if args.mouse_roi_radius is not None:
        config.mouse_roi_radius_px = args.mouse_roi_radius

    try:
        seq_files = find_seq_files(args.input_dir, recursive=args.recursive)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    if not seq_files:
        log.error(f"No .seq files found in {args.input_dir}")
        sys.exit(1)

    log.info(f"Found {len(seq_files)} .seq files")
    for f in seq_files:
        log.info(f"  {f.name}")
    log.info("")

    if config.arena_polygon is None:
        log.warning(
            "No arena polygon defined in config — using full image minus border for all files.\n"
            "For best results, run scripts/create_arena_mask.py first."
        )

    output_dir = args.output_dir or str(paths.TRACKING_OUTPUTS_DIR)
    results = process_batch(
        seq_files=seq_files,
        config=config,
        output_dir=output_dir,
        overwrite=args.overwrite,
        save_qc_images=not args.no_qc_images,
        n_qc_images=args.n_qc_images,
        save_qc_summary=True,
        save_plots=True,
    )

    # Print final table
    log.info("\n--- Batch results ---")
    for r in results:
        status = r["status"].upper()
        name = Path(r["seq_path"]).name
        err = f"  [{r['error']}]" if r["error"] else ""
        log.info(f"  {status:<8} {name}{err}")


if __name__ == "__main__":
    main()
