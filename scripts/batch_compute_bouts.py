"""
Batch-compute velocity + bouts for all tracking CSVs in a directory.

Usage:
    python scripts/batch_compute_bouts.py [--recursive] [--overwrite]

--input-dir defaults to trackingOutputs/ under the project root, since that's
where batch_track_temperatures.py writes tracking CSVs; override it only if
pointing at a non-default location. Config and output directory default to
analysis_config.json and bouts/ under the project root (src/paths.py).

Outputs:
    bouts/bout_table.csv             — all per-bout rows across all files
    bouts/bout_summary_per_file.csv  — per-file statistics
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.analysis_config import AnalysisConfig
from src.bouts import classify_stationary, compute_bout_metrics, compute_bout_summary, rle_bouts
from src.bout_qc import save_bout_diagnostic
from src.logging_utils import setup_logger
from src.velocity import compute_velocity
from scripts.compute_bouts import _get_thresholds


def main():
    parser = argparse.ArgumentParser(description="Batch compute bouts for all tracking CSVs")
    parser.add_argument("--input-dir", default=None,
                        help="Directory of tracking CSVs (default: trackingOutputs/ under the project root)")
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json (default: analysis_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: bouts/ under the project root)")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger("batch_compute_bouts")

    config_path = args.config or str(paths.DEFAULT_ANALYSIS_CONFIG)
    try:
        config = AnalysisConfig.load(config_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    in_dir = Path(args.input_dir) if args.input_dir else paths.TRACKING_OUTPUTS_DIR
    pattern = "**/*_tracking*.csv" if args.recursive else "*_tracking*.csv"
    csv_files = sorted(in_dir.glob(pattern))
    if not csv_files:
        log.warning(f"No tracking CSVs found in {in_dir}")
        sys.exit(0)

    log.info(f"Found {len(csv_files)} tracking CSVs")

    out = Path(args.output_dir) if args.output_dir else paths.BOUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    qc_dir = out / "qc_plots"
    qc_dir.mkdir(exist_ok=True)

    all_bouts = []
    all_summaries = []
    n_ok = 0
    n_fail = 0

    disp_thresh, vel_thresh = _get_thresholds(config, log)

    for csv_path in csv_files:
        try:
            log.info(f"Processing {csv_path.name}")
            df = pd.read_csv(str(csv_path))

            required = ["frame_number", "mouse_centroid_x", "elapsed_time_sec", "qc_flag"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                log.error(f"  {csv_path.name}: missing columns {missing} — skipping")
                n_fail += 1
                continue

            df = compute_velocity(df, config.bouts)
            stat_mask = classify_stationary(df, disp_thresh, vel_thresh)
            df["stationary"] = stat_mask

            bouts_df = rle_bouts(
                stat_mask,
                df["elapsed_time_sec"],
                config.bouts.min_bout_duration_sec,
                config.bouts.max_gap_merge_sec,
            )

            stem = csv_path.stem.replace("_tracking_every10frames", "").replace("_tracking", "")
            qc_path = qc_dir / f"{stem}_bouts_diagnostic"
            if not Path(str(qc_path) + ".png").exists() or args.overwrite:
                save_bout_diagnostic(df, bouts_df, str(qc_path), config.bouts)

            vf = str(df["video_file"].iloc[0]) if "video_file" in df.columns else csv_path.name
            meta = {"video_file": vf}

            bout_metrics = compute_bout_metrics(df, bouts_df, metadata=meta)
            all_bouts.append(bout_metrics)

            summary = compute_bout_summary(
                df, bouts_df, metadata=meta,
                trial_length_reference_sec=config.analysis.trial_length_reference_sec,
            )
            all_summaries.append(summary)

            log.info(f"  {len(bouts_df)} bouts detected")
            n_ok += 1

        except Exception as exc:
            log.error(f"  {csv_path.name}: failed — {exc}")
            n_fail += 1

    # Write combined tables
    if all_bouts:
        bout_table = pd.concat(all_bouts, ignore_index=True)
        bout_csv = out / "bout_table.csv"
        if bout_csv.exists() and not args.overwrite:
            log.warning(f"{bout_csv} exists; use --overwrite to replace")
        else:
            bout_table.to_csv(str(bout_csv), index=False)
            log.info(f"Wrote {bout_csv} ({len(bout_table)} bouts)")

    if all_summaries:
        summary_table = pd.DataFrame(all_summaries)
        summary_csv = out / "bout_summary_per_file.csv"
        if summary_csv.exists() and not args.overwrite:
            log.warning(f"{summary_csv} exists; use --overwrite to replace")
        else:
            summary_table.to_csv(str(summary_csv), index=False)
            log.info(f"Wrote {summary_csv} ({len(summary_table)} files)")

    log.info(f"\nBatch complete: {n_ok} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
