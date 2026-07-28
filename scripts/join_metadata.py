"""
Join tracking CSVs to the experimental metadata LUT.

Usage:
    python scripts/join_metadata.py [--overwrite]

--tracking-dir, --metadata, --config, and --output-dir all default to the
project-root-relative locations in src/paths.py (trackingOutputs/,
metadata/LUT_CLEAN_July6.csv, analysis_config.json, bouts/) and only need to
be passed to point at a non-default location.

Outputs:
    bouts/master_tracking_with_metadata.csv
    bouts/metadata_join_report.csv
    bouts/metadata_excluded.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.analysis_config import AnalysisConfig
from src.bouts import classify_stationary
from src.logging_utils import setup_logger
from src.metadata import filter_excluded, join_metadata, load_lut
from src.velocity import compute_velocity


def main():
    parser = argparse.ArgumentParser(description="Join tracking data to metadata LUT")
    parser.add_argument("--tracking-dir", default=None,
                        help="Directory of tracking CSVs (default: trackingOutputs/ under the project root)")
    parser.add_argument("--bouts-dir", default=None,
                        help="Directory of bout outputs (default: bouts/ under the project root)")
    parser.add_argument("--metadata", default=None,
                        help="Path to the metadata LUT (default: metadata/LUT_CLEAN_July6.csv under the project root)")
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json (default: analysis_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: bouts/ under the project root)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger("join_metadata")

    tracking_dir = args.tracking_dir or str(paths.TRACKING_OUTPUTS_DIR)
    metadata_path = args.metadata or str(paths.DEFAULT_METADATA_LUT)
    config_path = args.config or str(paths.DEFAULT_ANALYSIS_CONFIG)

    out = Path(args.output_dir) if args.output_dir else paths.BOUTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    master_csv = out / "master_tracking_with_metadata.csv"
    report_csv = out / "metadata_join_report.csv"
    excluded_csv = out / "metadata_excluded.csv"

    for p in [master_csv, report_csv, excluded_csv]:
        if p.exists() and not args.overwrite:
            log.error(f"{p} already exists. Use --overwrite to replace.")
            sys.exit(1)

    # Load LUT
    try:
        lut_df = load_lut(metadata_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"LUT loaded: {len(lut_df)} rows")

    # Exclude single-recording rows
    lut_kept, lut_dropped = filter_excluded(lut_df)
    log.info(f"  Excluded {len(lut_dropped)} rows (single recording / no seq name)")
    log.info(f"  Kept {len(lut_kept)} rows for joining")

    if len(lut_dropped) > 0:
        lut_dropped.to_csv(str(excluded_csv), index=False)
        log.info(f"  Excluded rows written to {excluded_csv}")

    # Surface rows with non-standard NOTES (but don't drop them)
    if "NOTES" in lut_kept.columns:
        flagged = lut_kept[lut_kept["NOTES"].fillna("") != ""]
        if len(flagged) > 0:
            log.info(f"  {len(flagged)} rows have non-empty NOTES (not auto-dropped):")
            for _, row in flagged.iterrows():
                log.info(f"    Mouse_ID={row.get('Mouse_ID','')}  NOTES={row.get('NOTES','')}")

    # Find tracking CSVs
    in_dir = Path(tracking_dir)
    tracking_files = sorted(in_dir.glob("*_tracking*.csv"))
    if not tracking_files:
        log.warning(f"No tracking CSVs found in {in_dir}")
        sys.exit(0)

    log.info(f"\nJoining {len(tracking_files)} tracking files to metadata...")

    master_df, report_df = join_metadata([str(p) for p in tracking_files], lut_kept)

    # Join statistics
    n_matched = (report_df["status"].str.startswith("matched")).sum() if "status" in report_df.columns else 0
    n_unmatched = (report_df["status"] == "unmatched").sum() if "status" in report_df.columns else 0
    n_ambiguous = (report_df["status"].str.contains("ambiguous", na=False)).sum() if "status" in report_df.columns else 0

    log.info(f"\nJoin coverage:")
    log.info(f"  Matched:   {n_matched}")
    log.info(f"  Unmatched: {n_unmatched}")
    log.info(f"  Ambiguous: {n_ambiguous}")

    # Flag seq name warnings
    if "seq_name_warning" in report_df.columns:
        warned = report_df[report_df["seq_name_warning"].fillna("") != ""]
        if len(warned) > 0:
            log.info(f"\n  {len(warned)} files have seq-name reliability warnings:")
            for _, row in warned.iterrows():
                log.info(f"    {row.get('tracking_file','')}  — {row.get('seq_name_warning','')}")

    # Add velocity + stationary classification to the master table.
    # Processed per video_file so velocity doesn't bleed across recordings.
    if not master_df.empty:
        try:
            bout_config = AnalysisConfig.load(config_path).bouts
            disp_thresh = bout_config.stationary_dispersion_threshold_px or 5.0
            vel_thresh = bout_config.stationary_velocity_threshold_px_per_s or 5.0

            parts = []
            for vf, group in master_df.groupby("video_file", sort=False):
                group = compute_velocity(group.copy(), bout_config)
                group["stationary"] = classify_stationary(group, disp_thresh, vel_thresh)
                parts.append(group)
            master_df = pd.concat(parts, ignore_index=True)
            n_stat = master_df["stationary"].sum()
            log.info(f"  Stationary classification: {n_stat}/{len(master_df)} frames stationary")
        except Exception as exc:
            log.warning(f"Could not add stationary column: {exc}")

    # Write outputs
    if not master_df.empty:
        master_df.to_csv(str(master_csv), index=False)
        log.info(f"\nMaster table: {len(master_df)} rows → {master_csv}")
    else:
        log.warning("Master table is empty — no files matched metadata")

    report_df.to_csv(str(report_csv), index=False)
    log.info(f"Join report: {report_csv}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
