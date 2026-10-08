"""
Join Stage 7's bout_output.csv / frame_output.csv files to the metadata LUT
(project_brief_v8.md §3.2).

Same LUT-matching logic as scripts/join_metadata.py (the legacy pipeline's
join), just keyed on the session+track columns Stage 7 already writes,
instead of parsing a video_file string -- see
src/metadata.py::join_landmark_metadata().

Usage:
    python scripts/join_landmark_metadata.py [--overwrite]
    python scripts/join_landmark_metadata.py --landmark-dir landmark_outputs \\
        thermalFeatures/stage7_inherited/landmark_outputs --overwrite   # all 71 lanes

--landmark-dir, --metadata, and --output-dir default to the project-root-
relative locations in src/paths.py (landmark_outputs/,
metadata/LUT_CLEAN_July6.csv, landmark_outputs/) and only need to be passed
to point at a non-default location.

Outputs (landmark_outputs/ by default):
    master_landmarks_with_metadata_bout.csv
    master_landmarks_with_metadata_frame.csv
    landmark_metadata_join_report_bout.csv
    landmark_metadata_join_report_frame.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.logging_utils import setup_logger
from src.metadata import filter_excluded, join_landmark_metadata, load_lut


def _join_one_kind(kind, pattern, in_dirs, lut_kept, out, overwrite, log):
    master_csv = out / f"master_landmarks_with_metadata_{kind}.csv"
    report_csv = out / f"landmark_metadata_join_report_{kind}.csv"

    for p in [master_csv, report_csv]:
        if p.exists() and not overwrite:
            log.error(f"{p} already exists. Use --overwrite to replace.")
            sys.exit(1)

    files = sorted(p for d in in_dirs for p in d.glob(pattern))
    if not files:
        log.warning(f"No {kind} files found in {[str(d) for d in in_dirs]} matching {pattern}")
        return
    names = [p.name for p in files]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:  # the same lane in two input dirs would be double-counted
        log.error(f"Same {kind} file in more than one --landmark-dir: {dupes}")
        sys.exit(1)

    log.info(f"\nJoining {len(files)} {kind}-level files to metadata...")
    master_df, report_df = join_landmark_metadata([str(p) for p in files], lut_kept)

    n_matched = (report_df["status"].str.startswith("matched")).sum() if "status" in report_df.columns else 0
    n_unmatched = (report_df["status"] == "unmatched").sum() if "status" in report_df.columns else 0
    n_ambiguous = (report_df["status"].str.contains("ambiguous", na=False)).sum() if "status" in report_df.columns else 0
    log.info(f"  Join coverage ({kind}): matched={n_matched} unmatched={n_unmatched} ambiguous={n_ambiguous}")

    if not master_df.empty:
        master_df.to_csv(str(master_csv), index=False)
        log.info(f"  {kind} master table: {len(master_df)} rows -> {master_csv}")
    else:
        log.warning(f"  {kind} master table is empty — no files matched metadata")

    report_df.to_csv(str(report_csv), index=False)
    log.info(f"  {kind} join report -> {report_csv}")


def main():
    parser = argparse.ArgumentParser(description="Join Stage 7 landmark outputs to metadata LUT")
    parser.add_argument("--landmark-dir", nargs="+", default=None,
                        help="One or more directories of bout_output.csv/frame_output.csv files, "
                             "combined into one master table (default: landmark_outputs/ under the "
                             "project root). E.g. landmark_outputs thermalFeatures/stage7_inherited/"
                             "landmark_outputs for Test_3/4/7 + the 35 inherited sessions.")
    parser.add_argument("--metadata", default=None,
                        help="Path to the metadata LUT (default: metadata/LUT_CLEAN_July6.csv under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: landmark_outputs/ under the project root)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger("join_landmark_metadata")

    in_dirs = [Path(d) for d in args.landmark_dir] if args.landmark_dir else [paths.LANDMARK_OUTPUTS_DIR]
    metadata_path = args.metadata or str(paths.DEFAULT_METADATA_LUT)
    out = Path(args.output_dir) if args.output_dir else paths.LANDMARK_OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    try:
        lut_df = load_lut(metadata_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)
    log.info(f"LUT loaded: {len(lut_df)} rows")

    lut_kept, lut_dropped = filter_excluded(lut_df)
    log.info(f"  Excluded {len(lut_dropped)} rows (single recording / no seq name)")
    log.info(f"  Kept {len(lut_kept)} rows for joining")

    _join_one_kind("bout", "*_bout_output.csv", in_dirs, lut_kept, out, args.overwrite, log)
    _join_one_kind("frame", "*_frame_output.csv", in_dirs, lut_kept, out, args.overwrite, log)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
