"""
Compute velocity and detect stationary bouts for a single tracking CSV.

Usage:
    python scripts/compute_bouts.py \\
        --input "trackingOutputs/07-28-25_..._Front_tracking_every10frames.csv" \\
        [--overwrite]

Config and output directory default to analysis_config.json and bouts/ under
the project root (src/paths.py) and can be overridden with --config /
--output-dir if needed.

Outputs (in --output-dir):
    qc_plots/<stem>_bouts_diagnostic.png / .pdf
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

PRELIMINARY_TAG = "PRELIMINARY — insufficient sample for inference"


def _get_thresholds(config, log):
    """Read thresholds from config; fall back to safe defaults if null."""
    disp = config.bouts.stationary_dispersion_threshold_px
    vel = config.bouts.stationary_velocity_threshold_px_per_s
    if disp is None:
        disp = 5.0
        log.warning("stationary_dispersion_threshold_px not set in config — using 5.0 px")
    if vel is None:
        vel = 5.0
        log.warning("stationary_velocity_threshold_px_per_s not set in config — using 5.0 px/s")
    return disp, vel


def run_compute_bouts(input_path: str, config: AnalysisConfig, output_dir: str,
                      overwrite: bool, log) -> dict:
    """Process a single tracking CSV. Returns per-file summary dict."""
    p = Path(input_path)
    out = Path(output_dir)

    qc_dir = out / "qc_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)
    stem = p.stem.replace("_tracking_every10frames", "").replace("_tracking", "")
    qc_path = qc_dir / f"{stem}_bouts_diagnostic.png"

    if qc_path.exists() and not overwrite:
        log.info(f"Skipping {p.name} — outputs exist (use --overwrite to regenerate)")
        return {}

    log.info(f"Loading {p.name}")
    try:
        df = pd.read_csv(str(p))
    except Exception as exc:
        log.error(f"Failed to read {p}: {exc}")
        return {}

    required = ["frame_number", "mouse_centroid_x", "elapsed_time_sec", "qc_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"{p.name}: missing required columns: {missing}")
        return {}

    # Velocity + dispersion
    log.info("  Computing velocity...")
    df = compute_velocity(df, config.bouts)

    # Thresholds
    disp_thresh, vel_thresh = _get_thresholds(config, log)

    # Classify + detect bouts
    log.info("  Detecting bouts...")
    stat_mask = classify_stationary(df, disp_thresh, vel_thresh)
    df["stationary"] = stat_mask

    bouts_df = rle_bouts(
        stat_mask,
        df["elapsed_time_sec"],
        config.bouts.min_bout_duration_sec,
        config.bouts.max_gap_merge_sec,
    )

    log.info(f"  Found {len(bouts_df)} stationary bouts")

    # QC plot
    save_bout_diagnostic(df, bouts_df, str(qc_dir / f"{stem}_bouts_diagnostic"), config.bouts, title_extra=p.name)
    log.info(f"  QC plot saved: {qc_path}")

    # Per-file summary
    vf = str(df["video_file"].iloc[0]) if "video_file" in df.columns else p.name
    meta = {"video_file": vf}
    summary = compute_bout_summary(
        df, bouts_df, metadata=meta,
        trial_length_reference_sec=config.analysis.trial_length_reference_sec,
    )

    # Save per-file bout rows as a sidecar CSV (batching script concatenates them)
    bout_metrics = compute_bout_metrics(df, bouts_df, metadata=meta)
    bout_sidecar = out / "qc_plots" / f"{stem}_bout_rows.csv"
    bout_metrics.to_csv(str(bout_sidecar), index=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute velocity + bouts for one tracking CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json (default: analysis_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: bouts/ under the project root)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger("compute_bouts")

    config_path = args.config or str(paths.DEFAULT_ANALYSIS_CONFIG)
    try:
        config = AnalysisConfig.load(config_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    output_dir = args.output_dir or str(paths.BOUTS_DIR)
    summary = run_compute_bouts(args.input, config, output_dir, args.overwrite, log)
    if summary:
        log.info(f"  Trial length: {summary.get('trial_length_sec', 'N/A'):.0f}s  "
                 f"  Bouts: {summary.get('n_stationary_bouts', 0)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
