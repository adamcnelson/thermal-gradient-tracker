"""
Collapse estimate_sync_trajectory.py's per-lane results into one RGB<->thermal
sync per session (src.landmarks.sync.consolidate_session_sync), and emit each
as a `sync_result` block in stage7_sessions_config.json's format.

r_squared / residual_max_sec are NaN: there is no anchor fit behind a trajectory
estimate, and inventing statistics would be worse (same precedent as
WindowedSyncResult.manual_low_confidence). Consequence: Stage 7's QC report shows
sync_passes=False for these sessions -- that means "not anchor-fitted", not
"failed"; the confidence_note says so. The extra keys (method, lanes, ...) are
ignored by stage7_real_run.load_sessions() and kept for traceability.

Usage:
    python scripts/consolidate_sync_estimates.py \\
        [--input thermalFeatures/trajectory_sync/sync_estimates.csv] \\
        [--output-dir thermalFeatures/trajectory_sync]

Outputs: <output-dir>/session_sync.csv, <output-dir>/session_sync_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.landmarks.sync import consolidate_session_sync

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "thermalFeatures" / "trajectory_sync"


def sync_result_block(row, source_csv: str) -> dict:
    note = (f"trajectory alignment (scripts/estimate_sync_trajectory.py, {source_csv}); "
            f"mean of {row.n_clean_lanes or row.n_lanes} lane(s): {row.lanes}. "
            "No anchor fit, so r_squared/residual_max_sec are NaN and sync_passes reads False. "
            "Method validated on Test_3/4/7 (6 lanes within 1s of human anchors).")
    if row.low_confidence:
        note += f" LOW CONFIDENCE: {row.low_confidence_reasons}."
    if row.review_reasons:
        note += f" Needs review: {row.review_reasons}."
    return {
        "offset_sec": row.offset_sec,
        "drift_slope": row.drift_slope,
        "r_squared": float("nan"),
        "residual_max_sec": float("nan"),
        "low_confidence": bool(row.low_confidence),
        "confidence_note": note,
        "method": "trajectory_alignment",
        "camera_fps": row.camera_fps,
        "lane_spread_sec": None if pd.isna(row.lane_spread_sec) else round(row.lane_spread_sec, 3),
        "review_reasons": row.review_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(DEFAULT_DIR / "sync_estimates.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    sessions = consolidate_session_sync(pd.read_csv(args.input))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(out_dir / "session_sync.csv", index=False)
    blocks = {r.session: sync_result_block(r, Path(args.input).name) for r in sessions.itertuples()}
    with open(out_dir / "session_sync_results.json", "w") as f:
        json.dump(blocks, f, indent=2)

    print(sessions[["session", "offset_sec", "drift_slope", "n_lanes", "n_clean_lanes", "lane_spread_sec",
                    "low_confidence_reasons", "review_reasons"]].to_string(index=False))
    print(f"\n{len(sessions)} sessions; {int(sessions.low_confidence.sum())} low-confidence; "
          f"{int((sessions.review_reasons != '').sum())} need review -> {out_dir}")


if __name__ == "__main__":
    main()
