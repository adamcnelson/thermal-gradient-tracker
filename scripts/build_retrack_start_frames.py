"""
Turn detect_entry_rgb.py's per-lane entry times into the start-frames CSV that
batch_track_temperatures.py --start-frames-csv reads (columns seq_stem,start_frame).

start_frame = ceil(thermal_entry_sec * camera_fps): the first thermal frame at or
after the mouse is alone in its lane. thermal_entry_sec is on the thermal clock
frame/10 (all 65 inherited lanes confirmed 10 fps), so --camera-fps must match
the re-tracking config (tracking_config_retrack_fps10.json).

Lanes without a detected entry are left out (their original tracking is kept).

Usage:
    python scripts/build_retrack_start_frames.py \\
        [--entry-times thermalFeatures/entry_detection/entry_times.csv] \\
        [--output thermalFeatures/entry_detection/retrack_start_frames.csv]
"""

import argparse
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LANE_NAMES = {"F": "Front", "B": "Back"}


def main():
    parser = argparse.ArgumentParser(description="Build per-lane re-tracking start frames from entry times")
    parser.add_argument("--entry-times", default=str(REPO / "thermalFeatures/entry_detection/entry_times.csv"))
    parser.add_argument("--output", default=str(REPO / "thermalFeatures/entry_detection/retrack_start_frames.csv"))
    parser.add_argument("--camera-fps", type=float, default=10.0)
    args = parser.parse_args()

    e = pd.read_csv(args.entry_times)
    found = e[e.thermal_entry_sec.notna()]
    out = pd.DataFrame({
        "seq_stem": [f"{r.session}_{LANE_NAMES[r.lane]}" for r in found.itertuples()],
        "start_frame": [max(0, math.ceil(t * args.camera_fps)) for t in found.thermal_entry_sec],
        "thermal_entry_sec": found.thermal_entry_sec.values,
        "rgb_entry_sec": found.rgb_entry_sec.values,
        "entry_via_other_lane": found.entry_via_other_lane.values,
    })
    out.to_csv(args.output, index=False)
    skipped = e[e.thermal_entry_sec.isna()]
    print(f"{len(out)} lanes -> {args.output}; skipped (no entry): "
          f"{[f'{r.session} {r.lane}' for r in skipped.itertuples()] or 'none'}")


if __name__ == "__main__":
    main()
