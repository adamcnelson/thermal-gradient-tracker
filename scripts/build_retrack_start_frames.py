"""
Turn detect_entry_rgb.py's per-lane entry times into the start-frames CSV that
batch_track_temperatures.py --start-frames-csv reads (columns seq_stem,start_frame).

start_frame = ceil(thermal_entry_sec * camera_fps): the first thermal frame at or
after the mouse is alone in its lane. thermal_entry_sec is on the thermal clock
frame/10 (all 65 inherited lanes confirmed 10 fps), so --camera-fps must match
the re-tracking config (tracking_config_retrack_fps10.json).

Lanes without a detected entry are left out (their original tracking is kept).

--overrides (default entry_time_overrides.csv, committed): rows session,lane,rgb_entry_sec,note
replace a detected RGB entry that was checked by eye and found wrong; the thermal entry is
recomputed with that lane's sync, (rgb - offset) / (1 + drift).

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
    parser.add_argument("--overrides", default=str(REPO / "entry_time_overrides.csv"))
    args = parser.parse_args()

    e = pd.read_csv(args.entry_times)
    e["overridden"] = False
    if args.overrides and Path(args.overrides).exists():
        for o in pd.read_csv(args.overrides).itertuples():
            m = (e.session == o.session) & (e.lane == o.lane)
            if not m.any():
                raise ValueError(f"override for unknown lane {o.session} {o.lane}")
            e.loc[m, "rgb_entry_sec"] = o.rgb_entry_sec
            e.loc[m, "thermal_entry_sec"] = (o.rgb_entry_sec - e.loc[m, "sync_offset_sec"]) / (1 + e.loc[m, "sync_drift"])
            e.loc[m, "overridden"] = True
            print(f"override: {o.session} {o.lane} rgb_entry -> {o.rgb_entry_sec}s")
    found = e[e.thermal_entry_sec.notna()]
    out = pd.DataFrame({
        "seq_stem": [f"{r.session}_{LANE_NAMES[r.lane]}" for r in found.itertuples()],
        "start_frame": [max(0, math.ceil(t * args.camera_fps)) for t in found.thermal_entry_sec],
        "thermal_entry_sec": found.thermal_entry_sec.values,
        "rgb_entry_sec": found.rgb_entry_sec.values,
        "entry_via_other_lane": found.entry_via_other_lane.values,
        "overridden": found.overridden.values,
    })
    out.to_csv(args.output, index=False)
    skipped = e[e.thermal_entry_sec.isna()]
    print(f"{len(out)} lanes -> {args.output}; skipped (no entry): "
          f"{[f'{r.session} {r.lane}' for r in skipped.itertuples()] or 'none'}")


if __name__ == "__main__":
    main()
