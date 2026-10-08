"""Tests for src/velocity.py::attach_tracking_velocity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.velocity import attach_tracking_velocity


def _tracking():
    rows = []
    for vf, base in (("S1_Front.seq", 0.0), ("S1_Back.seq", 100.0)):
        for t in range(10):
            rows.append(dict(video_file=vf, elapsed_time_sec=float(t), velocity_smooth_px_s=base + t))
    return pd.DataFrame(rows)


def test_nearest_within_tolerance_per_lane_and_order_preserved():
    frames = pd.DataFrame(dict(session=["S1", "S1", "S1", "S1"], track=["B", "F", "F", "F"],
                               elapsed_time_thermal_sec=[3.0, 4.25, 2.0, 7.75]))
    out = attach_tracking_velocity(frames, _tracking())
    assert list(out.elapsed_time_thermal_sec) == [3.0, 4.25, 2.0, 7.75]      # row order kept
    assert list(out.velocity_smooth_px_s) == [103.0, 4.0, 2.0, 8.0]           # lane-specific, nearest


def test_outside_tolerance_or_unknown_lane_is_nan():
    frames = pd.DataFrame(dict(session=["S1", "S2"], track=["F", "F"], elapsed_time_thermal_sec=[20.0, 3.0]))
    out = attach_tracking_velocity(frames, _tracking())
    assert out.velocity_smooth_px_s.isna().all()
    assert len(out) == 2
