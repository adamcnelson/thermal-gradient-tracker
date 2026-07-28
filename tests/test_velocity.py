"""Tests for src/velocity.py — centroid velocity computation."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis_config import BoutsConfig
from src.velocity import compute_velocity


def _make_df(n=30, add_invalid=True):
    """Synthetic tracking DataFrame."""
    t = np.arange(n) * (1 / 9.0)
    x = np.linspace(50, 200, n) + np.random.default_rng(0).normal(0, 1, n)
    y = np.full(n, 30.0)
    qc = ["ok"] * n
    valid = [True] * n

    if add_invalid:
        # Mark a few frames as invalid
        for i in [5, 6, 15]:
            qc[i] = "seg_error"
            valid[i] = False
            x[i] = np.nan
            y[i] = np.nan

    return pd.DataFrame({
        "frame_number": np.arange(n) * 10,
        "elapsed_time_sec": t,
        "mouse_centroid_x": x,
        "mouse_centroid_y": y,
        "qc_flag": qc,
        "mouse_roi_valid": valid,
    })


def test_velocity_columns_added():
    df = _make_df()
    config = BoutsConfig()
    out = compute_velocity(df, config)
    for col in ["centroid_x_roll_dispersion", "velocity_px_s", "velocity_x_px_s", "velocity_smooth_px_s"]:
        assert col in out.columns, f"Missing column: {col}"


def test_invalid_frames_produce_nan_dispersion():
    """Dispersion should be NaN at invalid frame positions when qc_valid_only=True."""
    df = _make_df(n=30, add_invalid=True)
    config = BoutsConfig(qc_valid_only_for_dispersion=True, dispersion_window_samples=3)
    out = compute_velocity(df, config)
    # Invalid frames at index 5, 6, 15 should have NaN dispersion
    for i in [5, 6, 15]:
        val = out["centroid_x_roll_dispersion"].iloc[i]
        assert np.isnan(val), f"Expected NaN dispersion at invalid frame {i}, got {val}"


def test_velocity_nan_across_gap():
    """Velocity across a gap (invalid frames) should be NaN or large, not zero."""
    df = _make_df(n=20, add_invalid=True)
    config = BoutsConfig(qc_valid_only_for_dispersion=True)
    out = compute_velocity(df, config)
    # The frame immediately after an invalid frame should have NaN velocity
    # (because valid centroid at index 5 is NaN → diff produces NaN)
    assert np.isnan(out["velocity_px_s"].iloc[5]) or np.isnan(out["velocity_px_s"].iloc[6])


def test_stationary_segment_low_dispersion():
    """A segment with truly stationary centroid should have low dispersion."""
    n = 50
    t = np.arange(n) / 9.0
    # All stationary at x=100
    x = np.full(n, 100.0) + np.random.default_rng(1).normal(0, 0.1, n)
    df = pd.DataFrame({
        "frame_number": np.arange(n) * 10,
        "elapsed_time_sec": t,
        "mouse_centroid_x": x,
        "mouse_centroid_y": np.full(n, 30.0),
        "qc_flag": ["ok"] * n,
        "mouse_roi_valid": [True] * n,
    })
    config = BoutsConfig(dispersion_window_samples=7)
    out = compute_velocity(df, config)
    # Mid-range dispersion should be very small
    mid_disp = out["centroid_x_roll_dispersion"].dropna().median()
    assert mid_disp < 1.0, f"Expected low dispersion for stationary segment, got {mid_disp:.3f}"


def test_dispersion_metrics():
    """All three dispersion metrics should run without error."""
    df = _make_df(n=30, add_invalid=False)
    for metric in ["mad", "iqr", "sd"]:
        config = BoutsConfig(dispersion_metric=metric, dispersion_window_samples=5)
        out = compute_velocity(df, config)
        assert "centroid_x_roll_dispersion" in out.columns
        assert out["centroid_x_roll_dispersion"].notna().any()
