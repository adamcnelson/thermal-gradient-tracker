"""Smoke tests for src/landmarks/bout_qc.py — no real data required.

Matches this project's existing convention for plotting code (the legacy
src/bout_qc.py has no unit tests either): verify the function runs
end-to-end on synthetic data and produces real output files, rather than
asserting on rendered pixel content.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.bout_qc import save_v7_bout_diagnostic, _sync_confidence_alpha


def _synthetic_tracking_df(n=200):
    t = np.linspace(0, 400, n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "elapsed_time_sec": t,
        "mouse_centroid_x": 200 + 50 * np.sin(t / 30) + rng.normal(0, 5, n),
        "centroid_x_roll_dispersion": np.abs(rng.normal(2, 3, n)),
        "velocity_smooth_px_s": np.abs(rng.normal(3, 4, n)),
        "floor_temp_mean": 22 + 3 * np.sin(t / 100),
        "mouse_surface_temp_mean": 30 + rng.normal(0, 0.5, n),
    })


def _synthetic_bouts_df():
    return pd.DataFrame({
        "bout_index": [0, 1, 2],
        "bout_start_sec": [10.0, 150.0, 300.0],
        "bout_end_sec": [60.0, 200.0, 350.0],
    })


def _synthetic_rgb_track_df(n=100):
    t = np.linspace(0, 400, n)
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "rgb_time_sec": t,  # already thermal-clock-aligned, as save_v7_bout_diagnostic expects
        "rgb_centroid_x_thermal_aligned": 200 + 50 * np.sin(t / 30) + rng.normal(0, 5, n),
        "rgb_velocity_px_s": np.abs(rng.normal(3, 4, n)),
    })


def _synthetic_bout_output_df():
    return pd.DataFrame({
        "bout_id": [0, 1, 2],
        "bout_start_thermal_sec": [10.0, 150.0, 300.0],
        "bout_end_thermal_sec": [60.0, 200.0, 350.0],
        "warm_spot_temp_c": [33.0, np.nan, 34.5],  # bout 1: curled, no landmarks
        "dorsal_mean_c": [29.0, 27.5, 30.1],
        "dorsal_median_c": [28.8, 27.3, 29.9],
        "tail_delta_t_c": [2.1, np.nan, -0.5],
        "qc_valid": [True, False, False],
    })


class TestSaveV7BoutDiagnostic:
    def test_produces_png_and_pdf(self, tmp_path):
        out = tmp_path / "session_v7_bouts_diagnostic"
        save_v7_bout_diagnostic(
            _synthetic_tracking_df(), _synthetic_bouts_df(), _synthetic_bout_output_df(),
            str(out), title_extra="synthetic",
        )
        assert (tmp_path / "session_v7_bouts_diagnostic.png").exists()
        assert (tmp_path / "session_v7_bouts_diagnostic.pdf").exists()
        assert (tmp_path / "session_v7_bouts_diagnostic.png").stat().st_size > 0

    def test_handles_all_nan_measurement_column(self, tmp_path):
        bout_output = _synthetic_bout_output_df()
        bout_output["tail_delta_t_c"] = np.nan  # every bout failed tail measurement
        out = tmp_path / "all_nan"
        save_v7_bout_diagnostic(_synthetic_tracking_df(), _synthetic_bouts_df(), bout_output, str(out))
        assert (out.with_suffix(".png")).exists()

    def test_handles_missing_legacy_temp_columns(self, tmp_path):
        tracking_df = _synthetic_tracking_df().drop(columns=["floor_temp_mean", "mouse_surface_temp_mean"])
        out = tmp_path / "no_legacy_temp"
        save_v7_bout_diagnostic(tracking_df, _synthetic_bouts_df(), _synthetic_bout_output_df(), str(out))
        assert (out.with_suffix(".png")).exists()

    def test_handles_empty_bout_output(self, tmp_path):
        empty = _synthetic_bout_output_df().iloc[0:0]
        out = tmp_path / "empty_bouts"
        save_v7_bout_diagnostic(_synthetic_tracking_df(), _synthetic_bouts_df(), empty, str(out))
        assert (out.with_suffix(".png")).exists()

    def test_adds_rgb_track_panels_when_given(self, tmp_path):
        out = tmp_path / "with_rgb"
        save_v7_bout_diagnostic(
            _synthetic_tracking_df(), _synthetic_bouts_df(), _synthetic_bout_output_df(),
            str(out), rgb_track_df=_synthetic_rgb_track_df(),
        )
        assert (out.with_suffix(".png")).exists()

    def test_omits_rgb_track_panels_when_empty_df_given(self, tmp_path):
        out = tmp_path / "empty_rgb"
        empty_rgb = _synthetic_rgb_track_df().iloc[0:0]
        save_v7_bout_diagnostic(
            _synthetic_tracking_df(), _synthetic_bouts_df(), _synthetic_bout_output_df(),
            str(out), rgb_track_df=empty_rgb,
        )
        assert (out.with_suffix(".png")).exists()

    def test_rgb_track_panels_with_missing_legacy_temp_columns(self, tmp_path):
        tracking_df = _synthetic_tracking_df().drop(columns=["floor_temp_mean", "mouse_surface_temp_mean"])
        out = tmp_path / "rgb_no_legacy_temp"
        save_v7_bout_diagnostic(
            tracking_df, _synthetic_bouts_df(), _synthetic_bout_output_df(),
            str(out), rgb_track_df=_synthetic_rgb_track_df(),
        )
        assert (out.with_suffix(".png")).exists()

    def test_renders_with_sync_anchor_given(self, tmp_path):
        out = tmp_path / "with_anchor"
        save_v7_bout_diagnostic(
            _synthetic_tracking_df(), _synthetic_bouts_df(), _synthetic_bout_output_df(),
            str(out), rgb_track_df=_synthetic_rgb_track_df(), rgb_sync_anchor_thermal_sec=150.0,
        )
        assert (out.with_suffix(".png")).exists()


class TestSyncConfidenceAlpha:
    def test_full_opacity_at_anchor(self):
        rt = pd.Series([100.0])
        alpha = _sync_confidence_alpha(rt, anchor_sec=100.0)
        assert alpha[0] == pytest.approx(1.0, abs=1e-6)

    def test_decays_toward_floor_far_from_anchor(self):
        rt = pd.Series([100.0, 100000.0])
        alpha = _sync_confidence_alpha(rt, anchor_sec=100.0, min_alpha=0.06)
        assert alpha[0] > alpha[1]
        assert alpha[1] == pytest.approx(0.06, abs=1e-3)

    def test_no_anchor_gives_flat_midpoint_alpha(self):
        rt = pd.Series([0.0, 500.0, 1000.0])
        alpha = _sync_confidence_alpha(rt, anchor_sec=None)
        assert np.all(alpha == 0.5)

    def test_nan_time_gets_floor_alpha(self):
        rt = pd.Series([np.nan])
        alpha = _sync_confidence_alpha(rt, anchor_sec=100.0, min_alpha=0.06)
        assert alpha[0] == pytest.approx(0.06, abs=1e-6)
