"""Tests for src/landmarks/outputs.py — no real data required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.outputs import (
    GRADIENT_ZONE_COOL,
    GRADIENT_ZONE_HOT,
    GRADIENT_ZONE_MID,
    build_bout_output_row,
    build_session_qc_report,
    bout_rows_to_dataframe,
    classify_gradient_zone,
    compute_landmark_yield_by_zone,
    rgb_time_to_thermal_time,
    thermal_time_to_rgb_time,
)
from src.landmarks.registration import HomographyFit, NudgeEvent
from src.landmarks.sync import WindowedSyncResult
from src.landmarks.thermal_measurement import DorsalSurfaceMeasurement, TailMeasurement


class TestClassifyGradientZone:
    def test_hot(self):
        assert classify_gradient_zone(40.0) == GRADIENT_ZONE_HOT

    def test_cool(self):
        assert classify_gradient_zone(15.0) == GRADIENT_ZONE_COOL

    def test_mid(self):
        assert classify_gradient_zone(26.0) == GRADIENT_ZONE_MID

    def test_boundary_values_are_inclusive_toward_the_extreme(self):
        assert classify_gradient_zone(33.0) == GRADIENT_ZONE_HOT
        assert classify_gradient_zone(21.0) == GRADIENT_ZONE_COOL

    def test_none_returns_none(self):
        assert classify_gradient_zone(None) is None

    def test_nan_returns_none(self):
        assert classify_gradient_zone(float("nan")) is None

    def test_custom_thresholds(self):
        assert classify_gradient_zone(25.0, cool_threshold_c=10.0, hot_threshold_c=20.0) == GRADIENT_ZONE_HOT


class TestTimeConversion:
    def test_round_trip(self):
        sync = WindowedSyncResult(
            window_centers_sec=np.array([0.0]),
            window_lags_sec=np.array([0.0]),
            offset_sec=1.5,
            drift_slope=0.002,
            r_squared=0.99,
            residual_max_sec=0.05,
        )
        thermal_t = 300.0
        rgb_t = thermal_time_to_rgb_time(thermal_t, sync)
        back = rgb_time_to_thermal_time(rgb_t, sync)
        assert back == pytest.approx(thermal_t, abs=1e-9)

    def test_zero_offset_zero_drift_is_identity(self):
        sync = WindowedSyncResult(
            window_centers_sec=np.array([0.0]),
            window_lags_sec=np.array([0.0]),
            offset_sec=0.0,
            drift_slope=0.0,
            r_squared=1.0,
            residual_max_sec=0.0,
        )
        assert thermal_time_to_rgb_time(100.0, sync) == pytest.approx(100.0)


class TestBuildBoutOutputRow:
    def _sync(self):
        return WindowedSyncResult(
            window_centers_sec=np.array([0.0]),
            window_lags_sec=np.array([0.0]),
            offset_sec=2.0,
            drift_slope=0.0,
            r_squared=1.0,
            residual_max_sec=0.01,
        )

    def test_assembles_all_fields(self):
        dorsal = DorsalSurfaceMeasurement(mean_c=30.0, median_c=29.5, pixel_count=500, valid=True)
        tail = TailMeasurement(tail_temp_c=28.0, floor_temp_c=26.0, delta_t_c=2.0, valid=True)

        row = build_bout_output_row(
            session="07-28-25_4540_B_4541_F_Test3-004",
            track="F",
            mouse_id=4541,
            bout_id=15,
            bout_start_thermal_sec=1360.0,
            bout_end_thermal_sec=1540.0,
            sync_result=self._sync(),
            mean_floor_temp_c=26.0,
            warm_spot_temp_c=32.0,
            dorsal=dorsal,
            tail=tail,
            n_frames_averaged=180,
            qc_valid=True,
        )
        assert row.session == "07-28-25_4540_B_4541_F_Test3-004"
        assert row.bout_start_rgb_sec == pytest.approx(1362.0)
        assert row.bout_end_rgb_sec == pytest.approx(1542.0)
        assert row.gradient_zone == GRADIENT_ZONE_MID
        assert row.dorsal_mean_c == 30.0
        assert row.tail_delta_t_c == 2.0
        assert row.qc_valid

    def test_low_confidence_sync_flag_carries_through_without_blocking_qc_valid(self):
        low_conf_sync = WindowedSyncResult.manual_low_confidence(
            offset_sec=11.0, drift_slope=0.0, note="Test_4: no method converged past ~1 anchor"
        )
        dorsal = DorsalSurfaceMeasurement(mean_c=27.0, median_c=26.8, pixel_count=400, valid=True)
        row = build_bout_output_row(
            session="07-30-25_4540_F_4541_B_Test4-008", track="F", mouse_id=4540, bout_id=9,
            bout_start_thermal_sec=100.0, bout_end_thermal_sec=140.0,
            sync_result=low_conf_sync, mean_floor_temp_c=27.0,
            warm_spot_temp_c=None, dorsal=dorsal, tail=None,
            n_frames_averaged=2, qc_valid=True, qc_reasons=[],
        )
        assert row.sync_low_confidence is True
        # a low-confidence sync does not, by itself, invalidate the row --
        # it's a separate advisory column, not folded into qc_valid/qc_reasons
        assert row.qc_valid is True
        assert row.qc_reasons == []

    def test_no_sync_result_defaults_low_confidence_false(self):
        row = build_bout_output_row(
            session="s", track="F", mouse_id=1, bout_id=0,
            bout_start_thermal_sec=0.0, bout_end_thermal_sec=10.0,
            sync_result=None, mean_floor_temp_c=None,
            warm_spot_temp_c=None, dorsal=None, tail=None,
            n_frames_averaged=0, qc_valid=False, qc_reasons=[],
        )
        assert row.sync_low_confidence is False

    def test_high_confidence_sync_flag_is_false(self):
        row = build_bout_output_row(
            session="s", track="F", mouse_id=1, bout_id=0,
            bout_start_thermal_sec=0.0, bout_end_thermal_sec=10.0,
            sync_result=self._sync(), mean_floor_temp_c=None,
            warm_spot_temp_c=None, dorsal=None, tail=None,
            n_frames_averaged=0, qc_valid=True, qc_reasons=[],
        )
        assert row.sync_low_confidence is False

    def test_missing_sync_leaves_rgb_times_none(self):
        row = build_bout_output_row(
            session="s", track="F", mouse_id=1, bout_id=0,
            bout_start_thermal_sec=0.0, bout_end_thermal_sec=10.0,
            sync_result=None, mean_floor_temp_c=None,
            warm_spot_temp_c=None, dorsal=None, tail=None,
            n_frames_averaged=0, qc_valid=False, qc_reasons=["no sync"],
        )
        assert row.bout_start_rgb_sec is None
        assert row.gradient_zone is None
        assert row.dorsal_mean_c is None
        assert row.qc_reasons == ["no sync"]

    def test_bout_rows_to_dataframe(self):
        row = build_bout_output_row(
            session="s", track="B", mouse_id=1, bout_id=0,
            bout_start_thermal_sec=0.0, bout_end_thermal_sec=10.0,
            sync_result=None, mean_floor_temp_c=15.0,
            warm_spot_temp_c=None, dorsal=None, tail=None,
            n_frames_averaged=5, qc_valid=True, qc_reasons=[],
        )
        df = bout_rows_to_dataframe([row])
        assert len(df) == 1
        assert df.iloc[0]["gradient_zone"] == GRADIENT_ZONE_COOL
        assert df.iloc[0]["qc_reasons"] == ""


class TestBuildSessionQCReport:
    def test_assembles_from_upstream_results(self):
        hfit = HomographyFit(H=np.eye(3), rmse_px=0.5, residuals_px=np.array([0.4, 0.6]))
        sync = WindowedSyncResult(
            window_centers_sec=np.array([0.0, 1.0]),
            window_lags_sec=np.array([0.0, 0.001]),
            offset_sec=0.1,
            drift_slope=0.0001,
            r_squared=0.98,
            residual_max_sec=0.02,
        )
        nudges = [NudgeEvent(time_sec=500.0, displacement_px=12.0)]

        report = build_session_qc_report(
            session="s", track="F",
            homography_fit=hfit, homography_max_rmse_px=1.0,
            sync_result=sync, sync_max_residual_sec=0.125,
            nudge_events=nudges,
            landmark_yield_by_zone={"cool": 0.9, "mid": 0.8, "hot": 0.3},
            fallback_invocation_count=3,
            rejected_detection_count=7,
        )
        assert report.homography_rmse_px == 0.5
        assert report.homography_passes
        assert report.sync_passes
        assert len(report.nudge_events) == 1
        assert report.landmark_yield_by_zone["hot"] == 0.3
        assert report.fallback_invocation_count == 3
        assert report.rejected_detection_count == 7

    def test_missing_upstream_results_leave_fields_none(self):
        report = build_session_qc_report(
            session="s", track="F",
            homography_fit=None, homography_max_rmse_px=1.0,
            sync_result=None, sync_max_residual_sec=0.125,
        )
        assert report.homography_rmse_px is None
        assert report.homography_passes is None
        assert report.sync_passes is None
        assert report.nudge_events == []
        assert report.sync_low_confidence is None

    def test_low_confidence_sync_flag_and_note_carry_through(self):
        low_conf_sync = WindowedSyncResult.manual_low_confidence(
            offset_sec=11.0, note="Test_4: only 1 real position-anchor transition found"
        )
        report = build_session_qc_report(
            session="07-30-25_4540_F_4541_B_Test4-008", track="F",
            homography_fit=None, homography_max_rmse_px=1.0,
            sync_result=low_conf_sync,
        )
        assert report.sync_low_confidence is True
        assert "Test_4" in report.sync_confidence_note
        # no real fit exists behind a manually-adopted offset -> can't pass
        assert report.sync_passes is False


class TestComputeLandmarkYieldByZone:
    def test_computes_fraction_valid_per_zone(self):
        zones = ["hot", "hot", "mid", "mid", "cool"]
        valid = [True, False, True, True, False]
        result = compute_landmark_yield_by_zone(zones, valid)
        assert result["hot"] == pytest.approx(0.5)
        assert result["mid"] == pytest.approx(1.0)
        assert result["cool"] == pytest.approx(0.0)

    def test_none_zones_excluded(self):
        result = compute_landmark_yield_by_zone(["hot", None], [True, True])
        assert "hot" in result
        assert len(result) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_landmark_yield_by_zone(["hot"], [True, False])

    def test_empty_returns_empty_dict(self):
        assert compute_landmark_yield_by_zone([], []) == {}
