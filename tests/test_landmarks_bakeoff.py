"""Tests for src/landmarks/bakeoff.py — no real data required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.bakeoff import (
    compute_landmark_pixel_error,
    compute_thermal_value_error,
    pixel_distance,
    polygon_mask_xy,
)


class TestPixelDistance:
    def test_computes_euclidean_distance(self):
        assert pixel_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_none_a_returns_none(self):
        assert pixel_distance(None, (1.0, 1.0)) is None

    def test_none_b_returns_none(self):
        assert pixel_distance((1.0, 1.0), None) is None

    def test_identical_points_zero_distance(self):
        assert pixel_distance((5.0, 5.0), (5.0, 5.0)) == 0.0


class TestComputeLandmarkPixelError:
    def test_converts_algo_row_col_to_xy_before_comparing(self):
        # label nose_point is [x=100, y=50]; algo nose_point is (row=50, col=100)
        # -- same real point, opposite axis order -- error should be ~0.
        result = compute_landmark_pixel_error(
            "test_label", "extended",
            label_head_point_xy=None,
            label_nose_point_xy=[100.0, 50.0],
            label_tail_base_point_xy=None,
            algo_nose_point=(50, 100),  # (row, col)
            algo_tail_base_point=None,
        )
        assert result.nose_error_px == pytest.approx(0.0, abs=1e-6)

    def test_real_offset_measured_correctly(self):
        result = compute_landmark_pixel_error(
            "test_label", "extended",
            label_head_point_xy=None,
            label_nose_point_xy=[100.0, 50.0],
            label_tail_base_point_xy=[500.0, 80.0],
            algo_nose_point=(53, 104),  # (row=53,col=104) vs label (x=100,y=50) -> dx=4,dy=3 -> 5px
            algo_tail_base_point=(80, 500),  # exact match
        )
        assert result.nose_error_px == pytest.approx(5.0)
        assert result.tail_base_error_px == pytest.approx(0.0, abs=1e-6)

    def test_missing_algo_landmarks_returns_none_with_reason(self):
        result = compute_landmark_pixel_error(
            "curled_label", "curled",
            label_head_point_xy=None,
            label_nose_point_xy=[100.0, 50.0],
            label_tail_base_point_xy=[500.0, 80.0],
            algo_nose_point=None,
            algo_tail_base_point=None,
        )
        assert result.nose_error_px is None
        assert result.tail_base_error_px is None
        assert "curled" in result.reason

    def test_missing_label_point_returns_none_for_that_landmark_only(self):
        result = compute_landmark_pixel_error(
            "partial_label", "extended",
            label_head_point_xy=None,
            label_nose_point_xy=None,
            label_tail_base_point_xy=[500.0, 80.0],
            algo_nose_point=(53, 104),
            algo_tail_base_point=(80, 500),
        )
        assert result.nose_error_px is None
        assert result.tail_base_error_px == pytest.approx(0.0, abs=1e-6)

    def test_head_point_used_as_fallback_with_caveat_when_no_nose_label(self):
        result = compute_landmark_pixel_error(
            "old_label", "extended",
            label_head_point_xy=[100.0, 50.0],
            label_nose_point_xy=None,
            label_tail_base_point_xy=None,
            algo_nose_point=(50, 100),
            algo_tail_base_point=None,
        )
        assert result.nose_error_px is None
        assert result.head_error_px == pytest.approx(0.0, abs=1e-6)
        assert "predates" in result.caveat

    def test_no_caveat_when_real_nose_label_present(self):
        result = compute_landmark_pixel_error(
            "new_label", "extended",
            label_head_point_xy=[90.0, 50.0],
            label_nose_point_xy=[100.0, 50.0],
            label_tail_base_point_xy=None,
            algo_nose_point=(50, 100),
            algo_tail_base_point=None,
        )
        assert result.nose_error_px == pytest.approx(0.0, abs=1e-6)
        assert result.head_error_px == pytest.approx(10.0)
        assert result.caveat == ""


class TestPolygonMaskXy:
    def test_rasterizes_square_polygon(self):
        # square from (2,2) to (5,5) in (x,y) -- expect roughly a 3x3 filled region
        polygon = [[2, 2], [5, 2], [5, 5], [2, 5]]
        mask = polygon_mask_xy(polygon, shape=(10, 10))
        assert mask.shape == (10, 10)
        assert mask[3, 3]  # (row=3,col=3) inside the square
        assert not mask[8, 8]  # well outside

    def test_empty_polygon_list_produces_empty_mask(self):
        mask = polygon_mask_xy([], shape=(10, 10))
        assert not np.any(mask)


class TestComputeThermalValueError:
    def _thermal_frame(self):
        # 20x20 frame, gradient background ~20C, a real "warm spot" hot region ~35C at (5-9, 5-9) (x,y)
        frame = np.full((20, 20), 20.0)
        frame[5:9, 5:9] = 35.0  # note: numpy indexing is [row, col] -- see note below
        return frame

    def test_computes_human_roi_stats_and_error(self):
        thermal = self._thermal_frame()
        # ROI drawn around x=5-8, y=5-8 (matches the hot region, since frame[5:9,5:9]
        # sets rows 5-8 and cols 5-8 to 35 -- a square region so x/y ambiguity doesn't matter here)
        roi = [[5, 5], [8, 5], [8, 8], [5, 8]]
        result = compute_thermal_value_error(
            "test_label", "extended", thermal, roi, algo_warm_spot_c=36.0,
        )
        assert result.human_roi_mean_c == pytest.approx(35.0, abs=0.5)
        assert result.human_roi_p95_c == pytest.approx(35.0, abs=0.5)
        assert result.error_c == pytest.approx(36.0 - result.human_roi_p95_c, abs=1e-6)
        assert result.reason == ""

    def test_no_roi_polygon_returns_none_with_reason(self):
        thermal = self._thermal_frame()
        result = compute_thermal_value_error("test_label", "curled", thermal, None, algo_warm_spot_c=None)
        assert result.human_roi_mean_c is None
        assert result.error_c is None
        assert "no human warm-spot ROI" in result.reason

    def test_no_algo_warm_spot_gives_none_error_but_real_human_stats(self):
        thermal = self._thermal_frame()
        roi = [[5, 5], [8, 5], [8, 8], [5, 8]]
        result = compute_thermal_value_error("test_label", "curled", thermal, roi, algo_warm_spot_c=None)
        assert result.human_roi_mean_c is not None
        assert result.error_c is None
        assert "no algorithmic warm-spot" in result.reason

    def test_too_few_polygon_points_treated_as_no_roi(self):
        thermal = self._thermal_frame()
        result = compute_thermal_value_error("test_label", "extended", thermal, [[1, 1], [2, 2]], algo_warm_spot_c=30.0)
        assert result.human_roi_mean_c is None
        assert "no human warm-spot ROI" in result.reason
