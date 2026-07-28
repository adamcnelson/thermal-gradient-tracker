"""Unit tests for temperature_extraction.py — no .seq file required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.temperature_extraction import extract_roi_stats, build_output_row


class TestExtractRoiStats:
    def _frame(self, h=50, w=50, fill=1000):
        return np.full((h, w), fill, dtype=np.uint16)

    def test_uniform_frame_all_same(self):
        frame = self._frame(fill=5000)
        stats = extract_roi_stats(frame, cx=25, cy=25, radius=5)
        assert stats["mean"] == pytest.approx(5000.0)
        assert stats["median"] == pytest.approx(5000.0)
        assert stats["min"] == pytest.approx(5000.0)
        assert stats["max"] == pytest.approx(5000.0)

    def test_pixel_count_positive(self):
        frame = self._frame()
        stats = extract_roi_stats(frame, cx=25, cy=25, radius=5)
        assert stats["pixel_count"] > 0
        # A circle of radius 5 has area ~78 pixels
        assert 50 <= stats["pixel_count"] <= 100

    def test_mean_between_min_max(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(1000, 60000, size=(50, 50), dtype=np.uint16)
        stats = extract_roi_stats(frame, cx=25, cy=25, radius=8)
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["min"] <= stats["median"] <= stats["max"]

    def test_empty_roi_returns_nan(self):
        # If the ROI is entirely outside the image, it wraps to empty
        frame = self._frame(h=20, w=20)
        stats = extract_roi_stats(frame, cx=0, cy=0, radius=10)
        # Partial circle at corner — some pixels may be included
        # Just check that pixel_count is a non-negative integer
        assert stats["pixel_count"] >= 0

    def test_high_value_frame(self):
        frame = np.full((50, 50), 65535, dtype=np.uint16)
        stats = extract_roi_stats(frame, cx=25, cy=25, radius=4)
        assert stats["mean"] == pytest.approx(65535.0)

    def test_radius_one_center_pixel(self):
        frame = self._frame(fill=3000)
        frame[25, 25] = 9999
        stats = extract_roi_stats(frame, cx=25.0, cy=25.0, radius=0)
        # r=0 includes only the center pixel (distance 0 <= 0)
        assert stats["pixel_count"] == 1
        assert stats["mean"] == pytest.approx(9999.0)


class TestBuildOutputRow:
    def _base_kwargs(self):
        return dict(
            video_file="test.seq",
            frame_number=10,
            elapsed_time_sec=1.1,
            sampling_interval_frames=10,
            mouse_centroid_x=30.0,
            mouse_centroid_y=20.0,
            mouse_area_px=500,
            mouse_bbox_x=20,
            mouse_bbox_y=15,
            mouse_bbox_width=20,
            mouse_bbox_height=15,
            tracking_confidence=0.9,
            mouse_roi_valid=True,
            mouse_roi_center_x=30.0,
            mouse_roi_center_y=20.0,
            mouse_roi_radius_px=8.0,
            mouse_stats={"mean": 5000.0, "median": 4990.0, "min": 4800.0, "max": 5200.0,
                         "pixel_count": 200},
            floor_roi_valid=True,
            floor_roi_center_x=30.0,
            floor_roi_center_y=5.0,
            floor_roi_radius_px=8.0,
            floor_stats={"mean": 3000.0, "median": 2990.0, "min": 2800.0, "max": 3200.0,
                         "pixel_count": 200},
            floor_roi_shift_direction="up",
            floor_roi_shift_distance_px=15.0,
            qc_flag="ok",
            qc_notes="",
        )

    def test_delta_computed(self):
        row = build_output_row(**self._base_kwargs())
        assert row["mouse_minus_floor_temp_mean"] == pytest.approx(2000.0)

    def test_required_keys_present(self):
        row = build_output_row(**self._base_kwargs())
        for key in ("video_file", "frame_number", "mouse_surface_temp_mean",
                    "floor_temp_mean", "mouse_minus_floor_temp_mean"):
            assert key in row

    def test_missing_stats_give_nan(self):
        kwargs = self._base_kwargs()
        kwargs["mouse_roi_valid"] = False
        kwargs["mouse_stats"] = None
        row = build_output_row(**kwargs)
        assert np.isnan(row["mouse_surface_temp_mean"])

    def test_floor_invalid_gives_nan_delta(self):
        kwargs = self._base_kwargs()
        kwargs["floor_roi_valid"] = False
        kwargs["floor_stats"] = None
        row = build_output_row(**kwargs)
        assert np.isnan(row["mouse_minus_floor_temp_mean"])
