"""Tests for src/landmarks/thermal_measurement.py — no real data required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.registration import apply_homography
from src.landmarks.rgb_landmarks import extract_landmarks_from_mask
from src.landmarks.thermal_measurement import (
    GatingThresholds,
    anterior_region_mask,
    annulus_mask,
    average_valid,
    dorsal_surface_mask,
    dorsal_surface_temperature,
    gate_measurement,
    local_floor_temperature,
    proximal_tail_points,
    sample_curve_max_percentile,
    tail_base_delta_t,
    warm_spot_temperature,
    warp_mask_to_thermal,
)
from tests.fixtures.synth_mouse_mask import make_synthetic_mouse_mask


def _landmarks():
    mask = make_synthetic_mouse_mask()
    return mask, extract_landmarks_from_mask(mask)


class TestAnteriorRegionMask:
    def test_anterior_third_covers_nose_end_not_tail_end(self):
        mask, landmarks = _landmarks()
        body_path = landmarks.path_nose_to_tail[: landmarks.tail_base_index + 1]
        anterior = anterior_region_mask(mask, body_path, fraction=1 / 3)
        ys, xs = np.where(anterior)
        assert xs.min() >= landmarks.nose_point[1] - 1
        # Should not extend anywhere near the tail base (x~146)
        assert xs.max() < landmarks.tail_base_point[1]

    def test_empty_mask_returns_empty(self):
        empty = np.zeros((10, 10), dtype=bool)
        result = anterior_region_mask(empty, [(0, 0), (0, 5)])
        assert not np.any(result)


class TestDorsalSurfaceMask:
    def test_excludes_tail_includes_body(self):
        mask, landmarks = _landmarks()
        dorsal = dorsal_surface_mask(mask, landmarks.path_nose_to_tail, landmarks.tail_base_index)
        ys, xs = np.where(dorsal)
        # Body spans nose_x(~15) to tail_base_x(~146); dorsal region shouldn't
        # extend meaningfully past the tail base into the thin tail filament.
        assert xs.min() <= landmarks.nose_point[1] + 2
        assert xs.max() <= landmarks.tail_base_point[1] + 5
        # Should recover most of the body's area (mask minus the thin tail).
        assert dorsal.sum() > 0.8 * mask.sum()

    def test_empty_mask_returns_empty(self):
        empty = np.zeros((10, 10), dtype=bool)
        result = dorsal_surface_mask(empty, [(0, 0), (0, 5)], tail_base_index=1)
        assert not np.any(result)

    def test_zero_tail_base_index_returns_empty(self):
        mask, landmarks = _landmarks()
        result = dorsal_surface_mask(mask, landmarks.path_nose_to_tail, tail_base_index=0)
        assert not np.any(result)


class TestDorsalSurfaceTemperature:
    def test_mean_and_median_of_warped_region(self):
        thermal = np.full((50, 50), 20.0)
        thermal[10:20, 10:20] = 30.0
        region = np.zeros((50, 50), dtype=bool)
        region[10:20, 10:20] = True
        result = dorsal_surface_temperature(thermal, region)
        assert result.valid
        assert result.mean_c == pytest.approx(30.0)
        assert result.median_c == pytest.approx(30.0)
        assert result.pixel_count == 100

    def test_invalid_when_too_few_pixels(self):
        thermal = np.full((50, 50), 20.0)
        region = np.zeros((50, 50), dtype=bool)
        region[0, 0] = True  # single pixel
        result = dorsal_surface_temperature(thermal, region, min_valid_pixels=10)
        assert not result.valid
        assert result.mean_c is None


class TestProximalTailPoints:
    def test_returns_points_near_tail_base_only(self):
        _mask, landmarks = _landmarks()
        proximal = proximal_tail_points(landmarks.tail_centerline, fraction=0.2)
        assert proximal[0] == landmarks.tail_centerline[0]
        assert len(proximal) < len(landmarks.tail_centerline)

    def test_degenerate_short_tail_keeps_base_point(self):
        result = proximal_tail_points([(0, 0)], fraction=0.2)
        assert result == [(0, 0)]


class TestWarmSpotAndTailIdentityHomography:
    """H = identity, so warped coordinates equal RGB coordinates directly —
    isolates the sampling logic from the warping logic (tested separately)."""

    def _setup(self):
        mask, landmarks = _landmarks()
        body_path = landmarks.path_nose_to_tail[: landmarks.tail_base_index + 1]
        anterior = anterior_region_mask(mask, body_path, fraction=1 / 3)
        proximal = proximal_tail_points(landmarks.tail_centerline, fraction=0.2)

        h, w = mask.shape
        thermal = (20 + 0.05 * np.tile(np.arange(w), (h, 1))).astype(np.float32)
        thermal[:, 15:60] = 38.0  # overlaps the anterior region
        thermal[30:50, 145:156] = 34.0  # overlaps the proximal tail region

        H = np.eye(3)
        return mask, landmarks, anterior, proximal, thermal, H

    def test_warm_spot_recovers_injected_temperature(self):
        _mask, _lm, anterior, _prox, thermal, H = self._setup()
        warped = warp_mask_to_thermal(anterior, H, thermal.shape)
        result = warm_spot_temperature(thermal, warped, percentile=95)
        assert result == pytest.approx(38.0, abs=0.1)

    def test_warm_spot_none_when_warped_region_empty(self):
        empty_mask = np.zeros((10, 10), dtype=bool)
        assert warm_spot_temperature(np.zeros((10, 10)), empty_mask) is None

    def test_tail_base_delta_t_recovers_injected_bump_above_local_floor(self):
        mask, _lm, _ant, proximal, thermal, H = self._setup()
        animal_mask_thermal = warp_mask_to_thermal(mask, H, thermal.shape)
        result = tail_base_delta_t(
            thermal, H, proximal,
            sample_radius_px=2, floor_inner_radius_px=8, floor_outer_radius_px=16,
            animal_mask_thermal=animal_mask_thermal,
        )
        assert result.valid
        assert result.tail_temp_c == pytest.approx(34.0, abs=0.1)
        # true local floor near x~150 is ~27.5C; annulus median should land close to that,
        # not near the injected 34C bump (which is excluded via the animal mask) or the
        # far-away 38C warm-spot region.
        assert 26.0 < result.floor_temp_c < 29.5
        assert result.delta_t_c > 5.0


class TestWarmSpotAndTailNonTrivialHomography:
    """H does real scale+translate work, exercising warp_mask_to_thermal /
    apply_homography together with the sampling logic, not just identity."""

    def test_recovers_correct_values_through_a_real_warp(self):
        mask, landmarks = _landmarks()
        body_path = landmarks.path_nose_to_tail[: landmarks.tail_base_index + 1]
        anterior = anterior_region_mask(mask, body_path, fraction=1 / 3)
        proximal = proximal_tail_points(landmarks.tail_centerline, fraction=0.2)

        H = np.array([[2.0, 0, 10.0], [0, 2.0, 5.0], [0, 0, 1.0]])
        thermal_shape = (200, 500)
        thermal = (10 + 0.03 * np.tile(np.arange(500), (200, 1))).astype(np.float32)

        warped_anterior = warp_mask_to_thermal(anterior, H, thermal_shape)
        assert np.any(warped_anterior)
        thermal[warped_anterior] = 40.0
        warm_temp = warm_spot_temperature(thermal, warped_anterior, percentile=95)
        assert warm_temp == pytest.approx(40.0, abs=0.1)

        warped_tail_xy = apply_homography(H, np.array([[p[1], p[0]] for p in proximal]))
        for x, y in warped_tail_xy:
            xi, yi = int(round(x)), int(round(y))
            thermal[max(0, yi - 2) : yi + 3, max(0, xi - 2) : xi + 3] = 36.0

        animal_mask_thermal = warp_mask_to_thermal(mask, H, thermal_shape)
        result = tail_base_delta_t(
            thermal, H, proximal,
            sample_radius_px=3, floor_inner_radius_px=10, floor_outer_radius_px=25,
            animal_mask_thermal=animal_mask_thermal,
        )
        assert result.valid
        assert result.tail_temp_c == pytest.approx(36.0, abs=0.1)
        assert result.delta_t_c > 10.0  # tail bump is well above the ~10-25C local gradient here


class TestLocalFloorTemperatureAndAnnulus:
    def test_annulus_rejects_invalid_radii(self):
        with pytest.raises(ValueError):
            annulus_mask(10, 10, inner_radius=10, outer_radius=5, shape=(50, 50))

    def test_returns_none_when_fully_excluded(self):
        thermal = np.full((50, 50), 20.0)
        full_exclude = np.ones((50, 50), dtype=bool)
        result = local_floor_temperature(thermal, (25, 25), 5, 15, exclude_masks=[full_exclude])
        assert result is None

    def test_returns_median_of_clean_annulus(self):
        thermal = np.full((50, 50), 20.0)
        thermal[0:10, 0:10] = 99.0  # far corner, shouldn't be sampled
        result = local_floor_temperature(thermal, (25, 25), 5, 15)
        assert result == pytest.approx(20.0)


class TestSampleCurveMaxPercentile:
    def test_no_valid_points_returns_none(self):
        thermal = np.zeros((10, 10))
        out_of_bounds = np.array([[-5, -5], [100, 100]])
        assert sample_curve_max_percentile(thermal, out_of_bounds, sample_radius_px=1) is None

    def test_partial_out_of_bounds_uses_valid_points_only(self):
        thermal = np.full((20, 20), 15.0)
        thermal[10, 10] = 30.0
        points = np.array([[-5, -5], [10, 10]])
        result = sample_curve_max_percentile(thermal, points, sample_radius_px=1, percentile=50)
        assert result == pytest.approx(30.0)


class TestGateMeasurement:
    def test_all_pass(self):
        valid, reasons = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=True
        )
        assert valid
        assert reasons == []

    def test_small_delta_t_fails(self):
        valid, reasons = gate_measurement(
            delta_t_c=0.05, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=True
        )
        assert not valid
        assert any("delta_t" in r for r in reasons)

    def test_low_confidence_fails(self):
        valid, reasons = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.1, sync_qc_pass=True, homography_qc_pass=True
        )
        assert not valid

    def test_sync_or_homography_failure_fails(self):
        valid, _ = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.9, sync_qc_pass=False, homography_qc_pass=True
        )
        assert not valid
        valid, _ = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=False
        )
        assert not valid

    def test_posture_none_does_not_invalidate(self):
        valid, reasons = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=True, posture_ok=None
        )
        assert valid

    def test_posture_false_invalidates(self):
        valid, reasons = gate_measurement(
            delta_t_c=1.0, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=True, posture_ok=False
        )
        assert not valid

    def test_none_delta_t_fails(self):
        valid, reasons = gate_measurement(
            delta_t_c=None, landmark_confidence=0.9, sync_qc_pass=True, homography_qc_pass=True
        )
        assert not valid

    def test_custom_thresholds(self):
        valid, _ = gate_measurement(
            delta_t_c=0.4,
            landmark_confidence=0.9,
            sync_qc_pass=True,
            homography_qc_pass=True,
            thresholds=GatingThresholds(min_abs_delta_t_c=0.5),
        )
        assert not valid


class TestAverageValid:
    def test_averages_only_non_none_values(self):
        assert average_valid([1.0, None, 3.0]) == pytest.approx(2.0)

    def test_below_min_count_returns_none(self):
        assert average_valid([1.0, None], min_count=2) is None

    def test_all_none_returns_none(self):
        assert average_valid([None, None]) is None
