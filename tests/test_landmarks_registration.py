"""Tests for src/landmarks/registration.py — no real data required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.registration import (
    HomographyFit,
    apply_homography,
    apply_height_correction,
    detect_nudges,
    fit_homography,
    homography_deviates_from_canonical,
    reprojection_error,
)

TRUE_H = np.array(
    [
        [1.05, 0.02, 15.0],
        [-0.01, 0.98, -8.0],
        [0.0002, -0.0001, 1.0],
    ]
)


def _project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1))
    homo = np.hstack([points, ones])
    proj = (H @ homo.T).T
    return proj[:, :2] / proj[:, 2:3]


class TestFitHomography:
    def test_exact_correspondences_recover_true_homography(self):
        rng = np.random.default_rng(0)
        rgb_pts = rng.uniform(0, 400, size=(8, 2))
        thermal_pts = _project(rgb_pts, TRUE_H)

        fit = fit_homography(rgb_pts, thermal_pts)
        assert fit.rmse_px < 1e-3  # near machine precision; not exactly 0 due to the linear solve
        assert fit.passes_acceptance(max_rmse_px=1.0)

    def test_noisy_correspondences_give_small_but_nonzero_rmse(self):
        rng = np.random.default_rng(1)
        rgb_pts = rng.uniform(0, 400, size=(10, 2))
        thermal_pts = _project(rgb_pts, TRUE_H) + rng.normal(0, 0.3, size=(10, 2))

        fit = fit_homography(rgb_pts, thermal_pts)
        assert 0 < fit.rmse_px < 0.5
        assert fit.passes_acceptance(max_rmse_px=1.0)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_homography(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError):
            fit_homography(np.zeros((4, 2)), np.zeros((5, 2)))

    def test_grossly_wrong_fit_fails_acceptance(self):
        rng = np.random.default_rng(2)
        rgb_pts = rng.uniform(0, 400, size=(6, 2))
        thermal_pts = _project(rgb_pts, TRUE_H) + rng.normal(0, 15.0, size=(6, 2))
        fit = fit_homography(rgb_pts, thermal_pts)
        assert not fit.passes_acceptance(max_rmse_px=1.0)


class TestApplyHomographyAndReprojectionError:
    def test_apply_homography_matches_manual_projection(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(0, 400, size=(5, 2))
        expected = _project(pts, TRUE_H)
        result = apply_homography(TRUE_H, pts)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_reprojection_error_zero_for_self_consistent_points(self):
        rng = np.random.default_rng(4)
        rgb_pts = rng.uniform(0, 400, size=(6, 2))
        thermal_pts = _project(rgb_pts, TRUE_H)
        rmse, residuals = reprojection_error(TRUE_H, rgb_pts, thermal_pts)
        assert rmse < 1e-6
        assert np.all(residuals < 1e-6)


class TestHomographyDeviatesFromCanonical:
    def test_identical_homographies_do_not_deviate(self):
        probe = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=float)
        deviates, disp = homography_deviates_from_canonical(TRUE_H, TRUE_H, probe, max_displacement_px=1.0)
        assert not deviates
        assert np.all(disp < 1e-6)

    def test_shifted_homography_deviates(self):
        H_nudged = TRUE_H.copy()
        H_nudged[0, 2] += 20.0  # 20px x-shift, simulating a rig nudge
        probe = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=float)
        deviates, disp = homography_deviates_from_canonical(TRUE_H, H_nudged, probe, max_displacement_px=1.0)
        assert deviates
        assert np.all(disp > 1.0)


class TestDetectNudges:
    def test_no_nudges_in_smooth_track(self):
        times = np.arange(20, dtype=float)
        positions = np.column_stack([times * 0.1, np.zeros(20)])  # slow smooth drift
        events = detect_nudges(times, positions, threshold_px=5.0)
        assert events == []

    def test_single_jump_detected(self):
        times = np.arange(20, dtype=float)
        positions = np.column_stack([np.zeros(20), np.zeros(20)])
        positions[10:] += 25.0  # sudden jump at t=10
        events = detect_nudges(times, positions, threshold_px=5.0)
        assert len(events) == 1
        assert events[0].time_sec == 10.0
        assert events[0].displacement_px == pytest.approx(25.0 * np.sqrt(2), abs=1e-6)

    def test_unsorted_input_is_sorted_first(self):
        times = np.array([2.0, 0.0, 1.0])
        positions = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        events = detect_nudges(times, positions, threshold_px=100.0)
        assert events == []  # smooth 0->1->2 once sorted; would look jumpy unsorted

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            detect_nudges(np.arange(5), np.zeros((4, 2)), threshold_px=1.0)

    def test_fewer_than_two_points_returns_empty(self):
        assert detect_nudges(np.array([0.0]), np.array([[0.0, 0.0]]), threshold_px=1.0) == []


class TestHeightCorrection:
    def test_default_is_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = apply_height_correction(pts)
        np.testing.assert_allclose(result, pts)

    def test_custom_correction_fn_applied(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = apply_height_correction(pts, correction_fn=lambda p: p + 5.0)
        np.testing.assert_allclose(result, pts + 5.0)
