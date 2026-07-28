"""Unit tests for roi_geometry.py — no .seq file required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.roi_geometry import (
    circular_mask,
    find_floor_roi,
    find_mouse_roi_center,
    largest_inscribed_circle,
    roi_fully_inside,
    roi_in_bounds,
    roi_overlaps_mask,
)


class TestCircularMask:
    def test_center_pixel_is_true(self):
        mask = circular_mask(10, 10, 5, (20, 20))
        assert mask[10, 10] is np.bool_(True)

    def test_outside_pixel_is_false(self):
        mask = circular_mask(10, 10, 5, (20, 20))
        assert not mask[0, 0]

    def test_shape_matches_input(self):
        mask = circular_mask(5, 5, 3, (30, 40))
        assert mask.shape == (30, 40)

    def test_radius_zero_single_pixel(self):
        mask = circular_mask(5, 5, 0, (10, 10))
        # Only the center pixel should be True (distance 0 <= 0)
        assert mask[5, 5]
        assert mask.sum() == 1

    def test_symmetry(self):
        mask = circular_mask(10, 10, 4, (20, 20))
        # Top half and bottom half should have equal True counts (center row excluded)
        above = mask[:10, :].sum()
        below = mask[11:, :].sum()
        assert above == below


class TestRoiBounds:
    def test_fits_inside(self):
        assert roi_in_bounds(10, 10, 5, (30, 30))

    def test_touches_left_edge(self):
        # cx=5, r=5 → leftmost pixel at x=0, which is on the edge
        assert roi_in_bounds(5, 10, 5, (30, 30))

    def test_exceeds_left_edge(self):
        assert not roi_in_bounds(3, 10, 5, (30, 30))

    def test_exceeds_right_edge(self):
        assert not roi_in_bounds(27, 10, 5, (30, 30))

    def test_exceeds_bottom_edge(self):
        assert not roi_in_bounds(10, 27, 5, (30, 30))


class TestRoiFullyInside:
    def _full_mask(self, h, w):
        return np.ones((h, w), dtype=bool)

    def test_inside_full_mask(self):
        mask = self._full_mask(50, 50)
        assert roi_fully_inside(25, 25, 5, mask)

    def test_outside_image_bounds(self):
        mask = self._full_mask(20, 20)
        assert not roi_fully_inside(2, 10, 5, mask)  # left edge

    def test_partial_mask_excludes_roi(self):
        mask = np.ones((50, 50), dtype=bool)
        mask[20:30, 20:30] = False  # hole in mask
        # ROI centered at (25, 25) radius 3 — overlaps the hole
        assert not roi_fully_inside(25, 25, 3, mask)

    def test_roi_just_fits(self):
        mask = self._full_mask(20, 20)
        # center=(5,5), r=5 → touches x=0, y=0, x=10, y=10 — all within 20x20
        assert roi_fully_inside(5, 5, 5, mask)


class TestLargestInscribedCircle:
    def test_square_mask_center(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:25, 5:25] = True
        cx, cy, r = largest_inscribed_circle(mask)
        # Center should be near (14.5, 14.5) and radius ~10
        assert 12 <= cx <= 17
        assert 12 <= cy <= 17
        assert r >= 9

    def test_thin_horizontal_stripe(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[14:16, 5:25] = True  # 2 rows tall
        _, _, r = largest_inscribed_circle(mask)
        assert r <= 2  # can't fit a large circle in a thin stripe


class TestFindMouseRoiCenter:
    def test_large_mask_returns_center(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True  # 30x30 block
        result = find_mouse_roi_center(mask, roi_radius=5)
        assert result is not None
        cx, cy = result
        # Should be well inside the block
        assert 15 <= cx <= 35
        assert 15 <= cy <= 35

    def test_too_small_mask_returns_none(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True  # 3x3 — can't fit r=5
        result = find_mouse_roi_center(mask, roi_radius=5)
        assert result is None

    def test_empty_mask_returns_none(self):
        mask = np.zeros((50, 50), dtype=bool)
        result = find_mouse_roi_center(mask, roi_radius=5)
        assert result is None


class TestFindFloorRoi:
    def _setup(self, h=100, w=60):
        """Full arena mask and a mouse blob in the middle."""
        arena = np.zeros((h, w), dtype=bool)
        arena[2:h-2, 2:w-2] = True  # full image minus border

        mouse = np.zeros((h, w), dtype=bool)
        mouse[45:55, 25:35] = True  # mouse at ~row 50

        return arena, mouse

    def test_finds_valid_roi(self):
        arena, mouse = self._setup()
        result = find_floor_roi(
            mouse_cx=30.0, mouse_cy=50.0,
            roi_radius=5,
            mouse_mask=mouse,
            arena_mask=arena,
            max_shift_px=40,
            step_px=2,
        )
        assert result is not None
        cx, cy, direction, dist = result
        assert direction in ("up", "down")
        assert dist > 0

    def test_direction_is_away_from_mouse(self):
        arena, mouse = self._setup()
        result = find_floor_roi(
            mouse_cx=30.0, mouse_cy=50.0,
            roi_radius=5,
            mouse_mask=mouse,
            arena_mask=arena,
            max_shift_px=40,
            step_px=2,
        )
        assert result is not None
        cx, cy, direction, dist = result
        if direction == "up":
            assert cy < 50
        else:
            assert cy > 50

    def test_returns_none_if_no_valid_position(self):
        # Arena mask that only covers the mouse area — no valid floor position
        h, w = 20, 20
        arena = np.zeros((h, w), dtype=bool)
        arena[8:12, 8:12] = True  # tiny valid region, all covered by mouse
        mouse = np.zeros((h, w), dtype=bool)
        mouse[8:12, 8:12] = True

        result = find_floor_roi(
            mouse_cx=10.0, mouse_cy=10.0,
            roi_radius=3,
            mouse_mask=mouse,
            arena_mask=arena,
            max_shift_px=10,
            step_px=1,
        )
        assert result is None

    def test_floor_roi_does_not_overlap_mouse(self):
        arena, mouse = self._setup()
        result = find_floor_roi(
            mouse_cx=30.0, mouse_cy=50.0,
            roi_radius=5,
            mouse_mask=mouse,
            arena_mask=arena,
            max_shift_px=40,
            step_px=2,
        )
        assert result is not None
        cx, cy, _, _ = result
        from src.roi_geometry import roi_overlaps_mask
        # Floor ROI should not overlap the mouse
        assert not roi_overlaps_mask(cx, cy, 5, mouse)
