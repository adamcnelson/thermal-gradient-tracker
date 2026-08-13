"""Tests for src/landmarks/rgb_landmarks.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.rgb_landmarks import (
    MouseSkeletonLandmarks,
    RgbBackgroundModel,
    extract_landmarks_from_mask,
    find_tail_base,
    is_plausible_mouse_blob,
    order_skeleton_path,
    segment_mouse_rgb,
    width_profile,
)
from tests.fixtures.synth_mouse_mask import make_synthetic_mouse_mask


class TestSegmentation:
    def test_background_model_recovers_static_background(self):
        rng = np.random.default_rng(0)
        bg_truth = rng.uniform(50, 60, size=(20, 20)).astype(np.float32)
        frames = [bg_truth + rng.normal(0, 1, size=(20, 20)) for _ in range(9)]
        model = RgbBackgroundModel.build(frames)
        np.testing.assert_allclose(model.background, bg_truth, atol=2.0)

    def test_segment_mouse_rgb_finds_a_darker_blob(self):
        # Frame must be large relative to the blob — the adaptive threshold
        # (mean + N*std of the foreground-score image) assumes the blob is a
        # small minority of pixels, same as the thermal-side algorithm. A
        # blob covering ~20% of a small frame pushes mean/std high enough
        # that nothing clears threshold (caught by this test initially).
        h, w = 80, 200
        rng = np.random.default_rng(1)
        bg_frames = [
            np.full((h, w), 200.0) + rng.normal(0, 1, size=(h, w)) for _ in range(9)
        ]
        model = RgbBackgroundModel.build(bg_frames)

        frame = np.full((h, w), 200.0)
        frame[30:45, 40:80] = 30.0  # dark mouse-like blob, ~3.75% of frame
        mask = segment_mouse_rgb(frame, model, min_area=50, max_area=2000)
        assert mask is not None
        assert 500 <= mask.sum() <= 620  # 15x40 blob, allowing for morphology erosion at edges

    def test_segment_mouse_rgb_none_when_nothing_qualifies(self):
        h, w = 40, 60
        model = RgbBackgroundModel.build([np.full((h, w), 200.0) for _ in range(5)])
        frame = np.full((h, w), 200.0)
        frame[10:12, 10:12] = 30.0  # tiny 2x2 blob, below any reasonable min_area
        mask = segment_mouse_rgb(frame, model, min_area=50, max_area=2000)
        assert mask is None


class TestIsPlausibleMouseBlob:
    def test_elongated_blob_passes(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:25, 5:45] = True  # 5x40 -> aspect 8
        assert is_plausible_mouse_blob(mask, min_area=50, max_area=5000)

    def test_round_blob_rejected_as_fecal_bolus(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:26, 20:26] = True  # 6x6 -> aspect 1
        assert not is_plausible_mouse_blob(mask, min_area=10, max_area=5000)

    def test_area_out_of_bounds_rejected(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:22, 5:45] = True  # small area, elongated
        assert not is_plausible_mouse_blob(mask, min_area=1000, max_area=5000)


class TestOrderSkeletonPath:
    def test_orders_full_skeleton_from_synthetic_mouse(self):
        mask = make_synthetic_mouse_mask()
        skel = skeletonize(mask)
        path = order_skeleton_path(skel)
        assert len(path) == int(skel.sum())

    def test_branched_skeleton_raises(self):
        # A 'Y' shape: three arms meeting at a center pixel -> 3 endpoints
        skel = np.zeros((20, 20), dtype=bool)
        skel[10, 2:10] = True
        skel[2:10, 10] = True
        for i in range(8):
            skel[10 + i, 10 + i] = True
        with pytest.raises(ValueError):
            order_skeleton_path(skel)

    def test_closed_loop_raises(self):
        skel = np.zeros((20, 20), dtype=bool)
        skel[5, 5:15] = True
        skel[15, 5:15] = True
        skel[5:15, 5] = True
        skel[5:15, 15] = True
        with pytest.raises(ValueError):
            order_skeleton_path(skel)


class TestWidthProfile:
    def test_widths_are_widest_at_trunk_narrowest_at_tail(self):
        mask = make_synthetic_mouse_mask()
        skel = skeletonize(mask)
        path = order_skeleton_path(skel)
        widths = width_profile(mask, path)
        assert widths.max() > 20  # trunk
        assert widths.min() < 5  # tail tip


class TestFindTailBase:
    def test_full_pipeline_orients_nose_first_and_finds_tail_base(self):
        landmarks = extract_landmarks_from_mask(make_synthetic_mouse_mask())
        assert isinstance(landmarks, MouseSkeletonLandmarks)
        # nose end (x ~ 15) should have a smaller x than tail end (x ~ 200)
        assert landmarks.nose_point[1] < landmarks.tail_tip_point[1]
        # tail base should sit near the true synthetic junction (x=150), not exact
        # since skeletonization/width-threshold introduce some slack
        assert 140 <= landmarks.tail_base_point[1] <= 155
        # tail centerline should run from tail base out to the tail tip
        assert landmarks.tail_centerline[0] == landmarks.tail_base_point
        assert landmarks.tail_centerline[-1] == landmarks.tail_tip_point
        # widths should be mostly decreasing along the tail centerline
        tail_widths = landmarks.widths_nose_to_tail[landmarks.tail_base_index :]
        assert tail_widths[0] > tail_widths[-1]

    def test_orientation_is_symmetric_regardless_of_which_endpoint_the_walk_started_from(self):
        mask = make_synthetic_mouse_mask()
        landmarks_a = extract_landmarks_from_mask(mask)
        # Reversing the raw path before calling find_tail_base should give the same answer
        skel = skeletonize(mask)
        path = order_skeleton_path(skel)
        widths = width_profile(mask, path)
        landmarks_b = find_tail_base(list(reversed(path)), widths[::-1])
        assert landmarks_a.nose_point == landmarks_b.nose_point
        assert landmarks_a.tail_base_point == landmarks_b.tail_base_point

    def test_short_path_raises(self):
        with pytest.raises(ValueError):
            find_tail_base([(0, 0), (0, 1)], np.array([5.0, 5.0]))
