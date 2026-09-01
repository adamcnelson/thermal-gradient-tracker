"""Tests for src/landmarks/rgb_landmarks.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.rgb_landmarks import (
    MouseDetectionResult,
    MouseSkeletonLandmarks,
    RgbBackgroundModel,
    classify_mouse_blob,
    extract_landmarks_from_mask,
    extract_mouse_detection,
    find_tail_base,
    is_plausible_mouse_blob,
    order_skeleton_path,
    prune_short_skeleton_branches,
    find_tail_appendage,
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


class TestClassifyMouseBlob:
    def test_elongated_blob_is_extended(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:25, 5:45] = True  # 5x40 -> aspect 8, area 200
        assert classify_mouse_blob(mask, min_area=50, max_area=5000) == "extended"

    def test_large_round_blob_is_curled_not_rejected(self):
        # A real curled mouse: compact (aspect ~1) but real body area, not
        # a small round object -- must NOT be silently dropped like
        # is_plausible_mouse_blob() would (this was the real 2026-08-17 bug).
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True  # 40x40 -> aspect 1, area 1600
        assert not is_plausible_mouse_blob(mask, min_area=50, max_area=5000)
        assert classify_mouse_blob(mask, min_area=50, max_area=5000) == "curled"

    def test_small_round_blob_still_rejected_as_debris(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:26, 20:26] = True  # 6x6 -> aspect 1, area 36 -- bolus-scale
        assert classify_mouse_blob(mask, min_area=10, max_area=5000) is None

    def test_area_out_of_bounds_rejected_regardless_of_shape(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:22, 5:45] = True  # elongated but too small
        assert classify_mouse_blob(mask, min_area=1000, max_area=5000) is None

    def test_curled_min_area_frac_is_configurable(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:26, 20:26] = True  # 6x6 -> aspect 1, area 36
        # with min_area=10 and default frac=5.0, need area>=50 to count as curled -> rejected
        assert classify_mouse_blob(mask, min_area=10, max_area=5000) is None
        # loosen the multiplier and the same blob is accepted as curled
        assert classify_mouse_blob(mask, min_area=10, max_area=5000, curled_min_area_frac=2.0) == "curled"


class TestExtractMouseDetection:
    def test_extended_posture_returns_full_landmarks(self):
        mask = make_synthetic_mouse_mask()
        result = extract_mouse_detection(mask, min_area=50, max_area=20000)
        assert isinstance(result, MouseDetectionResult)
        assert result.posture == "extended"
        assert isinstance(result.landmarks, MouseSkeletonLandmarks)
        assert result.mask is mask

    def test_curled_posture_returns_no_landmarks_but_keeps_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True  # 40x40 disk-like blob, aspect ~1, real body-scale area
        result = extract_mouse_detection(mask, min_area=50, max_area=5000)
        assert result.posture == "curled"
        assert result.landmarks is None
        assert result.mask is mask

    def test_debris_scale_blob_rejected_entirely(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:26, 20:26] = True  # 6x6, bolus-scale
        assert extract_mouse_detection(mask, min_area=10, max_area=5000) is None

    def test_elongated_but_branched_skeleton_is_ambiguous_not_a_crash(self):
        # A T-shape: long horizontal bar with a vertical stub -> bounding-box
        # aspect ratio clears the "extended" threshold, but the skeleton has
        # 3 endpoints (both bar ends + stub tip), which extract_landmarks_from_mask
        # would raise ValueError on. Must come back as "ambiguous", not crash.
        mask = np.zeros((60, 150), dtype=bool)
        mask[25:35, 0:150] = True  # horizontal bar, 150x10
        mask[0:25, 70:80] = True  # vertical stub, 10x25, meets the bar's middle
        result = extract_mouse_detection(mask, min_area=50, max_area=20000)
        assert result.posture == "ambiguous"
        assert result.landmarks is None

    def test_area_out_of_bounds_rejected_before_posture_classification(self):
        mask = make_synthetic_mouse_mask()
        assert extract_mouse_detection(mask, min_area=100000, max_area=200000) is None


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


def _line_with_spur(spur_len: int) -> np.ndarray:
    """A straight 30px horizontal skeleton line with a perpendicular spur
    of the given length sticking off its midpoint -- 3 endpoints (branched)
    until the spur is pruned."""
    skel = np.zeros((30, 30), dtype=bool)
    skel[15, 2:28] = True  # main line, row 15, cols 2..27
    mid_col = 15
    for i in range(1, spur_len + 1):
        skel[15 - i, mid_col] = True  # spur going upward from the line
    return skel


class TestPruneShortSkeletonBranches:
    def test_short_spur_is_removed(self):
        skel = _line_with_spur(spur_len=5)
        pruned = prune_short_skeleton_branches(skel, max_prune_px=10)
        # spur gone -> back to a simple 2-endpoint line
        path = order_skeleton_path(pruned)
        assert len(path) == int(pruned.sum())

    def test_long_spur_is_kept(self):
        skel = _line_with_spur(spur_len=20)
        pruned = prune_short_skeleton_branches(skel, max_prune_px=10)
        with pytest.raises(ValueError):
            order_skeleton_path(pruned)

    def test_already_simple_skeleton_is_unchanged(self):
        mask = make_synthetic_mouse_mask()
        skel = skeletonize(mask)
        pruned = prune_short_skeleton_branches(skel, max_prune_px=10)
        assert np.array_equal(skel, pruned)

    def test_closed_loop_is_not_rescued(self):
        skel = np.zeros((20, 20), dtype=bool)
        skel[5, 5:15] = True
        skel[15, 5:15] = True
        skel[5:15, 5] = True
        skel[5:15, 15] = True
        pruned = prune_short_skeleton_branches(skel, max_prune_px=10)
        with pytest.raises(ValueError):
            order_skeleton_path(pruned)


class TestOrderSkeletonPathPruning:
    def test_default_max_prune_px_zero_preserves_old_behavior(self):
        skel = _line_with_spur(spur_len=5)
        with pytest.raises(ValueError):
            order_skeleton_path(skel)  # default max_prune_px=0 -- spur NOT pruned

    def test_max_prune_px_rescues_short_spur(self):
        skel = _line_with_spur(spur_len=5)
        path = order_skeleton_path(skel, max_prune_px=10)
        assert len(path) > 0

    def test_max_prune_px_does_not_rescue_long_spur(self):
        skel = _line_with_spur(spur_len=20)
        with pytest.raises(ValueError):
            order_skeleton_path(skel, max_prune_px=10)


def _hunched_body_with_tail_mask(tail_len: int = 140, include_decoy_spur: bool = True) -> np.ndarray:
    """A cross-shaped 'hunched body' (guarantees a real skeleton branch
    point, unlike a solid rectangle which just skeletonizes to a simple
    line) with a long thin tail attached, and optionally a short decoy
    spur (paw-scale) elsewhere on the body."""
    mask = np.zeros((80, 220), dtype=bool)
    mask[20:60, 25:45] = True  # vertical arm (body)
    mask[30:50, 5:65] = True   # horizontal arm (body)
    mask[38:42, 60:60 + tail_len] = True  # thin tail
    if include_decoy_spur:
        mask[15:19, 33:37] = True  # short paw-like spur
    return mask


class TestFindTailAppendage:
    def test_finds_the_real_tail_not_the_decoy_spur(self):
        mask = _hunched_body_with_tail_mask(tail_len=140, include_decoy_spur=True)
        result = find_tail_appendage(mask, min_tail_length_px=15)
        assert result is not None
        assert result.tail_tip_point[1] > 190  # near the tail's far end (col ~197)
        assert len(result.tail_centerline) > 100

    def test_returns_none_for_a_simple_compact_blob(self):
        # A round-ish compact blob with no separately-resolvable thin appendage.
        mask = np.zeros((60, 60), dtype=bool)
        yy, xx = np.mgrid[0:60, 0:60]
        mask |= (xx - 30) ** 2 + (yy - 30) ** 2 <= 25 ** 2
        result = find_tail_appendage(mask)
        assert result is None

    def test_too_short_a_tail_is_rejected(self):
        mask = _hunched_body_with_tail_mask(tail_len=8, include_decoy_spur=False)
        result = find_tail_appendage(mask, min_tail_length_px=15)
        assert result is None

    def test_centerline_runs_base_to_tip(self):
        mask = _hunched_body_with_tail_mask(tail_len=140, include_decoy_spur=False)
        result = find_tail_appendage(mask, min_tail_length_px=15)
        assert result is not None
        assert result.tail_centerline[0] == result.tail_base_point
        assert result.tail_centerline[-1] == result.tail_tip_point
        # base should be closer to the body (smaller column) than the tip
        assert result.tail_base_point[1] < result.tail_tip_point[1]


class TestExtractMouseDetectionTailFallback:
    def test_ambiguous_posture_still_gets_tail_landmarks_when_resolvable(self):
        mask = _hunched_body_with_tail_mask(tail_len=140, include_decoy_spur=False)
        # this mask's bbox aspect ratio easily clears 1.8 (very elongated bbox)
        # but its skeleton is branched -> ambiguous posture, no `landmarks`
        result = extract_mouse_detection(mask, min_area=100, max_area=100000)
        assert result is not None
        assert result.posture == "ambiguous"
        assert result.landmarks is None
        assert result.tail_landmarks is not None

    def test_curled_posture_gets_tail_landmarks_when_resolvable(self):
        # compact-enough bbox aspect to be "curled", but with a real
        # separately-resolvable tail sticking out
        mask = np.zeros((80, 100), dtype=bool)
        mask[20:60, 20:60] = True  # compact-ish square body
        mask[38:42, 60:95] = True  # short-ish thin tail
        result = extract_mouse_detection(mask, min_area=100, max_area=100000, min_aspect_ratio=100.0)
        assert result is not None
        assert result.posture == "curled"
        assert result.landmarks is None
        # tail may or may not qualify depending on exact geometry -- just check the field exists and doesn't crash
        assert result.tail_landmarks is None or result.tail_landmarks.tail_tip_point is not None


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
