"""Tests for src/landmarks/entry_detection.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.entry_detection import (
    classify_frame_state_rgb,
    classify_frame_state_thermal,
    find_entry_frame_index_rgb,
    find_entry_frame_index_thermal,
    find_entry_index,
)
from src.landmarks.rgb_landmarks import RgbBackgroundModel
from src.mouse_segmentation import BackgroundModel as ThermalBackgroundModel

MIN_AREA = 50
MAX_AREA = 500


def _bg_frame(h=80, w=200, value=200.0):
    return np.full((h, w), value)


def _blob_frame(h, w, y0, y1, x0, x1, value=30.0, bg_value=200.0):
    frame = np.full((h, w), bg_value)
    frame[y0:y1, x0:x1] = value
    return frame


class TestFindEntryIndex:
    def test_no_intrusion_returns_none(self):
        states = ["empty", "mouse", "mouse", "mouse", "mouse", "mouse"]
        assert find_entry_index(states, min_sustained_detections=3) is None

    def test_intrusion_then_sustained_mouse_returns_run_start(self):
        states = ["empty", "empty", "intrusion", "intrusion", "mouse", "mouse", "mouse", "mouse"]
        assert find_entry_index(states, min_sustained_detections=3) == 4

    def test_intrusion_then_insufficient_run_returns_none(self):
        states = ["intrusion", "mouse", "mouse", "empty", "empty"]
        assert find_entry_index(states, min_sustained_detections=3) is None

    def test_run_interrupted_by_empty_resets_count(self):
        states = ["intrusion", "mouse", "mouse", "empty", "mouse", "mouse", "mouse"]
        # first run (len 2) breaks at index 3 -> second run starts at index 4
        assert find_entry_index(states, min_sustained_detections=3) == 4

    def test_returns_first_valid_entry_even_if_a_later_intrusion_exists(self):
        # A later intrusion (e.g. a mid-session hand adjustment) must NOT
        # retroactively invalidate an already-found, already-sustained real
        # entry earlier in the session -- the first valid entry point is
        # exactly what real usage (bout filtering) needs.
        states = [
            "intrusion", "mouse", "mouse", "mouse",  # sustained run after intrusion #1 -> real entry
            "intrusion",  # a second, later intrusion
            "mouse", "mouse", "mouse",
        ]
        assert find_entry_index(states, min_sustained_detections=3) == 1

    def test_exact_threshold_boundary(self):
        states = ["intrusion", "mouse", "mouse", "mouse"]
        assert find_entry_index(states, min_sustained_detections=3) == 1
        assert find_entry_index(states, min_sustained_detections=4) is None


class TestClassifyFrameStateRgb:
    def _model(self):
        return RgbBackgroundModel.build([_bg_frame() for _ in range(9)])

    def test_empty_frame_is_empty(self):
        state = classify_frame_state_rgb(_bg_frame(), self._model(), MIN_AREA, MAX_AREA)
        assert state == "empty"

    def test_mouse_scale_blob_is_mouse(self):
        # 15x40 = 600px, wait must be within [50,500] -- use a smaller elongated blob
        frame = _blob_frame(80, 200, 30, 40, 60, 90)  # 10x30 = 300px, aspect 3 -> elongated -> "extended"
        state = classify_frame_state_rgb(frame, self._model(), MIN_AREA, MAX_AREA)
        assert state == "mouse"

    def test_oversized_blob_is_intrusion(self):
        # A hand/arm-scale blob: bigger than max_area (500) but still within the
        # widened intrusion search bound (500 * 3.0 = 1500) so it's actually
        # detected rather than filtered out; also small enough relative to the
        # whole frame (80x200=16000px) that it doesn't skew the adaptive
        # threshold into detecting nothing at all (see the ~20%-of-frame
        # caveat noted in test_landmarks_rgb_landmarks.py).
        frame = _blob_frame(80, 200, 20, 40, 50, 100)  # 20x50 = 1000px
        state = classify_frame_state_rgb(frame, self._model(), MIN_AREA, MAX_AREA)
        assert state == "intrusion"


class TestClassifyFrameStateThermal:
    def _model(self):
        return ThermalBackgroundModel(_bg_frame())

    def test_empty_frame_is_empty(self):
        state = classify_frame_state_thermal(_bg_frame(), self._model(), MIN_AREA, MAX_AREA)
        assert state == "empty"

    def test_mouse_scale_blob_is_mouse(self):
        frame = _blob_frame(80, 200, 30, 40, 60, 90)
        state = classify_frame_state_thermal(frame, self._model(), MIN_AREA, MAX_AREA)
        assert state == "mouse"

    def test_oversized_blob_is_intrusion(self):
        frame = _blob_frame(80, 200, 20, 40, 50, 100)  # 20x50 = 1000px
        state = classify_frame_state_thermal(frame, self._model(), MIN_AREA, MAX_AREA)
        assert state == "intrusion"


class TestFindEntryFrameIndexEndToEnd:
    def test_rgb_pipeline_detects_intrusion_then_settled_mouse(self):
        model = RgbBackgroundModel.build([_bg_frame() for _ in range(9)])
        frames = [
            _bg_frame(), _bg_frame(),  # empty, before anything happens
            _blob_frame(80, 200, 20, 40, 50, 100),  # hand/arm intrusion (1000px)
            _blob_frame(80, 200, 20, 40, 50, 100),
            _blob_frame(80, 200, 30, 40, 60, 90),  # mouse settles, elongated (300px)
            _blob_frame(80, 200, 30, 40, 62, 92),
            _blob_frame(80, 200, 30, 40, 64, 94),
        ]
        idx = find_entry_frame_index_rgb(frames, model, MIN_AREA, MAX_AREA, min_sustained_detections=3)
        assert idx == 4

    def test_thermal_pipeline_detects_intrusion_then_settled_mouse(self):
        model = ThermalBackgroundModel(_bg_frame())
        frames = [
            _bg_frame(),
            _blob_frame(80, 200, 20, 40, 50, 100),
            _blob_frame(80, 200, 30, 40, 60, 90),
            _blob_frame(80, 200, 30, 40, 62, 92),
            _blob_frame(80, 200, 30, 40, 64, 94),
        ]
        idx = find_entry_frame_index_thermal(frames, model, MIN_AREA, MAX_AREA, min_sustained_detections=3)
        assert idx == 2

    def test_no_intrusion_in_clip_returns_none(self):
        model = RgbBackgroundModel.build([_bg_frame() for _ in range(9)])
        frames = [_blob_frame(80, 200, 30, 40, 60, 90) for _ in range(5)]
        idx = find_entry_frame_index_rgb(frames, model, MIN_AREA, MAX_AREA, min_sustained_detections=3)
        assert idx is None


class TestChangedFractionIntrusion:
    """A light glove/bare arm against the backlit lane (Test_7 Front, 2026-10-05) is invisible to the
    dark-blob area rule; intrusion_changed_fraction catches it."""

    def _model(self):
        return RgbBackgroundModel.build([_bg_frame() for _ in range(9)])

    def _light_arm(self):
        return _blob_frame(80, 200, 10, 70, 20, 130, value=245.0)  # 60x110 = 41% of the crop, lighter than bg

    def test_light_arm_missed_by_area_rule_alone(self):
        assert classify_frame_state_rgb(self._light_arm(), self._model(), MIN_AREA, MAX_AREA) != "intrusion"

    def test_light_arm_is_intrusion_with_changed_fraction(self):
        state = classify_frame_state_rgb(self._light_arm(), self._model(), MIN_AREA, MAX_AREA,
                                         intrusion_changed_fraction=0.05)
        assert state == "intrusion"

    def test_mouse_alone_stays_mouse_with_changed_fraction(self):
        frame = _blob_frame(80, 200, 30, 40, 60, 90)  # 300px = 1.9% of the crop
        state = classify_frame_state_rgb(frame, self._model(), MIN_AREA, MAX_AREA, intrusion_changed_fraction=0.05)
        assert state == "mouse"

    def test_end_to_end_entry_after_light_arm(self):
        model = self._model()
        frames = [_bg_frame(), self._light_arm(), self._light_arm(),
                  _blob_frame(80, 200, 30, 40, 60, 90), _blob_frame(80, 200, 30, 40, 62, 92),
                  _blob_frame(80, 200, 30, 40, 64, 94)]
        states = [classify_frame_state_rgb(f, model, MIN_AREA, MAX_AREA, intrusion_changed_fraction=0.05)
                  for f in frames]
        assert find_entry_index(states, min_sustained_detections=3) == 3


class TestMinIntrusionRun:
    def test_brief_blip_does_not_count_as_intrusion(self):
        states = ["empty", "intrusion", "mouse", "mouse", "mouse", "mouse"]  # 1-sample blip, then a static "mouse"
        assert find_entry_index(states, min_sustained_detections=3, min_intrusion_run=2) is None

    def test_blip_then_real_placement_returns_entry_after_placement(self):
        states = (["intrusion"] + ["mouse"] * 5          # blip + static artifact (Test_4 Front, t=14s)
                  + ["intrusion"] * 4 + ["mouse"] * 4)   # real placement, then the mouse alone
        assert find_entry_index(states, min_sustained_detections=3, min_intrusion_run=3) == 10

    def test_default_keeps_previous_behavior(self):
        states = ["intrusion", "mouse", "mouse", "mouse"]
        assert find_entry_index(states, min_sustained_detections=3) == 1


class TestEmptyLaneAndSessionEpisode:
    def test_tiny_static_blob_is_empty_with_mouse_floor(self):
        model = RgbBackgroundModel.build([_bg_frame() for _ in range(9)])
        frame = _blob_frame(80, 200, 30, 33, 60, 90)  # 90px static blob = 0.56% of the crop, under a 1% floor
        state = classify_frame_state_rgb(frame, model, 50, MAX_AREA, min_changed_fraction_for_mouse=0.01)
        assert state == "empty"

    def test_external_intrusion_opens_window(self):
        states = ["empty", "empty", "mouse", "mouse", "mouse"]  # this lane never saw the (faint) arm
        assert find_entry_index(states, min_sustained_detections=3) is None
        assert find_entry_index(states, min_sustained_detections=3, intrusion_seen_from=1) == 2

    def test_own_intrusion_still_resets_after_external_window(self):
        states = ["mouse", "mouse", "intrusion", "mouse", "mouse", "mouse"]
        assert find_entry_index(states, min_sustained_detections=3, intrusion_seen_from=0) == 3
