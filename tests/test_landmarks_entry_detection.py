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
