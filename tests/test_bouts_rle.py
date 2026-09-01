"""Tests for src/bouts.py — RLE bout detection."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bouts import classify_frame_state, classify_stationary, rle_bouts


def _make_mask_and_time(pattern, fps=9.0):
    """
    Build a stationary mask from a pattern list of (is_stationary, duration_sec) tuples.
    Returns (mask Series, elapsed_time_sec Series).
    """
    samples = []
    t = 0.0
    dt = 1.0 / fps
    for is_stat, dur in pattern:
        n = int(dur * fps)
        for _ in range(n):
            samples.append((is_stat, t))
            t += dt
    mask = pd.Series([s for s, _ in samples], dtype=bool)
    elapsed = pd.Series([t for _, t in samples])
    return mask, elapsed


def test_single_long_bout_detected():
    mask, t = _make_mask_and_time([(False, 10), (True, 30), (False, 10)])
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 1
    assert bouts.iloc[0]["bout_duration_sec"] == pytest.approx(30.0, abs=0.5)


def test_short_bout_filtered():
    mask, t = _make_mask_and_time([(False, 5), (True, 3), (False, 5)])
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 0


def test_gap_merging():
    """Two bouts separated by a short gap should be merged into one."""
    mask, t = _make_mask_and_time([
        (False, 5),
        (True, 10),
        (False, 1.5),   # gap < max_gap_sec=2.0 → merge
        (True, 10),
        (False, 5),
    ])
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 1
    assert bouts.iloc[0]["bout_duration_sec"] == pytest.approx(21.5, abs=1.0)


def test_gap_not_merged_when_too_large():
    mask, t = _make_mask_and_time([
        (False, 5),
        (True, 10),
        (False, 5),   # gap > max_gap_sec=2.0 → no merge
        (True, 10),
        (False, 5),
    ])
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 2


def test_multiple_bouts_indexed():
    mask, t = _make_mask_and_time([
        (False, 5), (True, 8), (False, 5),
        (True, 12), (False, 5), (True, 6), (False, 5),
    ])
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=1.0)
    assert len(bouts) == 3
    assert list(bouts["bout_index"]) == [0, 1, 2]


def test_empty_mask():
    mask = pd.Series([], dtype=bool)
    t = pd.Series([], dtype=float)
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 0


def test_all_stationary():
    mask = pd.Series([True] * 90)
    t = pd.Series(np.arange(90) / 9.0)
    bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
    assert len(bouts) == 1
    assert bouts.iloc[0]["bout_duration_sec"] == pytest.approx(10.0 - 1/9, abs=0.5)



def test_classify_stationary_basic():
    """Frames below dispersion threshold should be classified as stationary."""
    df = pd.DataFrame({
        "qc_flag": ["ok", "ok", "ok", "seg_error"],
        "mouse_roi_valid": [True, True, True, False],
        "centroid_x_roll_dispersion": [2.0, 1.5, 3.0, 1.0],
        "velocity_smooth_px_s": [1.0, 2.0, 1.0, 1.0],
    })
    stat = classify_stationary(df, disp_thresh=5.0, vel_thresh=15.0)
    assert list(stat) == [True, True, True, False]


def test_classify_stationary_velocity_guard():
    """High velocity should override low dispersion."""
    df = pd.DataFrame({
        "qc_flag": ["ok", "ok"],
        "mouse_roi_valid": [True, True],
        "centroid_x_roll_dispersion": [2.0, 2.0],
        "velocity_smooth_px_s": [1.0, 50.0],
    })
    stat = classify_stationary(df, disp_thresh=5.0, vel_thresh=10.0)
    assert stat.iloc[0] == True
    assert stat.iloc[1] == False


def test_classify_frame_state_three_way():
    df = pd.DataFrame({
        "qc_flag": ["ok", "ok", "no_mouse_roi", "ok"],
        "centroid_x_roll_dispersion": [2.0, 8.0, np.nan, 2.0],
        "velocity_smooth_px_s": [1.0, 1.0, np.nan, 1.0],
    })
    state = classify_frame_state(df, disp_thresh=5.0, vel_thresh=10.0)
    assert list(state) == ["stationary", "moving", "unknown", "stationary"]


def _make_state_and_time(pattern, fps=9.0):
    """
    pattern: list of (state, duration_sec) where state in
    {"stationary", "moving", "unknown"}. Returns (stationary_mask,
    frame_state, elapsed_time_sec).
    """
    samples = []
    t = 0.0
    dt = 1.0 / fps
    for state, dur in pattern:
        n = int(dur * fps)
        for _ in range(n):
            samples.append((state, t))
            t += dt
    frame_state = pd.Series([s for s, _ in samples])
    mask = frame_state == "stationary"
    elapsed = pd.Series([t for _, t in samples])
    return mask, frame_state, elapsed


class TestRleBoutsFrameStateAwareMerging:
    def test_default_behavior_unchanged_when_frame_state_omitted(self):
        """No frame_state/max_unknown_gap_sec passed -> identical to the original API."""
        mask, t = _make_mask_and_time([(False, 5), (True, 10), (False, 5), (True, 10), (False, 5)])
        bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0)
        assert len(bouts) == 2

    def test_brief_unknown_gap_bridges_a_real_bout(self):
        """A short no_mouse_roi-style dropout (all 'unknown') between two
        stationary runs should merge under max_unknown_gap_sec even though
        it exceeds the tighter max_gap_sec."""
        mask, state, t = _make_state_and_time([
            ("moving", 5),
            ("stationary", 10),
            ("unknown", 4.0),   # exceeds max_gap_sec=2.0, within max_unknown_gap_sec=10.0
            ("stationary", 10),
            ("moving", 5),
        ])
        bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0, frame_state=state, max_unknown_gap_sec=10.0)
        assert len(bouts) == 1
        assert bouts.iloc[0]["bout_duration_sec"] == pytest.approx(24.0, abs=1.0)

    def test_confirmed_moving_gap_never_bridges_even_if_short(self):
        """A gap containing real 'moving' evidence must NOT bridge under
        max_unknown_gap_sec, even if it's shorter than max_unknown_gap_sec --
        only all-unknown gaps get the more permissive tolerance."""
        mask, state, t = _make_state_and_time([
            ("stationary", 10),
            ("moving", 3.0),   # real motion, within max_unknown_gap_sec=10.0 but NOT all-unknown
            ("stationary", 10),
        ])
        bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0, frame_state=state, max_unknown_gap_sec=10.0)
        assert len(bouts) == 2

    def test_mixed_gap_with_any_moving_sample_uses_tight_tolerance(self):
        """A gap that's mostly unknown but contains even one 'moving' sample
        must use max_gap_sec, not max_unknown_gap_sec."""
        mask, state, t = _make_state_and_time([
            ("stationary", 10),
            ("unknown", 2.0),
            ("moving", 1 / 9.0),  # single moving sample
            ("unknown", 2.0),
            ("stationary", 10),
        ])
        bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0, frame_state=state, max_unknown_gap_sec=10.0)
        assert len(bouts) == 2

    def test_unknown_gap_beyond_max_unknown_gap_sec_still_not_bridged(self):
        """An all-unknown gap longer than max_unknown_gap_sec itself should
        not be bridged -- the tolerance is real, not infinite."""
        mask, state, t = _make_state_and_time([
            ("stationary", 10),
            ("unknown", 15.0),  # exceeds max_unknown_gap_sec=10.0
            ("stationary", 10),
        ])
        bouts = rle_bouts(mask, t, min_bout_sec=5.0, max_gap_sec=2.0, frame_state=state, max_unknown_gap_sec=10.0)
        assert len(bouts) == 2
