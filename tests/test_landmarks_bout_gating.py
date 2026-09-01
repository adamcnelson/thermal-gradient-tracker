"""Tests for src/landmarks/bout_gating.py — no real data required."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.bout_gating import filter_bouts_after_entry, iter_stationary_bouts, select_bout_frames


def _tracking_df():
    return pd.DataFrame(
        {
            "video_file": ["a.seq"] * 6 + ["b.seq"] * 4,
            "frame_number": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
            "elapsed_time_sec": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0],
            "qc_flag": ["ok", "ok", "no_mouse_roi", "ok", "ok", "ok", "ok", "ok", "ok", "ok"],
            "mouse_roi_valid": [True, True, False, True, True, True, True, True, True, True],
        }
    )


def _bouts_df():
    return pd.DataFrame(
        {
            "video_file": ["a.seq", "a.seq", "b.seq"],
            "bout_index": [0, 1, 0],
            "bout_start_sec": [0.0, 3.0, 0.0],
            "bout_end_sec": [2.0, 5.0, 3.0],
        }
    )


class TestFilterBoutsAfterEntry:
    def test_drops_bouts_starting_before_entry_time(self):
        bouts = _bouts_df()  # bout_start_sec: 0.0, 3.0, 0.0
        filtered = filter_bouts_after_entry(bouts, entry_time_sec=2.5)
        assert list(filtered["bout_start_sec"]) == [3.0]

    def test_keeps_bout_exactly_at_entry_time(self):
        bouts = _bouts_df()
        filtered = filter_bouts_after_entry(bouts, entry_time_sec=3.0)
        assert list(filtered["bout_start_sec"]) == [3.0]

    def test_entry_time_before_all_bouts_keeps_everything(self):
        bouts = _bouts_df()
        filtered = filter_bouts_after_entry(bouts, entry_time_sec=-1.0)
        assert len(filtered) == len(bouts)

    def test_reindexes_after_filtering(self):
        bouts = _bouts_df()
        filtered = filter_bouts_after_entry(bouts, entry_time_sec=2.5)
        assert list(filtered.index) == [0]


class TestSelectBoutFrames:
    def test_selects_only_rows_in_window_for_matching_video(self):
        bout = _bouts_df().iloc[0]  # a.seq, [0, 2] sec — contains the qc_flag failure at t=2
        frames = select_bout_frames(_tracking_df(), bout)
        # frame at t=2.0 is dropped: qc_flag == "no_mouse_roi"
        assert list(frames["frame_number"]) == [0, 1]

    def test_window_boundaries_are_inclusive(self):
        bout = _bouts_df().iloc[1]  # a.seq, [3, 5] sec
        frames = select_bout_frames(_tracking_df(), bout)
        assert list(frames["elapsed_time_sec"]) == [3.0, 4.0, 5.0]

    def test_does_not_cross_contaminate_between_videos(self):
        bout = _bouts_df().iloc[2]  # b.seq, [0, 3] sec — a.seq has frames in this range too
        frames = select_bout_frames(_tracking_df(), bout)
        assert (frames["video_file"] == "b.seq").all()
        assert len(frames) == 4

    def test_drops_invalid_roi_rows_inside_an_otherwise_valid_window(self):
        bout = pd.Series({"video_file": "a.seq", "bout_start_sec": 0.0, "bout_end_sec": 5.0})
        frames = select_bout_frames(_tracking_df(), bout)
        assert (frames["qc_flag"] == "ok").all()
        assert frames["mouse_roi_valid"].all()
        assert 2 not in list(frames["frame_number"])  # the no_mouse_roi frame


class TestIterStationaryBouts:
    def test_yields_one_entry_per_bout_with_valid_frames(self):
        results = list(iter_stationary_bouts(_tracking_df(), _bouts_df()))
        assert len(results) == 3
        bout_indices = [(b["video_file"], b["bout_index"]) for b, _frames in results]
        assert bout_indices == [("a.seq", 0), ("a.seq", 1), ("b.seq", 0)]

    def test_skips_bouts_with_zero_valid_frames(self):
        tracking = _tracking_df()
        # Make every a.seq frame invalid so the first two bouts have nothing to yield
        tracking.loc[tracking["video_file"] == "a.seq", "qc_flag"] = "no_mouse"
        results = list(iter_stationary_bouts(tracking, _bouts_df()))
        assert len(results) == 1
        assert results[0][0]["video_file"] == "b.seq"
