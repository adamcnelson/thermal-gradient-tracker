"""Tests for src/landmarks/webcam_preprocessing.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.webcam_preprocessing import (
    detect_track_split_row,
    iter_track_crop_frames,
    lane_for_seq_path,
    lookup_session_rows,
    session_stem_from_seq_path,
    split_track_crops,
)


class TestDetectTrackSplitRow:
    def _two_track_frame(self, h=300, w=100, top_center=140, bottom_center=230, sigma=15):
        """
        Two smooth Gaussian-shaped bright bands (tracks), like the real
        rig's rounded brightness profile — a flat-topped rectangular band
        was tried first and rejected: with no falloff, the naive
        height//2 split can tie between "still inside the first band" and
        "already at peak of the second," which real (non-flat) brightness
        profiles don't do.
        """
        rows = np.arange(h)
        top_band = 200 * np.exp(-0.5 * ((rows - top_center) / sigma) ** 2)
        bottom_band = 200 * np.exp(-0.5 * ((rows - bottom_center) / sigma) ** 2)
        profile = 20 + top_band + bottom_band
        return np.tile(profile[:, None], (1, w))

    def test_finds_true_gap_between_the_two_peaks(self):
        frame = self._two_track_frame()
        gap = detect_track_split_row(frame)
        assert 180 <= gap <= 190  # true trough sits at row ~185

    def test_naive_midpoint_would_have_been_wrong_here(self):
        """Sanity check that this fixture actually exercises the bug: the
        naive height//2 split lands inside the top band, not the gap."""
        frame = self._two_track_frame()
        naive_midpoint = frame.shape[0] // 2  # 150
        gap = detect_track_split_row(frame)
        assert abs(gap - naive_midpoint) > 30

    def test_stable_across_horizontal_shift(self):
        """Gap detection uses row means, so it shouldn't care where content sits horizontally."""
        frame = self._two_track_frame(w=200)
        frame = np.roll(frame, shift=50, axis=1)
        gap = detect_track_split_row(frame)
        assert 180 <= gap <= 190

    def test_flat_frame_raises(self):
        flat = np.full((100, 50), 100.0)
        with pytest.raises(ValueError):
            detect_track_split_row(flat)

    def test_tiny_frame_raises(self):
        with pytest.raises(ValueError):
            detect_track_split_row(np.zeros((2, 10)))


class TestSplitTrackCrops:
    def test_default_split_is_vertical_midpoint(self):
        frame = np.arange(40 * 10).reshape(40, 10)
        top, bottom = split_track_crops(frame)
        assert top.shape == (20, 10)
        assert bottom.shape == (20, 10)
        np.testing.assert_array_equal(top, frame[:20])
        np.testing.assert_array_equal(bottom, frame[20:])

    def test_explicit_split_row(self):
        frame = np.arange(30 * 5).reshape(30, 5)
        top, bottom = split_track_crops(frame, split_row=12)
        assert top.shape == (12, 5)
        assert bottom.shape == (18, 5)

    def test_out_of_bounds_split_row_raises(self):
        frame = np.zeros((30, 5))
        with pytest.raises(ValueError):
            split_track_crops(frame, split_row=30)
        with pytest.raises(ValueError):
            split_track_crops(frame, split_row=0)


class TestSessionStemAndLutJoin:
    def _lut(self):
        return pd.DataFrame(
            {
                "Mouse_ID": [4540, 4541, 4548, 4551],
                "Lane": ["B", "F", "B", "F"],
                "Video_name_SEQ": [
                    "07-28-25_4540_B_4541_F_Test3-004",
                    "07-28-25_4540_B_4541_F_Test3-004",
                    "08-07-25_4551_F_4548_B_Test7-019",
                    "08-07-25_4551_F_4548_B_Test7-019",
                ],
            }
        )

    def test_session_stem_strips_track_suffix(self):
        assert (
            session_stem_from_seq_path("07-28-25_4540_B_4541_F_Test3-004_Front.seq")
            == "07-28-25_4540_B_4541_F_Test3-004"
        )
        assert (
            session_stem_from_seq_path("/some/dir/x_Back.seq")
            == "x"
        )

    def test_session_stem_requires_known_suffix(self):
        with pytest.raises(ValueError):
            session_stem_from_seq_path("no_track_suffix_here.seq")

    def test_lookup_session_rows_returns_both_mice(self):
        rows = lookup_session_rows(
            "07-28-25_4540_B_4541_F_Test3-004_Front.seq", self._lut()
        )
        assert sorted(rows["Mouse_ID"]) == [4540, 4541]

    def test_lookup_session_rows_missing_session_raises(self):
        with pytest.raises(ValueError):
            lookup_session_rows("09-01-25_9999_B_9998_F_TestX-000_Front.seq", self._lut())

    def test_lane_for_front_file_is_lane_f_mouse(self):
        result = lane_for_seq_path(
            "07-28-25_4540_B_4541_F_Test3-004_Front.seq", self._lut()
        )
        assert result == {"mouse_id": 4541, "lane": "F"}

    def test_lane_for_back_file_is_lane_b_mouse(self):
        result = lane_for_seq_path(
            "07-28-25_4540_B_4541_F_Test3-004_Back.seq", self._lut()
        )
        assert result == {"mouse_id": 4540, "lane": "B"}

    def test_second_session_front_back(self):
        front = lane_for_seq_path(
            "08-07-25_4551_F_4548_B_Test7-019_Front.seq", self._lut()
        )
        back = lane_for_seq_path(
            "08-07-25_4551_F_4548_B_Test7-019_Back.seq", self._lut()
        )
        assert front == {"mouse_id": 4551, "lane": "F"}
        assert back == {"mouse_id": 4548, "lane": "B"}


REAL_LUT_PATH = Path("metadata/LUT_CLEAN_July6.csv")


@pytest.mark.skipif(not REAL_LUT_PATH.exists(), reason="Real LUT not present on this machine")
class TestRealLutJoin:
    """Confirms the join key and Lane<->suffix mapping against the real LUT
    for every session we have local .seq data for, not just a synthetic stand-in."""

    @pytest.mark.parametrize(
        "seq_stem,expected_front,expected_back",
        [
            ("07-28-25_4540_B_4541_F_Test3-004", 4541, 4540),
            ("08-07-25_4551_F_4548_B_Test7-019", 4551, 4548),
            ("08-07-25_4541_F_4540_B_Test7-020", 4541, 4540),
        ],
    )
    def test_real_sessions(self, seq_stem, expected_front, expected_back):
        lut = pd.read_csv(REAL_LUT_PATH)
        front = lane_for_seq_path(f"{seq_stem}_Front.seq", lut)
        back = lane_for_seq_path(f"{seq_stem}_Back.seq", lut)
        assert front["mouse_id"] == expected_front
        assert back["mouse_id"] == expected_back


def _mjpg_writer_available() -> bool:
    """MJPG/.avi is used for the synthetic test video below (not .mp4 —
    mp4v/H.264 writer support is inconsistent across opencv-python builds;
    MJPG/.avi is what actually opens for writing in this environment).
    iter_track_crop_frames() itself is container-agnostic — it only reads."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter("/tmp/_mjpg_probe.avi", fourcc, 10.0, (4, 4))
    ok = vw.isOpened()
    vw.release()
    return ok


@pytest.mark.skipif(not _mjpg_writer_available(), reason="No working video writer codec in this environment")
class TestIterTrackCropFrames:
    def _write_synthetic_video(self, tmp_path, n_frames=5, h=30, w=20):
        path = str(tmp_path / "synthetic.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[: h // 2] = (50, 50, 50)  # top half = "B" track
            frame[h // 2 :] = (200, 200, 200)  # bottom half = "F" track
            frame[0, i % w] = (255, 0, 0)  # per-frame marker
            vw.write(frame)
        vw.release()
        return path, h, w

    def test_lane_b_yields_top_half(self, tmp_path):
        path, h, w = self._write_synthetic_video(tmp_path)
        frames = list(iter_track_crop_frames(path, "B", grayscale=False))
        assert len(frames) == 5
        assert frames[0].shape == (h // 2, w, 3)
        assert frames[0].mean() == pytest.approx(50, abs=5)

    def test_lane_f_yields_bottom_half(self, tmp_path):
        path, h, w = self._write_synthetic_video(tmp_path)
        frames = list(iter_track_crop_frames(path, "F", grayscale=False))
        assert len(frames) == 5
        assert frames[0].shape == (h - h // 2, w, 3)
        assert frames[0].mean() == pytest.approx(200, abs=5)

    def test_grayscale_default_reduces_channels(self, tmp_path):
        path, _h, _w = self._write_synthetic_video(tmp_path)
        frames = list(iter_track_crop_frames(path, "B"))
        assert frames[0].ndim == 2

    def test_invalid_lane_raises(self, tmp_path):
        path, _h, _w = self._write_synthetic_video(tmp_path)
        with pytest.raises(ValueError):
            list(iter_track_crop_frames(path, "X"))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            list(iter_track_crop_frames("/no/such/video.avi", "B"))
