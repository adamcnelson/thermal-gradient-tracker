"""Tests for src/landmarks/video_io.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.video_io import SequentialFrameReader

N_FRAMES = 60


@pytest.fixture
def indexed_video(tmp_path):
    """Frame i is a flat image of intensity 4*i (lossy codec, so compare approximately)."""
    path = str(tmp_path / "indexed.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), 4 * i, np.uint8))
    writer.release()
    return path


def _frame_index(frame):
    return int(round(frame.mean() / 4))


def test_reads_requested_frames_in_forward_order(indexed_video):
    reader = SequentialFrameReader(indexed_video)
    for idx in [0, 1, 7, 7 + 1, 30, 59]:
        ok, frame = reader.read(idx)
        assert ok and _frame_index(frame) == idx
    reader.release()


def test_backward_request_reopens_and_stays_correct(indexed_video):
    reader = SequentialFrameReader(indexed_video)
    assert _frame_index(reader.read(40)[1]) == 40
    assert _frame_index(reader.read(10)[1]) == 10
    reader.release()


def test_past_end_returns_not_ok(indexed_video):
    reader = SequentialFrameReader(indexed_video)
    ok, frame = reader.read(N_FRAMES + 5)
    assert not ok and frame is None
    reader.release()
