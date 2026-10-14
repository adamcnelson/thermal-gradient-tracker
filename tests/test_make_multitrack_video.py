import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "make_multitrack_video",
    Path(__file__).resolve().parent.parent / "scripts" / "make_multitrack_video.py",
)
mmv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmv)


def test_frame_offset_handles_distinct_first_frame():
    assert mmv.frame_offset(0, 1000, 800) == 0
    assert mmv.frame_offset(1, 1000, 800) == 1000
    assert mmv.frame_offset(5, 1000, 800) == 1000 + 4 * 800


def test_trace_up_to_interpolates_cursor_point():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])
    tt, yy = mmv.trace_up_to(t, y, 1.5)
    np.testing.assert_allclose(tt, [0.0, 1.0, 1.5])
    np.testing.assert_allclose(yy, [0.0, 10.0, 15.0])


def test_trace_up_to_before_first_sample_is_empty():
    tt, yy = mmv.trace_up_to(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.5)
    assert len(tt) == 0 and len(yy) == 0
