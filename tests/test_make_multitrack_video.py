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


def test_roi_at_uses_latest_past_sample_only():
    import pandas as pd
    trk = pd.DataFrame({
        "elapsed_time_sec": [10.0, 11.0],
        "mouse_roi_center_x": [100.0, 200.0],
        "mouse_roi_center_y": [5.0, 6.0],
        "mouse_roi_radius_px": [6.0, 6.0],
    })
    assert mmv.roi_at(trk, "mouse_roi", 9.9) is None
    x, _, _, age = mmv.roi_at(trk, "mouse_roi", 10.8)  # nearest would be 11.0; must stay at 10.0
    assert x == 100.0 and abs(age - 0.8) < 1e-9
    x, _, _, age = mmv.roi_at(trk, "mouse_roi", 11.0)
    assert x == 200.0 and age == 0.0
