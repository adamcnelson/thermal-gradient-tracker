"""
Stage 2 — webcam preprocessing (project_brief_v7.md §6 Stage 2).

Splits a combined-frame webcam recording into per-track crops and joins
each track to its session/mouse metadata via LUT_CLEAN_July6.csv.

B/F row-assignment (brief: "Verify the B/F row assignment empirically on
the first session rather than trusting the convention"): confirmed by
Adam (2026-08-12) — in the webcam videos, the top crop is always Back,
the bottom crop is always Front. Lane "B" is the upper crop, "F" is the
lower crop, and Lane letter maps directly to the cropped .seq filename's
trailing "_Back"/"_Front" token (also confirmed self-consistent against
real LUT rows for all 3 sessions we have local .seq data for).
save_track_split_preview() below still exists as a one-off visual sanity
check tool, e.g. for a new rig configuration.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Per brief §4: two tracks per frame, Lane "B" (Back) on top, "F" (Front) below.
LANE_TOP = "B"
LANE_BOTTOM = "F"


def split_track_crops(
    frame: np.ndarray, split_row: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split one combined webcam frame into (crop_b_top, crop_f_bottom).

    split_row : pixel row where the bottom crop begins. Defaults to the
        vertical midpoint — but confirmed WRONG for the real rig (2026-08-13,
        Test_3 session): the two tracks are not symmetrically centered
        (much more black background/ceiling above the top track than below
        the bottom track), so height//2 falls inside the top track itself
        rather than in the gap between the two tracks. This leaks a chunk
        of the top track's mouse into what's supposed to be an isolated
        bottom-track crop. Pass detect_track_split_row(frame)'s result
        explicitly rather than relying on this default for real footage.
    """
    height = frame.shape[0]
    if split_row is None:
        split_row = height // 2
    if not 0 < split_row < height:
        raise ValueError(f"split_row {split_row} out of bounds for frame height {height}")
    return frame[:split_row], frame[split_row:]


def detect_track_split_row(frame: np.ndarray) -> int:
    """
    Detect the true row boundary between the two tracks from brightness
    alone: each track shows up as a bright horizontal band (the plate),
    against a much darker background above/below/between them. Finds the
    brightest row in each half of the frame (one track per half, assuming
    the split falls roughly in the right half to begin with) as bracketing
    peaks, then returns the darkest row strictly between them — the real
    gap, not necessarily height // 2.

    A naive "find the single darkest row in the frame" was tried first and
    rejected: the background above the top track is even darker than the
    inter-track gap, so an unbracketed search locks onto that instead.
    Peak-bracketing avoids it. Confirmed stable (within a few px) across 5
    real frames spanning a full session (2026-08-13).
    """
    height = frame.shape[0]
    if height < 4:
        raise ValueError(f"Frame too small ({height} rows) to detect a track split")
    row_mean = frame.astype(np.float64).mean(axis=1)
    mid = height // 2
    # peak_top is always in [0, mid) and peak_bottom always in [mid, height) by
    # construction, so they can never coincide/invert — no need to guard that.
    peak_top = int(np.argmax(row_mean[:mid]))
    peak_bottom = mid + int(np.argmax(row_mean[mid:]))
    if row_mean[peak_top] - row_mean.min() < 1e-6 and row_mean[peak_bottom] - row_mean.min() < 1e-6:
        raise ValueError("Frame has no brightness contrast — cannot locate two track bands")
    between = row_mean[peak_top:peak_bottom]
    return peak_top + int(np.argmin(between))


def save_track_split_preview(frame: np.ndarray, split_row: Optional[int], output_path: str) -> None:
    """
    Save a copy of `frame` with the B/top-F/bottom split boundary drawn on
    it, for a human to visually confirm on the first real session — this
    is the empirical check the brief asks for; it is not automated here
    because animal identity isn't derivable from pixels alone.
    """
    import cv2

    height = frame.shape[0]
    row = split_row if split_row is not None else height // 2
    annotated = frame.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    cv2.line(annotated, (0, row), (annotated.shape[1], row), (0, 0, 255), 2)
    cv2.putText(annotated, "B (top)", (5, max(row - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(annotated, "F (bottom)", (5, min(row + 25, annotated.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def session_stem_from_seq_path(seq_path: str) -> str:
    """
    Strip the trailing "_Front"/"_Back" track token from a cropped .seq
    filename stem, to get the session key LUT_CLEAN_July6.csv's
    Video_name_SEQ column uses (confirmed against real rows: e.g.
    "07-28-25_4540_B_4541_F_Test3-004_Front" -> "..._Test3-004").
    """
    stem = Path(seq_path).stem
    for suffix in ("_Front", "_Back"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    raise ValueError(f"Expected a '_Front' or '_Back' suffix on .seq stem: {stem}")


def lookup_session_rows(seq_path: str, lut_df: pd.DataFrame) -> pd.DataFrame:
    """Return the LUT_CLEAN_July6.csv rows (one per mouse/lane) for this session."""
    session_stem = session_stem_from_seq_path(seq_path)
    rows = lut_df[lut_df["Video_name_SEQ"] == session_stem]
    if rows.empty:
        raise ValueError(f"No LUT rows found for session '{session_stem}' (from {seq_path})")
    return rows


def lane_for_seq_path(seq_path: str, lut_df: pd.DataFrame) -> dict:
    """
    Return {"mouse_id": ..., "lane": "B"|"F"} for the animal whose track
    this specific cropped .seq file (identified by its _Front/_Back
    suffix) is expected to contain, per the Lane<->suffix convention
    documented at module level.
    """
    stem = Path(seq_path).stem
    expected_lane = LANE_TOP if stem.endswith("_Back") else LANE_BOTTOM
    rows = lookup_session_rows(seq_path, lut_df)
    match = rows[rows["Lane"] == expected_lane]
    if len(match) != 1:
        raise ValueError(
            f"Expected exactly one LUT row with Lane={expected_lane!r} for {seq_path}, "
            f"found {len(match)}"
        )
    row = match.iloc[0]
    return {"mouse_id": int(row["Mouse_ID"]), "lane": expected_lane}


def iter_track_crop_frames(
    video_path: str, lane: str, split_row: Optional[int] = None, grayscale: bool = True
) -> Iterator[np.ndarray]:
    """
    Yield frames from one webcam video, already cropped to the requested
    lane's track ("B" = top, "F" = bottom — see module docstring). Used to
    feed src/landmarks/sync.py's rgb_motion_energy_from_frames(), which
    must only ever see the track-matched crop, never the full frame.
    """
    if lane not in (LANE_TOP, LANE_BOTTOM):
        raise ValueError(f"lane must be 'B' or 'F', got {lane!r}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(frame, split_row=split_row)
            yield top if lane == LANE_TOP else bottom
    finally:
        cap.release()
