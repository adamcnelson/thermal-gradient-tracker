"""
Session entry-time detection (2026-08-24; not literally spelled out in
project_brief_v7.md, but a direct consequence of trusting Stage 1's bout
output — this closes a real gap that surfaced during Stage 5 QC review).

Both existing per-modality trackers (src/mouse_segmentation.py for
thermal, src/landmarks/rgb_landmarks.py for RGB) segment by looking for a
blob whose AREA falls in a real-mouse-sized range. The human hand/forearm
that drops the mouse into the arena is, in both modalities, dramatically
larger than that range — a real, sharp, unambiguous, one-time event per
session.

Cross-checking Test_3 directly (2026-08-24) found this matters: the
existing thermal tracker's own pre_entry->ok transition (qc_flag) fires at
thermal-clock 47.5s, but the RGB video shows the arena is empty until a
gloved hand/arm sweeps in at RGB-clock ~33s and drops the mouse. Combined
with the independently well-established Test_3 sync offset (thermal leads
RGB by ~93-96s), those two facts are inconsistent by roughly 80 seconds —
strong evidence the existing thermal tracker's own entry marker fires
before the animal is actually present for at least this session. This
module detects the real intrusion event directly, in each modality's own
native data, rather than trusting either tracker's internal QC heuristic.

Two things it's used for:
1. A DIRECT per-modality entry-time estimate, used to filter out spurious
   pre-entry bouts (see bout_gating.filter_bouts_after_entry()) without
   depending on the cross-modality sync offset at all.
2. Optionally, a NEW sync anchor: if the same physical intrusion event is
   detected in both modalities, the two entry times directly give an
   independent single-point offset estimate (rgb_entry_time -
   thermal_entry_time), a useful cross-check against the motion-energy and
   position-anchor methods in sync.py — the hand/arm is a much larger,
   sharper signal than anything else available.
"""

from typing import List, Optional, Sequence

import numpy as np

FrameState = str  # "empty" | "mouse" | "intrusion"


def classify_frame_state_rgb(
    frame: np.ndarray,
    background_model,  # rgb_landmarks.RgbBackgroundModel
    min_area: int,
    max_area: int,
    intrusion_area_multiple: float = 3.0,
) -> FrameState:
    """
    Classify one RGB (track-matched-crop) frame as "empty" (no foreground
    blob), "mouse" (a real, plausible mouse-scale detection), or
    "intrusion" (a blob far too large to be the mouse itself — the
    hand/forearm). Widens segment_mouse_rgb's own max_area bound rather
    than narrowing it, specifically to let the oversized intrusion blob
    through so it can be recognized rather than silently discarded.
    """
    from .rgb_landmarks import segment_mouse_rgb, classify_mouse_blob

    mask = segment_mouse_rgb(
        frame, background_model, min_area=min_area, max_area=int(max_area * intrusion_area_multiple)
    )
    if mask is None:
        return "empty"
    area = int(np.sum(mask))
    if area > max_area:
        return "intrusion"
    return "mouse" if classify_mouse_blob(mask, min_area, max_area) is not None else "empty"


def classify_frame_state_thermal(
    frame: np.ndarray,
    background_model,  # mouse_segmentation.BackgroundModel
    min_area: int,
    max_area: int,
    intrusion_area_multiple: float = 3.0,
    threshold_sigma: float = 3.0,
) -> FrameState:
    """Thermal-side counterpart to classify_frame_state_rgb(), same logic against
    src/mouse_segmentation.py's segment_mouse() and its BackgroundModel."""
    from ..mouse_segmentation import segment_mouse

    mask, _centroid, _confidence, _debug = segment_mouse(
        frame, background_model, min_area=min_area,
        max_area=int(max_area * intrusion_area_multiple), threshold_sigma=threshold_sigma,
    )
    if mask is None:
        return "empty"
    area = int(np.sum(mask))
    return "intrusion" if area > max_area else "mouse"


def find_entry_index(states: Sequence[FrameState], min_sustained_detections: int = 5) -> Optional[int]:
    """
    Given a chronological sequence of per-frame states, return the index
    of the first frame of the sustained real "mouse" run that follows the
    MOST RECENT "intrusion" state. Returns None if no intrusion appears in
    the sequence at all, or if no sufficiently sustained "mouse" run
    follows the last one seen (each new intrusion resets the search, since
    a session could in principle show more than one hand/arm event — e.g.
    a brief mid-session adjustment — and only the run following the last
    one represents the animal actually settled and left alone).
    """
    intrusion_seen = False
    run_start: Optional[int] = None
    run_len = 0
    for i, s in enumerate(states):
        if s == "intrusion":
            intrusion_seen = True
            run_start = None
            run_len = 0
            continue
        if not intrusion_seen:
            continue
        if s == "mouse":
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_sustained_detections:
                return run_start
        else:
            run_start = None
            run_len = 0
    return None


def find_entry_frame_index_rgb(
    frames: Sequence[np.ndarray],
    background_model,
    min_area: int,
    max_area: int,
    intrusion_area_multiple: float = 3.0,
    min_sustained_detections: int = 5,
) -> Optional[int]:
    """Full RGB entry-detection pipeline over an ordered sequence of already
    track-matched-crop frames. Returns an index into `frames`, or None."""
    states: List[FrameState] = [
        classify_frame_state_rgb(f, background_model, min_area, max_area, intrusion_area_multiple)
        for f in frames
    ]
    return find_entry_index(states, min_sustained_detections)


def find_entry_frame_index_thermal(
    frames: Sequence[np.ndarray],
    background_model,
    min_area: int,
    max_area: int,
    intrusion_area_multiple: float = 3.0,
    min_sustained_detections: int = 5,
    threshold_sigma: float = 3.0,
) -> Optional[int]:
    """Thermal counterpart to find_entry_frame_index_rgb()."""
    states: List[FrameState] = [
        classify_frame_state_thermal(f, background_model, min_area, max_area, intrusion_area_multiple, threshold_sigma)
        for f in frames
    ]
    return find_entry_index(states, min_sustained_detections)
