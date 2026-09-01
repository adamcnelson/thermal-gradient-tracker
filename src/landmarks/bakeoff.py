"""
Stage 5 bake-off metrics (project_brief_v7.md §6 Stage 5 "Bake-off protocol").

Evaluates a landmark-extraction method against a human-labeled validation
frame (validation_labels/*.json, produced by scripts/label_validation_frame.py)
for the brief's first two bake-off metrics:
  1. Landmark pixel error in RGB
  2. Thermal-value error vs. human-drawn ROI

Metric #3 (yield stratified by gradient zone) is NOT computed here — it's
a property of a full session's worth of bouts, not a single labeled
frame, and already lives in outputs.compute_landmark_yield_by_zone(),
run against real Stage 7 batch output.

IMPORTANT — this is not yet a real "bake-off" in the brief's literal
sense ("evaluate classical vs. supervised"): no supervised-pose fallback
(SLEAP/DeepLabCut) has been built (see Stage 5 status memory), so there
is only one method to evaluate. What this module CAN do is measure the
classical method's own accuracy against real human ground truth — useful
and real, just not yet a comparison.

Coordinate convention note (verified against scripts/label_validation_frame.py's
onclick handler, 2026-08-25): every point in a label JSON — head_point,
tail_base_point, mask_polygon, tail_centerline, warm_spot_roi_polygon_thermal
— is stored as [x, y] i.e. [column, row] (matplotlib's xdata/ydata,
captured directly). This project's own internal landmark representation
(rgb_landmarks.Point) is (row, col) = (y, x) throughout. Every function
here that accepts both a label point and an algorithmic point takes them
in their OWN native convention and converts internally — callers should
not pre-convert.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .rgb_landmarks import Point  # (row, col) = (y, x)

XYPoint = Tuple[float, float]  # (x, y) = (col, row) — the label JSON convention


def _algo_point_to_xy(p: Optional[Point]) -> Optional[XYPoint]:
    """Convert this project's internal (row, col) landmark convention to
    the label JSON's (x, y) convention."""
    if p is None:
        return None
    row, col = p
    return (float(col), float(row))


def pixel_distance(a: Optional[XYPoint], b: Optional[XYPoint]) -> Optional[float]:
    """Euclidean distance between two (x, y) points, or None if either is missing."""
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


@dataclass
class LandmarkPixelError:
    label_name: str
    posture: Optional[str]
    nose_error_px: Optional[float]
    head_error_px: Optional[float]
    tail_base_error_px: Optional[float]
    reason: str = ""
    caveat: str = ""


def compute_landmark_pixel_error(
    label_name: str,
    posture: Optional[str],
    label_head_point_xy: Optional[Sequence[float]],
    label_nose_point_xy: Optional[Sequence[float]],
    label_tail_base_point_xy: Optional[Sequence[float]],
    algo_nose_point: Optional[Point],
    algo_tail_base_point: Optional[Point],
) -> LandmarkPixelError:
    """
    Brief bake-off metric #1: landmark pixel error in RGB.

    label_head_point_xy / label_nose_point_xy / label_tail_base_point_xy
    come directly from a validation_labels/*.json file (already in [x, y]
    form). algo_nose_point / algo_tail_base_point come from this project's
    own rgb_landmarks.extract_mouse_detection() output (in (row, col)
    form) — None for curled/ambiguous postures, where no landmarks exist
    to compare at all (a real, expected absence, not missing data).

    "head" and "nose" are DELIBERATELY SEPARATE landmarks here (fixed
    2026-08-25 — a real, direct visual comparison against the project's
    first 3 labels, which predate the nose-tip labeling mode, found the
    algorithm's own nose_point sitting right at the true silhouette tip
    every time, while human head_point clicks landed further back toward
    the center of the head mass — a genuine labeling-convention
    difference, not an algorithm error). nose_error_px is the real,
    apples-to-apples comparison and should be preferred once a label has
    a real nose_point (scripts/label_validation_frame.py's "n" mode, added
    2026-08-25). head_error_px is kept for continuity with older labels
    that only have head_point, but is NOT the same comparison — see
    `caveat`, which is set whenever a label has no real nose_point and
    head_error_px is standing in for it.
    """
    algo_nose_xy = _algo_point_to_xy(algo_nose_point)
    algo_tail_xy = _algo_point_to_xy(algo_tail_base_point)
    label_head_xy = tuple(label_head_point_xy) if label_head_point_xy is not None else None
    label_nose_xy = tuple(label_nose_point_xy) if label_nose_point_xy is not None else None
    label_tail_xy = tuple(label_tail_base_point_xy) if label_tail_base_point_xy is not None else None

    nose_err = pixel_distance(label_nose_xy, algo_nose_xy)
    head_err = pixel_distance(label_head_xy, algo_nose_xy)
    tail_err = pixel_distance(label_tail_xy, algo_tail_xy)

    reason = ""
    if algo_nose_point is None and algo_tail_base_point is None:
        reason = f"no algorithmic landmarks available (posture={posture})"

    caveat = ""
    if label_nose_xy is None and label_head_xy is not None:
        caveat = (
            "no real nose-tip label on this frame (predates the 2026-08-25 nose-tip mode) — "
            "head_error_px compares against head_point, a different, less precise landmark, "
            "not a true nose-tip comparison"
        )

    return LandmarkPixelError(
        label_name=label_name, posture=posture,
        nose_error_px=nose_err, head_error_px=head_err, tail_base_error_px=tail_err,
        reason=reason, caveat=caveat,
    )


def polygon_mask_xy(polygon_xy: Sequence[Sequence[float]], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a human-drawn polygon (list of [x, y] points, the label
    JSON's own convention) into a boolean mask of the given (height, width)
    shape. Uses cv2.fillPoly directly — no coordinate conversion needed,
    since cv2 points are already (x, y). Fewer than 3 points can't enclose
    an area, so returns an all-False mask rather than calling into
    cv2.fillPoly (which raises on an empty/degenerate point array)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon_xy) < 3:
        return mask.astype(bool)
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


@dataclass
class ThermalValueError:
    label_name: str
    posture: Optional[str]
    human_roi_pixel_count: int
    human_roi_mean_c: Optional[float]
    human_roi_p95_c: Optional[float]
    algo_warm_spot_c: Optional[float]
    error_c: Optional[float]
    reason: str = ""


def compute_thermal_value_error(
    label_name: str,
    posture: Optional[str],
    thermal_celsius: np.ndarray,
    human_roi_polygon_xy: Optional[Sequence[Sequence[float]]],
    algo_warm_spot_c: Optional[float],
    percentile: float = 95.0,
) -> ThermalValueError:
    """
    Brief bake-off metric #2: thermal-value error vs. human-drawn ROI.

    Compares this project's algorithmic warm-spot measurement
    (thermal_measurement.warm_spot_temperature(), a 95th-percentile
    sample within the RGB-derived, homography-warped anterior region) to
    the same percentile computed directly within the human-drawn
    warm_spot_roi_polygon_thermal — the human ROI is ground truth here,
    not a second algorithm, so this measures the whole RGB-landmark ->
    warp -> sample pipeline's real-world accuracy in one number.

    error_c is None whenever algo_warm_spot_c is None (curled/ambiguous
    posture, or a warp that landed fully outside the thermal frame) —
    there is nothing to compare in that case, not a zero error.
    """
    if human_roi_polygon_xy is None or len(human_roi_polygon_xy) < 3:
        return ThermalValueError(
            label_name=label_name, posture=posture, human_roi_pixel_count=0,
            human_roi_mean_c=None, human_roi_p95_c=None, algo_warm_spot_c=algo_warm_spot_c,
            error_c=None, reason="no human warm-spot ROI drawn for this label",
        )

    mask = polygon_mask_xy(human_roi_polygon_xy, thermal_celsius.shape)
    count = int(np.sum(mask))
    if count == 0:
        return ThermalValueError(
            label_name=label_name, posture=posture, human_roi_pixel_count=0,
            human_roi_mean_c=None, human_roi_p95_c=None, algo_warm_spot_c=algo_warm_spot_c,
            error_c=None, reason="human ROI polygon rasterized to zero pixels",
        )

    vals = thermal_celsius[mask]
    human_mean = float(np.mean(vals))
    human_p95 = float(np.percentile(vals, percentile))
    error = (algo_warm_spot_c - human_p95) if algo_warm_spot_c is not None else None
    reason = "" if algo_warm_spot_c is not None else f"no algorithmic warm-spot available (posture={posture})"

    return ThermalValueError(
        label_name=label_name, posture=posture, human_roi_pixel_count=count,
        human_roi_mean_c=human_mean, human_roi_p95_c=human_p95,
        algo_warm_spot_c=algo_warm_spot_c, error_c=error, reason=reason,
    )
