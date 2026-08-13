"""
Stage 5 — RGB landmark extraction, classical-CV primary method
(project_brief_v7.md §6 Stage 5).

A black mouse on a white plate is near-solved with a temporal background
model + threshold, per the brief. This module is deliberately decoupled
from src/mouse_segmentation.py (the thermal-side equivalent) rather than
sharing code with it — project_brief_v7.md §3's core principle is that RGB
geometry and thermal radiometry never mix roles, and that separation is
easiest to keep true if the two segmentation implementations don't share a
code path either, even though the underlying algorithm (temporal median
background + adaptive threshold + morphology) is conceptually the same.

Pipeline: segment_mouse_rgb() -> binary mask -> skeletonize -> order into a
nose<->tail path -> width profile via distance transform -> tail-base
landmark (width threshold crossing) -> tail centerline (path from tail
base to tail tip).

The supervised-pose fallback (SLEAP/DeepLabCut/Lightning Pose) named in the
brief for frames where classical CV fails QC is not implemented here —
it's GPU/ARCC-trained per the brief and out of scope until the classical
method's yield is measured against the (not-yet-labeled) validation set.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

Point = Tuple[int, int]  # (row, col) = (y, x), matching numpy array indexing


# ── segmentation (temporal background model, decoupled from the thermal side) ──


class RgbBackgroundModel:
    """Temporal median background over a sample of track-matched-crop RGB frames."""

    def __init__(self, background: np.ndarray):
        self._bg = background.astype(np.float32)

    @property
    def background(self) -> np.ndarray:
        return self._bg

    @classmethod
    def build(cls, frames: Sequence[np.ndarray]) -> "RgbBackgroundModel":
        if len(frames) == 0:
            raise ValueError("Need at least 1 frame to build a background model")
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        return cls(np.median(stack, axis=0))

    def foreground_score(self, frame: np.ndarray) -> np.ndarray:
        return np.abs(frame.astype(np.float32) - self._bg)


def segment_mouse_rgb(
    frame: np.ndarray,
    background_model: RgbBackgroundModel,
    min_area: int,
    max_area: int,
    threshold_sigma: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Segment the mouse from one grayscale RGB (track-matched-crop) frame.
    Returns a bool mask of the single largest qualifying blob, or None.
    """
    fg_score = background_model.foreground_score(frame)
    threshold = float(np.mean(fg_score) + threshold_sigma * np.std(fg_score))
    raw_mask = (fg_score > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    labeled, n_labels = ndimage.label(cleaned)
    if n_labels == 0:
        return None

    best_mask, best_area = None, 0
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        area = int(np.sum(component))
        if min_area <= area <= max_area and area > best_area:
            best_mask, best_area = component, area

    return best_mask


def is_plausible_mouse_blob(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_aspect_ratio: float = 1.8,
) -> bool:
    """
    False-positive filtering (brief: "mandatory, not optional" — fecal boli
    are small, roughly round/compact blobs). Rejects by area and by
    bounding-box aspect ratio (an extended mouse is elongated; a bolus is
    close to isotropic). Track-continuity filtering is a temporal check
    that belongs at the calling/orchestration layer, not here.
    """
    area = int(np.sum(mask))
    if not (min_area <= area <= max_area):
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_idx, col_idx = np.where(rows)[0], np.where(cols)[0]
    height = row_idx[-1] - row_idx[0] + 1
    width = col_idx[-1] - col_idx[0] + 1
    aspect = max(height, width) / max(min(height, width), 1)
    return aspect >= min_aspect_ratio


# ── skeleton path ordering ──────────────────────────────────────────────────


def order_skeleton_path(skeleton: np.ndarray) -> List[Point]:
    """
    Walk a single-pixel-wide skeleton from one endpoint to the other and
    return the ordered list of (row, col) pixels.

    Raises ValueError if the skeleton isn't a simple open path (0 endpoints
    -> a closed loop; >2 endpoints -> branched, e.g. a curled/self-occluded
    animal) — both are legitimate QC-reject cases for the caller, not bugs.
    """
    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, np.ones((3, 3), dtype=np.uint8)) - skel_u8
    endpoints = [tuple(p) for p in np.argwhere(skeleton & (neighbor_count == 1))]

    if len(endpoints) != 2:
        raise ValueError(
            f"Expected a simple open skeleton path (2 endpoints), found {len(endpoints)} "
            "— likely a branched or closed skeleton (curled posture, self-occlusion, or "
            "segmentation artifact)."
        )

    pixels = {tuple(p) for p in np.argwhere(skeleton)}
    start = endpoints[0]
    visited = {start}
    path = [start]
    current = start
    while True:
        y, x = current
        candidates = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
            and (y + dy, x + dx) in pixels
            and (y + dy, x + dx) not in visited
        ]
        if not candidates:
            break
        nxt = candidates[0]
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    if len(path) != len(pixels):
        raise ValueError(
            f"Skeleton walk covered {len(path)}/{len(pixels)} pixels — the skeleton has a "
            "branch the endpoint-count check didn't catch (e.g. a short spur)."
        )
    return path


def width_profile(mask: np.ndarray, path: Sequence[Point]) -> np.ndarray:
    """Local width (px) at each path point, via 2x the distance-to-background transform."""
    dist = ndimage.distance_transform_edt(mask)
    return np.array([2.0 * dist[y, x] for (y, x) in path])


# ── nose/tail orientation + tail-base landmark ──────────────────────────────


@dataclass
class MouseSkeletonLandmarks:
    path_nose_to_tail: List[Point]
    widths_nose_to_tail: np.ndarray
    nose_point: Point
    tail_tip_point: Point
    tail_base_point: Point
    tail_base_index: int  # index into path_nose_to_tail / widths_nose_to_tail
    tail_centerline: List[Point]  # path_nose_to_tail[tail_base_index:]


def find_tail_base(
    path: Sequence[Point], widths: np.ndarray, trunk_width_frac: float = 0.35
) -> MouseSkeletonLandmarks:
    """
    Orient the path nose-first and locate the tail-base landmark (brief §6
    Stage 5): the point where width drops below trunk_width_frac * trunk
    width and stays there. Orientation is determined by which end has the
    longer sustained run of low width — the tail is thin along its whole
    length, whereas the nose end can dip briefly right at the tip but does
    not sustain it.
    """
    if len(path) < 3:
        raise ValueError("Path too short to determine nose/tail orientation")

    trunk_width = float(np.max(widths))
    threshold = trunk_width_frac * trunk_width
    below = widths < threshold

    def leading_run(arr: np.ndarray) -> int:
        n = 0
        for v in arr:
            if v:
                n += 1
            else:
                break
        return n

    lead = leading_run(below)
    trail = leading_run(below[::-1])

    if lead == trail:
        raise ValueError(
            "Cannot determine nose/tail orientation — both ends have equally sustained "
            "low width; not a plausible mouse silhouette."
        )

    if lead > trail:
        # path[0] end is the tail -> reverse so we return nose-first
        path_oriented = list(reversed(path))
        widths_oriented = widths[::-1]
        tail_base_index = len(path) - lead
    else:
        path_oriented = list(path)
        widths_oriented = widths
        tail_base_index = len(path) - trail

    return MouseSkeletonLandmarks(
        path_nose_to_tail=path_oriented,
        widths_nose_to_tail=widths_oriented,
        nose_point=path_oriented[0],
        tail_tip_point=path_oriented[-1],
        tail_base_point=path_oriented[tail_base_index],
        tail_base_index=tail_base_index,
        tail_centerline=path_oriented[tail_base_index:],
    )


# ── orchestration ────────────────────────────────────────────────────────────


def extract_landmarks_from_mask(
    mask: np.ndarray, trunk_width_frac: float = 0.35
) -> MouseSkeletonLandmarks:
    """Full classical-CV pipeline from a binary mouse mask to landmarks."""
    skeleton = skeletonize(mask)
    path = order_skeleton_path(skeleton)
    widths = width_profile(mask, path)
    return find_tail_base(path, widths, trunk_width_frac=trunk_width_frac)
