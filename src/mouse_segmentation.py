"""
Mouse detection and segmentation from thermal frames.

Strategy: temporal background modeling + foreground detection.

Background model: median of a sparse sample of frames taken throughout the
video, so the mouse (which moves) is averaged out and only the static
thermal gradient remains.

Foreground: pixels where |frame - background| exceeds an adaptive threshold
(mean + N * std of the difference image). This isolates the mouse as a blob
that differs from the empty gradient.

The adaptive threshold scales with per-frame noise, which helps at gradient
temperatures near the mouse's own body temperature where contrast is low.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import List, Optional, Tuple

from .seq_io import SeqReader


class BackgroundModel:
    """Temporal median background built from a sample of frames."""

    def __init__(self, background: np.ndarray):
        # Store as float32 for subtraction arithmetic
        self._bg = background.astype(np.float32)

    @property
    def background(self) -> np.ndarray:
        return self._bg

    @classmethod
    def build(cls, reader: SeqReader, n_frames: int, random_seed: int = 123) -> "BackgroundModel":
        """
        Sample n_frames evenly from the video and compute their pixel-wise median.

        Evenly spaced sampling ensures the mouse doesn't dominate any region
        of the background (it would need to stay perfectly still throughout
        to corrupt a given pixel's median).
        """
        total = reader.count_frames()
        n = min(n_frames, total)
        rng = np.random.default_rng(random_seed)

        # Evenly spaced indices, then add small random jitter to avoid aliasing
        indices = np.linspace(0, total - 1, n, dtype=int)
        jitter = rng.integers(-2, 3, size=n)
        indices = np.clip(indices + jitter, 0, total - 1)
        target_set = set(indices.tolist())

        frames: List[np.ndarray] = []
        for frame_idx, pixel_data in reader.frames():
            if frame_idx in target_set:
                frames.append(pixel_data.astype(np.float32))
            if len(frames) == n:
                break

        if not frames:
            raise RuntimeError("Could not read any frames to build background model.")

        stack = np.stack(frames, axis=0)
        background = np.median(stack, axis=0)
        return cls(background)

    def foreground_score(self, frame: np.ndarray) -> np.ndarray:
        """Per-pixel absolute difference from background (float32)."""
        return np.abs(frame.astype(np.float32) - self._bg)

    def adaptive_threshold(self, fg_score: np.ndarray, sigma: float) -> float:
        """Threshold = mean + sigma * std of the foreground score image."""
        return float(np.mean(fg_score) + sigma * np.std(fg_score))


def segment_mouse(
    frame: np.ndarray,
    background_model: BackgroundModel,
    min_area: int,
    max_area: int,
    threshold_sigma: float = 3.0,
    last_centroid: Optional[Tuple[float, float]] = None,
    search_radius: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], float, dict]:
    """
    Segment the mouse from one thermal frame.

    Parameters
    ----------
    frame : uint16 array (height, width)
    background_model : BackgroundModel
    min_area, max_area : pixel area bounds for the mouse blob
    threshold_sigma : adaptive threshold multiplier
    last_centroid : (cx, cy) from the previous frame, used to prefer nearby blobs
    search_radius : if set, only consider blobs within this distance of last_centroid

    Returns
    -------
    mouse_mask : bool array or None
    centroid_xy : (cx, cy) or None
    confidence : float in [0, 1]
    debug_info : dict with intermediate values for diagnostics
    """
    fg_score = background_model.foreground_score(frame)
    threshold = background_model.adaptive_threshold(fg_score, threshold_sigma)
    raw_mask = (fg_score > threshold).astype(np.uint8)

    # Close small gaps (connect nearby blobs), then open (remove small speckles)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    labeled, n_labels = ndimage.label(cleaned)
    debug_info = {
        "fg_threshold": threshold,
        "n_blobs_before_filter": n_labels,
    }

    if n_labels == 0:
        return None, None, 0.0, debug_info

    # Gather blob properties and filter by area
    candidates = []
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        area = int(np.sum(component))
        if min_area <= area <= max_area:
            cy, cx = ndimage.center_of_mass(component)
            mean_score = float(np.mean(fg_score[component]))
            candidates.append(
                {"area": area, "cx": cx, "cy": cy, "mask": component, "mean_score": mean_score}
            )

    debug_info["n_blobs_in_area_range"] = len(candidates)

    if not candidates:
        return None, None, 0.0, debug_info

    # If we know where the mouse was, prefer blobs close to that location
    if last_centroid is not None and search_radius is not None:
        lx, ly = last_centroid
        nearby = [
            c for c in candidates
            if np.sqrt((c["cx"] - lx) ** 2 + (c["cy"] - ly) ** 2) <= search_radius
        ]
        if nearby:
            candidates = nearby

    # Pick the largest qualifying blob
    best = max(candidates, key=lambda c: c["area"])

    # Confidence: ratio of mean foreground score to threshold, scaled by area adequacy.
    # A score well above threshold and a reasonably sized blob → high confidence.
    score_ratio = best["mean_score"] / max(threshold, 1.0)
    area_fraction = min(1.0, best["area"] / max(min_area * 2, 1))
    confidence = float(min(1.0, score_ratio * 0.7 + area_fraction * 0.3))

    return (
        best["mask"].astype(bool),
        (best["cx"], best["cy"]),
        confidence,
        debug_info,
    )


def local_fallback_segment(
    fg_score: np.ndarray,
    last_centroid: Tuple[float, float],
    search_radius_px: int,
    threshold_percentile: float,
    min_area: int,
    max_area: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], dict]:
    """
    Recover the mouse when the global adaptive threshold finds nothing, by
    re-searching a small window (symmetric in x and y) around the last known
    centroid at a locally-computed, lower percentile threshold.

    This is the mechanism that recovers the animal when its surface temperature
    nearly matches the floor: the contrast that's too faint to clear the global
    mean+sigma threshold over the whole frame is a much larger fraction of the
    local variance within a small window, and clears a percentile cut computed
    just within that window.

    Returns
    -------
    mouse_mask : bool array (full frame size) or None
    centroid_xy : (cx, cy) in full-frame coordinates, or None
    debug_info : dict with intermediate values for diagnostics
    """
    height, width = fg_score.shape
    lx, ly = last_centroid
    lx, ly = int(round(lx)), int(round(ly))

    x0 = max(lx - search_radius_px, 0)
    x1 = min(lx + search_radius_px + 1, width)
    y0 = max(ly - search_radius_px, 0)
    y1 = min(ly + search_radius_px + 1, height)

    debug_info = {"fallback_window": (x0, x1, y0, y1)}

    window = fg_score[y0:y1, x0:x1]
    if window.size == 0:
        return None, None, debug_info

    local_threshold = float(np.percentile(window, threshold_percentile))
    debug_info["fallback_threshold"] = local_threshold

    local_mask = (window > local_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)

    labeled, n_labels = ndimage.label(local_mask)
    debug_info["fallback_n_blobs"] = n_labels
    if n_labels == 0:
        return None, None, debug_info

    best_component = None
    best_area = 0
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        area = int(np.sum(component))
        if min_area <= area <= max_area and area > best_area:
            best_component = component
            best_area = area

    if best_component is None:
        return None, None, debug_info

    full_mask = np.zeros(fg_score.shape, dtype=bool)
    full_mask[y0:y1, x0:x1] = best_component
    cy_local, cx_local = ndimage.center_of_mass(best_component)
    global_centroid = (cx_local + x0, cy_local + y0)

    debug_info["fallback_area"] = best_area
    return full_mask, global_centroid, debug_info


def compute_mouse_bbox(mouse_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x, y, width, height) bounding box of the mouse mask."""
    rows = np.any(mouse_mask, axis=1)
    cols = np.any(mouse_mask, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    rmin, rmax = int(row_idx[0]), int(row_idx[-1])
    cmin, cmax = int(col_idx[0]), int(col_idx[-1])
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1
