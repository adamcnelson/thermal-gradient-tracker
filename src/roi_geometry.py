"""
Circular ROI geometry operations.

All coordinates use image convention: x = column index, y = row index.
Origin (0, 0) is the top-left pixel.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


def circular_mask(
    cx: float,
    cy: float,
    radius: float,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Return a boolean array (shape) with True inside the circle at (cx, cy).
    Pixels on the boundary (distance == radius) are included.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist_sq <= radius ** 2


def roi_in_bounds(cx: float, cy: float, radius: float, shape: Tuple[int, int]) -> bool:
    """Return True if the full circle fits within the image dimensions."""
    h, w = shape
    return (
        cx - radius >= 0
        and cx + radius <= w - 1
        and cy - radius >= 0
        and cy + radius <= h - 1
    )


def roi_fully_inside(
    cx: float,
    cy: float,
    radius: float,
    valid_mask: np.ndarray,
) -> bool:
    """
    Return True if every pixel in the circular ROI is True in valid_mask.
    Also checks image bounds so out-of-range circles always return False.
    """
    if not roi_in_bounds(cx, cy, radius, valid_mask.shape):
        return False
    disk = circular_mask(cx, cy, radius, valid_mask.shape)
    return bool(np.all(valid_mask[disk]))


def roi_overlaps_mask(
    cx: float,
    cy: float,
    radius: float,
    mask: np.ndarray,
) -> bool:
    """Return True if any pixel in the circular ROI is True in mask."""
    if not roi_in_bounds(cx, cy, radius, mask.shape):
        return False
    disk = circular_mask(cx, cy, radius, mask.shape)
    return bool(np.any(mask[disk]))


def find_mouse_roi_center(
    mouse_mask: np.ndarray,
    roi_radius: float,
) -> Optional[Tuple[float, float]]:
    """
    Find the best center for a circular ROI of roi_radius inside mouse_mask.

    Uses the Euclidean distance transform: every pixel's value is its distance
    to the nearest background pixel. Points with distance >= roi_radius are fully
    inside the mask when a circle of that radius is centered there. We pick the
    point with the maximum distance (deepest inside the mask — most stable center).

    Returns (cx, cy) in image coordinates, or None if no valid center exists.
    """
    dist = ndimage.distance_transform_edt(mouse_mask)
    # Pixels where a circle of roi_radius would be fully inside the mask
    candidates = dist >= roi_radius
    if not np.any(candidates):
        return None
    # Pick the deepest candidate (largest inscribed circle center)
    masked_dist = np.where(candidates, dist, 0.0)
    flat_idx = int(np.argmax(masked_dist))
    cy, cx = np.unravel_index(flat_idx, dist.shape)
    return float(cx), float(cy)


def largest_inscribed_circle(
    binary_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Find the center and radius of the largest circle that fits inside binary_mask.
    Returns (cx, cy, radius).
    """
    dist = ndimage.distance_transform_edt(binary_mask)
    flat_idx = int(np.argmax(dist))
    cy, cx = np.unravel_index(flat_idx, dist.shape)
    return float(cx), float(cy), float(dist[cy, cx])


def find_floor_roi(
    mouse_cx: float,
    mouse_cy: float,
    roi_radius: float,
    mouse_mask: np.ndarray,
    arena_mask: np.ndarray,
    max_shift_px: int,
    step_px: int,
) -> Optional[Tuple[float, float, str, float]]:
    """
    Search vertically above and below the mouse ROI center for a valid floor ROI.

    A valid floor ROI position must:
    - Have its full circle inside arena_mask (which excludes walls and image borders)
    - Not overlap the mouse mask (with a small added buffer)

    Searches upward (decreasing row index) and downward simultaneously.
    Returns the closest valid position: (cx, cy, direction, shift_px).
    direction is 'up' (smaller row index) or 'down' (larger row index).
    Returns None if no valid position is found within max_shift_px.
    """
    # Dilate mouse mask slightly to prevent floor ROI from grazing the mouse edge
    buffer_iters = max(1, int(roi_radius) // 4)
    mouse_exclusion = ndimage.binary_dilation(mouse_mask, iterations=buffer_iters)

    best_up: Optional[Tuple[float, float, str, float]] = None
    best_down: Optional[Tuple[float, float, str, float]] = None

    for shift in range(step_px, max_shift_px + 1, step_px):
        if best_up is None:
            cy_up = mouse_cy - shift
            if (roi_fully_inside(mouse_cx, cy_up, roi_radius, arena_mask)
                    and not roi_overlaps_mask(mouse_cx, cy_up, roi_radius, mouse_exclusion)):
                best_up = (mouse_cx, cy_up, "up", float(shift))

        if best_down is None:
            cy_down = mouse_cy + shift
            if (roi_fully_inside(mouse_cx, cy_down, roi_radius, arena_mask)
                    and not roi_overlaps_mask(mouse_cx, cy_down, roi_radius, mouse_exclusion)):
                best_down = (mouse_cx, cy_down, "down", float(shift))

        if best_up is not None and best_down is not None:
            break

    if best_up is None and best_down is None:
        return None
    if best_up is None:
        return best_down
    if best_down is None:
        return best_up
    # Both valid — return the closer one
    return best_up if best_up[3] <= best_down[3] else best_down
