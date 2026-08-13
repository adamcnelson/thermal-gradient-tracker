"""
Stage 6 — thermal measurement extraction (project_brief_v7.md §6 Stage 6).

Landmarks and mask regions are derived entirely in RGB space (Stage 5) —
per §3's core principle, thermal is only ever sampled at coordinates, never
used to find the animal. This module warps RGB-derived geometry into
thermal coordinates via the Stage 4 homography and samples radiometric
values there. All partitioning/tracing (anterior third, proximal tail
segment) happens in RGB space BEFORE warping, not after — warping a mask
region is well-defined (cv2.warpPerspective), but re-deriving "the first
third of the body" from an already-warped, potentially-distorted mask
would not be.

Posture classification (curled/rearing, mentioned in the brief's gating
list) is not implemented anywhere yet — gate_measurement() takes it as a
caller-supplied bool and defaults to "not gated on this" if omitted.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..roi_geometry import circular_mask
from ..temperature_extraction import extract_roi_stats
from .registration import apply_homography
from .rgb_landmarks import Point

# ── RGB-space geometry: partitioning body/tail regions before warping ──────


def _path_arc_length_fractions(path: Sequence[Point]) -> np.ndarray:
    """Cumulative arc length along path, normalized to [0, 1]."""
    pts = np.array(path, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total == 0:
        return np.zeros(len(pts))
    return cum / total


def _nearest_path_indices(
    mask: np.ndarray, path: Sequence[Point]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every True pixel in mask, find the index of its nearest point on
    path. Returns (ys, xs, nearest_idx), one entry per foreground pixel.
    Brute-force distance matrix — fine at this scale (mask sizes here are a
    few thousand px at most, paths a few hundred points)."""
    path_arr = np.array(path, dtype=float)  # (row, col) i.e. (y, x)
    ys, xs = np.where(mask)
    pixel_pts = np.column_stack([ys, xs]).astype(float)
    dists = np.linalg.norm(pixel_pts[:, None, :] - path_arr[None, :, :], axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    return ys, xs, nearest_idx


def anterior_region_mask(
    mask: np.ndarray, body_path_nose_to_tail_base: Sequence[Point], fraction: float = 1 / 3
) -> np.ndarray:
    """
    Sub-region of `mask` (brief: "anterior third") — every foreground pixel
    is assigned to its nearest point on the ordered nose->tail-base path,
    and kept if that nearest point's arc-length fraction is < `fraction`.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    fractions = _path_arc_length_fractions(body_path_nose_to_tail_base)
    ys, xs, nearest_idx = _nearest_path_indices(mask, body_path_nose_to_tail_base)
    pixel_fractions = fractions[nearest_idx]

    out = np.zeros_like(mask, dtype=bool)
    out[ys[pixel_fractions < fraction], xs[pixel_fractions < fraction]] = True
    return out


def dorsal_surface_mask(
    mask: np.ndarray, full_path_nose_to_tail: Sequence[Point], tail_base_index: int
) -> np.ndarray:
    """
    Whole-body dorsal-surface region: `mask` minus the tail filament (not in
    the brief — added at Adam's request, 2026-08-13, as an upgrade to the
    pre-v7 pipeline's mouse_surface_temp_mean/median, which used a fixed
    circular ROI rather than the true RGB-derived silhouette shape).

    Every mask pixel is assigned to its nearest point on the FULL
    nose->tail-tip path and kept only if that nearest point is on the body
    portion (index < tail_base_index) — i.e. the pixel is geometrically
    closer to the body than to the tail. This index-based criterion is
    simpler than reusing anterior_region_mask's arc-length-fraction cutoff
    for this purpose: it doesn't depend on every tail pixel coincidentally
    landing at exactly fraction==1.0 to be excluded.
    """
    if not np.any(mask) or tail_base_index <= 0:
        return np.zeros_like(mask, dtype=bool)

    ys, xs, nearest_idx = _nearest_path_indices(mask, full_path_nose_to_tail)
    keep = nearest_idx < tail_base_index

    out = np.zeros_like(mask, dtype=bool)
    out[ys[keep], xs[keep]] = True
    return out


def proximal_tail_points(
    tail_centerline_base_to_tip: Sequence[Point], fraction: float = 0.2
) -> List[Point]:
    """First `fraction` of the tail centerline by arc length, starting at the tail base."""
    fractions = _path_arc_length_fractions(tail_centerline_base_to_tip)
    keep = fractions < fraction
    if not np.any(keep):
        # Degenerate (very short tail segment): always keep at least the base point.
        return [tail_centerline_base_to_tip[0]]
    return [p for p, k in zip(tail_centerline_base_to_tip, keep) if k]


# ── warping RGB geometry into thermal space ─────────────────────────────────


def warp_mask_to_thermal(mask: np.ndarray, H: np.ndarray, thermal_shape: Tuple[int, int]) -> np.ndarray:
    """Warp a boolean RGB-space mask into thermal-space coordinates via homography H."""
    h, w = thermal_shape
    warped = cv2.warpPerspective(mask.astype(np.uint8), H, (w, h))
    return warped > 0


def warp_points_xy(points_yx: Sequence[Point], H: np.ndarray) -> np.ndarray:
    """Warp a list of (row, col) = (y, x) points into thermal-space (x, y) coordinates."""
    xy = np.array([[p[1], p[0]] for p in points_yx], dtype=float)
    return apply_homography(H, xy)


# ── sampling thermal values ─────────────────────────────────────────────────


def warm_spot_temperature(
    thermal_celsius: np.ndarray, warped_anterior_mask: np.ndarray, percentile: float = 95.0
) -> Optional[float]:
    """
    Brief: "Maximum (or 95th percentile) within the anterior third of the
    warped mask." Returns None if the warped region has no valid pixels
    (e.g. warped fully outside the thermal frame).
    """
    if not np.any(warped_anterior_mask):
        return None
    values = thermal_celsius[warped_anterior_mask]
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


@dataclass
class DorsalSurfaceMeasurement:
    mean_c: Optional[float]
    median_c: Optional[float]
    pixel_count: int
    valid: bool
    reason: str = ""


def dorsal_surface_temperature(
    thermal_celsius: np.ndarray, warped_dorsal_mask: np.ndarray, min_valid_pixels: int = 10
) -> DorsalSurfaceMeasurement:
    """Mean/median over the whole warped dorsal-surface region (not in the brief; see dorsal_surface_mask())."""
    count = int(np.sum(warped_dorsal_mask))
    if count < min_valid_pixels:
        return DorsalSurfaceMeasurement(None, None, count, False, f"only {count} warped pixels (<{min_valid_pixels})")
    values = thermal_celsius[warped_dorsal_mask]
    return DorsalSurfaceMeasurement(float(np.mean(values)), float(np.median(values)), count, True)


def sample_curve_max_percentile(
    thermal_celsius: np.ndarray,
    warped_points_xy: np.ndarray,
    sample_radius_px: float,
    percentile: float = 90.0,
) -> Optional[float]:
    """
    Brief §7 tail-base method: at each point along the (warped, proximal)
    tail curve, take the local max (small-radius ROI, guarding against the
    curve being ~1px off the true tail due to warp/scale uncertainty), then
    report a high percentile ACROSS those per-point maxima — not a single
    max/mean over the whole segment, which would be dominated by whichever
    single point happens to land best.
    """
    h, w = thermal_celsius.shape
    per_point_max = []
    for x, y in warped_points_xy:
        if not (0 <= x < w and 0 <= y < h):
            continue
        stats = extract_roi_stats(thermal_celsius, x, y, sample_radius_px)
        if stats["pixel_count"] > 0:
            per_point_max.append(stats["max"])
    if not per_point_max:
        return None
    return float(np.percentile(per_point_max, percentile))


def annulus_mask(cx: float, cy: float, inner_radius: float, outer_radius: float, shape: Tuple[int, int]) -> np.ndarray:
    if outer_radius <= inner_radius:
        raise ValueError(f"outer_radius ({outer_radius}) must exceed inner_radius ({inner_radius})")
    return circular_mask(cx, cy, outer_radius, shape) & ~circular_mask(cx, cy, inner_radius, shape)


def local_floor_temperature(
    thermal_celsius: np.ndarray,
    center_xy: Tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
    exclude_masks: Sequence[np.ndarray] = (),
    min_valid_pixels: int = 10,
) -> Optional[float]:
    """
    Median temperature in an annulus around center_xy, excluding any pixel
    covered by exclude_masks (animal mask, nestlet mask — brief §6/§2b).
    Returns None (invalid, NOT a fallback to some other reference) if fewer
    than min_valid_pixels remain — brief explicitly warns against falling
    back to a session-mean floor temperature, which would reintroduce the
    position-dependent bias the local reference exists to remove.
    """
    cx, cy = center_xy
    ring = annulus_mask(cx, cy, inner_radius_px, outer_radius_px, thermal_celsius.shape)
    for excl in exclude_masks:
        ring = ring & ~excl
    if np.sum(ring) < min_valid_pixels:
        return None
    return float(np.median(thermal_celsius[ring]))


# ── orchestration: tail-base ΔT ─────────────────────────────────────────────


@dataclass
class TailMeasurement:
    tail_temp_c: Optional[float]
    floor_temp_c: Optional[float]
    delta_t_c: Optional[float]
    valid: bool
    reason: str = ""


def tail_base_delta_t(
    thermal_celsius: np.ndarray,
    H: np.ndarray,
    proximal_tail_points_rgb: Sequence[Point],
    sample_radius_px: float,
    floor_inner_radius_px: float,
    floor_outer_radius_px: float,
    animal_mask_thermal: Optional[np.ndarray] = None,
    nestlet_mask_thermal: Optional[np.ndarray] = None,
    tail_percentile: float = 90.0,
    min_floor_pixels: int = 10,
) -> TailMeasurement:
    """Full Stage 6 tail-base pipeline: warp -> sample tail -> sample local floor -> ΔT."""
    warped = warp_points_xy(proximal_tail_points_rgb, H)
    tail_temp = sample_curve_max_percentile(thermal_celsius, warped, sample_radius_px, tail_percentile)
    if tail_temp is None:
        return TailMeasurement(None, None, None, False, "tail curve warped outside thermal frame")

    # Floor annulus centered on the proximal tail segment's own centroid in thermal space.
    center = tuple(warped.mean(axis=0))
    exclude = [m for m in (animal_mask_thermal, nestlet_mask_thermal) if m is not None]
    floor_temp = local_floor_temperature(
        thermal_celsius, center, floor_inner_radius_px, floor_outer_radius_px, exclude, min_floor_pixels
    )
    if floor_temp is None:
        return TailMeasurement(tail_temp, None, None, False, "insufficient clean floor annulus pixels")

    return TailMeasurement(tail_temp, floor_temp, tail_temp - floor_temp, True)


# ── gating (brief §6 "Gating") ───────────────────────────────────────────────


@dataclass
class GatingThresholds:
    min_abs_delta_t_c: float = 0.3
    min_landmark_confidence: float = 0.5


def gate_measurement(
    delta_t_c: Optional[float],
    landmark_confidence: float,
    sync_qc_pass: bool,
    homography_qc_pass: bool,
    posture_ok: Optional[bool] = None,
    thresholds: Optional[GatingThresholds] = None,
) -> Tuple[bool, List[str]]:
    """
    Brief §6 Gating: valid only if |ΔT| exceeds threshold, landmark
    confidence exceeds threshold, posture is not curled/rearing, and sync +
    homography QC pass. posture_ok=None means "not evaluated" (posture
    classification isn't implemented yet) and does not itself invalidate
    the measurement — pass an explicit bool once it exists.
    """
    thresholds = thresholds or GatingThresholds()
    reasons = []
    if delta_t_c is None:
        reasons.append("no delta_t (upstream measurement invalid)")
    elif abs(delta_t_c) < thresholds.min_abs_delta_t_c:
        reasons.append(f"|delta_t|={abs(delta_t_c):.2f} below threshold {thresholds.min_abs_delta_t_c}")
    if landmark_confidence < thresholds.min_landmark_confidence:
        reasons.append(f"landmark_confidence={landmark_confidence:.2f} below threshold")
    if not sync_qc_pass:
        reasons.append("sync QC failed")
    if not homography_qc_pass:
        reasons.append("homography QC failed")
    if posture_ok is False:
        reasons.append("posture is curled/rearing")

    return len(reasons) == 0, reasons


# ── temporal averaging within a bout (brief §6 "Temporal averaging") ───────


def average_valid(values: Sequence[Optional[float]], min_count: int = 1) -> Optional[float]:
    """
    Mean of the non-None values in `values`, or None if fewer than
    min_count are present. Brief: averaging reduces noise, not the
    partial-volume bias that the curve-sampling approach above exists to
    handle — this function is deliberately just a mean, nothing fancier.
    """
    valid = [v for v in values if v is not None]
    if len(valid) < min_count:
        return None
    return float(np.mean(valid))
