"""
Stage 4 — spatial registration (project_brief_v7.md §6 Stage 4).

Per-session homography (RGB -> thermal), fitted from point correspondences
on the floor plane (gradient-track plate edges/corners, visible in both
modalities). This module is the fitting/application/QC math only — it does
not locate corners in real images. That's an open question (manual
calibration click, mirroring scripts/create_arena_mask.py's existing
pattern, vs. automatic corner detection) that needs a decision before this
can run on a real session; see the module-level TODO.

Parallax correction (brief: "the mouse sits ~15-20mm above the floor plane
... quantify empirically [with] a warm object of known height at several
positions") is stubbed as an explicit no-op by default, NOT implemented
with fabricated numbers — that calibration measurement doesn't exist yet
and isn't something to guess at.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np


def fit_homography(rgb_points: np.ndarray, thermal_points: np.ndarray) -> "HomographyFit":
    """
    Fit RGB -> thermal homography from >=4 point correspondences (brief:
    "initialize from a canonical homography; refine per session" — refining
    here means literally refitting from this session's own correspondences,
    since a homography's 8 DOF are solved directly/linearly and don't
    benefit from an iterative initial guess; see homography_deviates_from_canonical()
    for the "compare against canonical" QC half of that brief item).
    """
    rgb_points = np.asarray(rgb_points, dtype=np.float64)
    thermal_points = np.asarray(thermal_points, dtype=np.float64)
    if rgb_points.shape != thermal_points.shape or rgb_points.shape[0] < 4:
        raise ValueError(
            f"Need >=4 matching (rgb, thermal) point pairs, got shapes "
            f"{rgb_points.shape} and {thermal_points.shape}"
        )

    H, _mask = cv2.findHomography(rgb_points, thermal_points, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed to fit a homography from these points")

    rmse, residuals = reprojection_error(H, rgb_points, thermal_points)
    return HomographyFit(H=H, rmse_px=rmse, residuals_px=residuals)


@dataclass
class HomographyFit:
    H: np.ndarray
    rmse_px: float
    residuals_px: np.ndarray

    def passes_acceptance(self, max_rmse_px: float = 1.0) -> bool:
        """Brief §6 Stage 4 acceptance: reprojection RMSE below ~1px."""
        return self.rmse_px < max_rmse_px


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp an (N, 2) array of RGB points into thermal coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points, H)
    return warped.reshape(-1, 2)


def reprojection_error(
    H: np.ndarray, rgb_points: np.ndarray, thermal_points: np.ndarray
) -> "tuple[float, np.ndarray]":
    """Return (rmse, per-point residual distances in px)."""
    projected = apply_homography(H, rgb_points)
    residuals = np.linalg.norm(projected - np.asarray(thermal_points, dtype=np.float64), axis=1)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return rmse, residuals


def homography_deviates_from_canonical(
    H_session: np.ndarray,
    H_canonical: np.ndarray,
    probe_points_rgb: np.ndarray,
    max_displacement_px: float,
) -> "tuple[bool, np.ndarray]":
    """
    Compare a session's fitted homography against a canonical (long-run
    average / trusted-reference) one by warping the same probe points
    through both and measuring displacement — a sanity check that catches
    a grossly wrong per-session fit (e.g. mismatched correspondences),
    independent of the fit's own internal reprojection RMSE, which can look
    fine even for a fit that's globally off. Returns (deviates, per-point
    displacement_px).
    """
    a = apply_homography(H_session, probe_points_rgb)
    b = apply_homography(H_canonical, probe_points_rgb)
    displacement = np.linalg.norm(a - b, axis=1)
    return bool(np.any(displacement > max_displacement_px)), displacement


# ── within-session nudge detection ──────────────────────────────────────────


@dataclass
class NudgeEvent:
    time_sec: float
    displacement_px: float


def detect_nudges(
    times_sec: np.ndarray, reference_positions: np.ndarray, threshold_px: float
) -> List[NudgeEvent]:
    """
    Flag frame-to-frame discontinuities in a tracked reference point (e.g. a
    plate corner) over the course of a session — brief: "track plate-edge
    positions over time; flag discontinuities." Does not itself track the
    reference point in real frames; takes an already-tracked (time,
    position) series.
    """
    times_sec = np.asarray(times_sec, dtype=np.float64)
    reference_positions = np.asarray(reference_positions, dtype=np.float64)
    if len(times_sec) != len(reference_positions):
        raise ValueError("times_sec and reference_positions must be the same length")
    if len(times_sec) < 2:
        return []

    order = np.argsort(times_sec)
    times_sorted = times_sec[order]
    pos_sorted = reference_positions[order]

    diffs = np.diff(pos_sorted, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    return [
        NudgeEvent(time_sec=float(times_sorted[i + 1]), displacement_px=float(d))
        for i, d in enumerate(dists)
        if d > threshold_px
    ]


# ── parallax / height-offset correction ─────────────────────────────────────

# brief: "the mouse sits ~15-20mm above the floor plane, and obliquity grows
# toward the ends of the track ... quantify empirically (warm object of
# known height at several positions), and apply a fixed height-offset
# correction if the residual exceeds ~1px." No such calibration measurement
# exists yet (see STAGE0_AUDIT.md's open-parameters framing for the same
# kind of gap) — this is deliberately a no-op until it does, not a fabricated
# correction.
HeightCorrectionFn = Callable[[np.ndarray], np.ndarray]


def identity_height_correction(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def apply_height_correction(
    points: np.ndarray, correction_fn: Optional[HeightCorrectionFn] = None
) -> np.ndarray:
    """Apply a parallax/height-offset correction before homography warp, if one exists yet."""
    fn = correction_fn or identity_height_correction
    return fn(points)
