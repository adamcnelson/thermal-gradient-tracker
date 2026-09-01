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


@dataclass
class OrientationCheck:
    """
    Result of validate_homography_orientation() -- see that function's
    docstring. `recommendation` is "as_clicked" or "x_mirrored";
    `flagged` is True whenever the clicked correspondence itself (fit
    with no mirroring) is NOT the recommended candidate, meaning the
    homography that would actually get saved/used is wrong and needs the
    horizontal-flip correction before use.
    """

    mean_err_as_clicked_px: float
    mean_err_x_mirrored_px: float
    n_validation_points: int
    recommendation: str
    flagged: bool
    note: str


def validate_homography_orientation(
    rgb_points: np.ndarray,
    thermal_points: np.ndarray,
    matched_rgb_xy: np.ndarray,
    matched_thermal_xy: np.ndarray,
    crop_width: float,
    min_samples: int = 15,
) -> OrientationCheck:
    """
    Catch a left-right (horizontal) orientation mismatch in a session's
    clicked point correspondences -- a real, confirmed failure mode
    (Test_4, 2026-08-25: the rig's documented "vertical flip only"
    convention, true for Test_3, turned out to be a full 180-degree
    rotation for Test_4, most likely because the thermal camera was
    physically repositioned between the two recording days while the RGB
    webcam mounting stayed fixed. Two independent, careful interactive
    calibration attempts both reproduced the same wrong mapping, because
    a human clicking "the same physical corner in both panels" under the
    wrong mental model of the rig's orientation will do so consistently
    wrong -- the resulting fit's own reprojection RMSE looks fine either
    way, since it only measures self-consistency with the (possibly
    mismatched) points it was given, not agreement with physical
    reality. See thermalFeatures/Track_Alignment_Test4.pptx for the
    photographic ground truth that finally caught this.

    This is why reprojection RMSE and homography_deviates_from_canonical()
    (which assumes the session's TRUE transform should resemble a
    trusted reference) cannot catch this class of bug on their own: a
    session can legitimately have a different real physical orientation
    from every other session (a remounted camera), so "deviates from
    canonical" is the wrong test, and "low RMSE against its own clicked
    points" can't detect a self-consistent human mistake. The only real
    check is against INDEPENDENT ground truth: does the fitted mapping
    agree with real, independently-tracked mouse motion in both
    modalities, observed simultaneously?

    Fits two candidate homographies from the same clicked points -- one
    as given (`rgb_points`), one with rgb_points' x-coordinate mirrored
    about `crop_width` before fitting (thermal_points unchanged either
    way) -- then applies each to `matched_rgb_xy` (independent,
    already-time-aligned real RGB centroids, NOT used for fitting; e.g.
    bout-median centroids from a full-session RGB track) and compares
    against `matched_thermal_xy` (the real thermal centroids for those
    same real moments). Whichever candidate lands closer to the real
    thermal positions, on average, wins.

    matched_rgb_xy / matched_thermal_xy should come from genuinely
    independent, high-confidence real tracking (e.g. long, stable
    stationary-bout medians) spanning a real range of track positions --
    a narrow-range or noisy sample can't discriminate the two candidates
    reliably, which is why min_samples exists (though even satisfying it
    doesn't guarantee enough real positional spread; inspect
    mean_err_as_clicked_px vs mean_err_x_mirrored_px's actual separation,
    not just which one is numerically smaller, before trusting a
    borderline call).
    """
    matched_rgb_xy = np.asarray(matched_rgb_xy, dtype=np.float64)
    matched_thermal_xy = np.asarray(matched_thermal_xy, dtype=np.float64)
    n = len(matched_rgb_xy)
    if n < min_samples:
        return OrientationCheck(
            mean_err_as_clicked_px=float("nan"), mean_err_x_mirrored_px=float("nan"),
            n_validation_points=n, recommendation="as_clicked", flagged=True,
            note=f"only {n} independent validation points (<{min_samples}) -- cannot validate orientation, treat as unverified",
        )

    fit_as_clicked = fit_homography(rgb_points, thermal_points)
    rgb_points_mirrored = np.asarray(rgb_points, dtype=np.float64).copy()
    rgb_points_mirrored[:, 0] = crop_width - rgb_points_mirrored[:, 0]
    fit_mirrored = fit_homography(rgb_points_mirrored, thermal_points)

    # Mirroring only the CALIBRATION points before fitting already bakes the
    # x-flip into fit_mirrored.H itself -- applying it to a mirrored copy of
    # the query too would cancel back out (mirror composed with its own
    # inverse is a no-op), silently making both candidates identical. The
    # query must stay in its raw, un-mirrored form here.
    warped_as_clicked = apply_homography(fit_as_clicked.H, matched_rgb_xy)
    warped_mirrored = apply_homography(fit_mirrored.H, matched_rgb_xy)

    err_as_clicked = float(np.mean(np.linalg.norm(warped_as_clicked - matched_thermal_xy, axis=1)))
    err_mirrored = float(np.mean(np.linalg.norm(warped_mirrored - matched_thermal_xy, axis=1)))

    if err_mirrored < err_as_clicked:
        recommendation, flagged = "x_mirrored", True
        note = (
            f"ORIENTATION MISMATCH: the as-clicked correspondence lands {err_as_clicked:.0f}px "
            f"from real, independently-tracked thermal positions on average; mirroring rgb_points' "
            f"x before fitting reduces this to {err_mirrored:.0f}px. This session likely needs a "
            f"full 180-degree (vertical+horizontal) correspondence, not the vertical-flip-only "
            f"convention -- verify with a direct photographic comparison before trusting either fit."
        )
    else:
        recommendation, flagged = "as_clicked", False
        note = (
            f"as-clicked correspondence matches independent tracking well ({err_as_clicked:.0f}px "
            f"mean error vs {err_mirrored:.0f}px for the mirrored alternative) -- no orientation "
            f"mismatch detected."
        )

    return OrientationCheck(
        mean_err_as_clicked_px=err_as_clicked, mean_err_x_mirrored_px=err_mirrored,
        n_validation_points=n, recommendation=recommendation, flagged=flagged, note=note,
    )


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
