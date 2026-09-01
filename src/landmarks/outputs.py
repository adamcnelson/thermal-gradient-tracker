"""
Stage 7 — outputs and QC (project_brief_v7.md §6 Stage 7).

Pure aggregation: combines the already-built, already-tested pieces from
Stages 1 (bout gating), 3 (sync), 4 (registration), 5 (landmarks), and 6
(thermal measurement) into the three output types the brief specifies. No
new measurement logic lives here — if a number looks wrong, the bug is in
one of those upstream modules, not this one.

"QC is what makes this trustworthy across ~100 sessions. Build it in Stage
3, not at the end" (brief) — the individual QC signals (sync residuals,
homography RMSE, gating reasons) were built alongside each stage as they
were written; this module's job is only to collect them into one
per-session report, not to invent new QC criteria at the last minute.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .registration import HomographyFit, NudgeEvent
from .sync import WindowedSyncResult
from .thermal_measurement import DorsalSurfaceMeasurement, TailMeasurement

GRADIENT_ZONE_COOL = "cool"
GRADIENT_ZONE_MID = "mid"
GRADIENT_ZONE_HOT = "hot"


def classify_gradient_zone(
    floor_temp_c: Optional[float], cool_threshold_c: float = 21.0, hot_threshold_c: float = 33.0
) -> Optional[str]:
    """
    Thirds by local floor temperature. Defaults come from the real corpus
    measured during the v7 Stage 0 audit (2026-08-12): observed floor
    range was ~9-45C across sessions, and the hot-end sign-flip crossover
    (§0 audit) sat around 28-33C — cool_threshold/hot_threshold split that
    ~36C range into three roughly-equal bands. Override per-corpus if a
    different session set has a different observed range.
    """
    if floor_temp_c is None or (isinstance(floor_temp_c, float) and np.isnan(floor_temp_c)):
        return None
    if floor_temp_c >= hot_threshold_c:
        return GRADIENT_ZONE_HOT
    if floor_temp_c <= cool_threshold_c:
        return GRADIENT_ZONE_COOL
    return GRADIENT_ZONE_MID


def thermal_time_to_rgb_time(thermal_time_sec: float, sync_result: WindowedSyncResult) -> float:
    """Map a thermal-clock timestamp to its RGB-clock equivalent via the Stage 3 affine fit."""
    return thermal_time_sec + sync_result.offset_sec + sync_result.drift_slope * thermal_time_sec


def rgb_time_to_thermal_time(rgb_time_sec: float, sync_result: WindowedSyncResult) -> float:
    """Inverse of thermal_time_to_rgb_time — solves the affine map for thermal_time."""
    denom = 1 + sync_result.drift_slope
    return (rgb_time_sec - sync_result.offset_sec) / denom if denom != 0 else rgb_time_sec


# ── per-bout table (primary output) ─────────────────────────────────────────


@dataclass
class BoutOutputRow:
    session: str
    track: str  # lane "B" or "F"
    mouse_id: int
    bout_id: int
    bout_start_thermal_sec: float
    bout_end_thermal_sec: float
    bout_start_rgb_sec: Optional[float]
    bout_end_rgb_sec: Optional[float]
    gradient_zone: Optional[str]
    mean_floor_temp_c: Optional[float]
    warm_spot_temp_c: Optional[float]
    dorsal_mean_c: Optional[float]
    dorsal_median_c: Optional[float]
    dorsal_pixel_count: int
    tail_delta_t_c: Optional[float]
    n_frames_averaged: int
    qc_valid: bool
    qc_reasons: List[str] = field(default_factory=list)
    sync_low_confidence: bool = False
    n_registration_low_confidence: int = 0
    # Count of this bout's averaged samples flagged registration_low_confidence
    # (see FrameOutputRow) -- set by the caller after construction (same
    # pattern as n_samples_dorsal_measured in stage7_real_run.py), not part of
    # build_bout_output_row()'s kwargs since it's a per-sample-list aggregate,
    # not a single upstream measurement.
    n_local_edge_refined: int = 0
    # Count of this bout's averaged samples where local_edge_refine() applied a
    # second, small correction -- see FrameOutputRow.local_edge_refined. Same
    # post-construction-assignment pattern as n_registration_low_confidence.


def build_bout_output_row(
    *,
    session: str,
    track: str,
    mouse_id: int,
    bout_id: int,
    bout_start_thermal_sec: float,
    bout_end_thermal_sec: float,
    sync_result: Optional[WindowedSyncResult],
    mean_floor_temp_c: Optional[float],
    warm_spot_temp_c: Optional[float],
    dorsal: Optional[DorsalSurfaceMeasurement],
    tail: Optional[TailMeasurement],
    n_frames_averaged: int,
    qc_valid: bool,
    qc_reasons: Sequence[str] = (),
    gradient_zone_kwargs: Optional[dict] = None,
) -> BoutOutputRow:
    """
    Assemble one per-bout output row from already-computed per-stage results.

    sync_low_confidence carries WindowedSyncResult.low_confidence through
    onto every row (2026-08-24) — deliberately a separate column from
    qc_valid/qc_reasons, not folded into the gating decision: a
    low-confidence sync offset (e.g. Test_4's manually-adopted +11s, see
    WindowedSyncResult.manual_low_confidence()) doesn't block a
    measurement from being computed and reported, it just needs to stay
    visible to anyone filtering/weighting these rows downstream.
    """
    if sync_result is not None:
        rgb_start = thermal_time_to_rgb_time(bout_start_thermal_sec, sync_result)
        rgb_end = thermal_time_to_rgb_time(bout_end_thermal_sec, sync_result)
    else:
        rgb_start = rgb_end = None

    zone_kwargs = gradient_zone_kwargs or {}
    zone = classify_gradient_zone(mean_floor_temp_c, **zone_kwargs)

    return BoutOutputRow(
        session=session,
        track=track,
        mouse_id=mouse_id,
        bout_id=bout_id,
        bout_start_thermal_sec=bout_start_thermal_sec,
        bout_end_thermal_sec=bout_end_thermal_sec,
        bout_start_rgb_sec=rgb_start,
        bout_end_rgb_sec=rgb_end,
        gradient_zone=zone,
        mean_floor_temp_c=mean_floor_temp_c,
        warm_spot_temp_c=warm_spot_temp_c,
        dorsal_mean_c=dorsal.mean_c if dorsal else None,
        dorsal_median_c=dorsal.median_c if dorsal else None,
        dorsal_pixel_count=dorsal.pixel_count if dorsal else 0,
        tail_delta_t_c=tail.delta_t_c if tail else None,
        n_frames_averaged=n_frames_averaged,
        qc_valid=qc_valid,
        qc_reasons=list(qc_reasons),
        sync_low_confidence=sync_result.low_confidence if sync_result is not None else False,
    )


def bout_rows_to_dataframe(rows: Sequence[BoutOutputRow]) -> pd.DataFrame:
    records = []
    for r in rows:
        d = r.__dict__.copy()
        d["qc_reasons"] = "; ".join(d["qc_reasons"])
        records.append(d)
    return pd.DataFrame.from_records(records)


# ── per-frame table (diagnostic) ────────────────────────────────────────────


@dataclass
class FrameOutputRow:
    session: str
    track: str
    frame_number: int
    elapsed_time_thermal_sec: float
    nose_rgb_xy: Optional[tuple]
    tail_base_rgb_xy: Optional[tuple]
    nose_thermal_xy: Optional[tuple]
    tail_base_thermal_xy: Optional[tuple]
    mouse_surface_temp_mean_c: Optional[float]
    floor_temp_mean_c: Optional[float]
    qc_flag: str
    posture: Optional[str] = None  # "extended" | "curled" | "ambiguous" | None (rgb_landmarks.classify_mouse_blob)
    sync_low_confidence: bool = False
    registration_low_confidence: bool = False
    local_edge_refined: bool = False
    # True when stage7_real_run.py's local_edge_refine() applied a small (bounded
    # search window) translation on top of the whole-mask centroid correction, to
    # better align the mask boundary with a real local thermal edge. Adam,
    # 2026-08-31: found real cases where the whole-mask correction left the mask
    # confidently but wrongly positioned near a raised/parallax-shifted body part
    # (see [[project-v7-local-edge-refinement]]) -- this flag says the sample's
    # position was adjusted a second time, not just corrected once.
    # True when the RGB-derived position (before the per-sample thermal-native
    # translation correction) disagreed with the thermal-native track by more
    # than REGISTRATION_LOW_CONFIDENCE_PX -- see stage7_real_run.py. Adam,
    # 2026-08-27: flag inconsistent RGB/thermal cases (e.g. the Test_7 bout07
    # t=967s head/tail-orientation mismatch, [[project-v7-rotation-correction-attempt]])
    # rather than trying to auto-correct them -- large real motion during the
    # sample is the dominant cause (see [[project-v7-test7-alignment-investigation]]),
    # but a genuine registration problem is the other real cause, and this
    # column doesn't distinguish them -- it's a "look closer before trusting
    # this sample" signal, not a verdict.


def frame_rows_to_dataframe(rows: Sequence[FrameOutputRow]) -> pd.DataFrame:
    return pd.DataFrame.from_records([r.__dict__ for r in rows])


# ── per-session QC report ───────────────────────────────────────────────────


@dataclass
class SessionQCReport:
    session: str
    track: str
    homography_rmse_px: Optional[float]
    homography_passes: Optional[bool]
    sync_offset_sec: Optional[float]
    sync_drift_slope: Optional[float]
    sync_r_squared: Optional[float]
    sync_residual_max_sec: Optional[float]
    sync_passes: Optional[bool]
    sync_low_confidence: Optional[bool] = None
    sync_confidence_note: str = ""
    nudge_events: List[NudgeEvent] = field(default_factory=list)
    landmark_yield_by_zone: dict = field(default_factory=dict)  # {"cool": frac, "mid": ..., "hot": ...}
    fallback_invocation_count: int = 0
    rejected_detection_count: int = 0


def build_session_qc_report(
    *,
    session: str,
    track: str,
    homography_fit: Optional[HomographyFit],
    homography_max_rmse_px: float,
    sync_result: Optional[WindowedSyncResult],
    sync_max_residual_sec: Optional[float] = None,
    nudge_events: Sequence[NudgeEvent] = (),
    landmark_yield_by_zone: Optional[dict] = None,
    fallback_invocation_count: int = 0,
    rejected_detection_count: int = 0,
) -> SessionQCReport:
    from .sync import passes_acceptance as sync_passes_acceptance

    return SessionQCReport(
        session=session,
        track=track,
        homography_rmse_px=homography_fit.rmse_px if homography_fit else None,
        homography_passes=homography_fit.passes_acceptance(homography_max_rmse_px) if homography_fit else None,
        sync_offset_sec=sync_result.offset_sec if sync_result else None,
        sync_drift_slope=sync_result.drift_slope if sync_result else None,
        sync_r_squared=sync_result.r_squared if sync_result else None,
        sync_residual_max_sec=sync_result.residual_max_sec if sync_result else None,
        sync_passes=(
            sync_passes_acceptance(
                sync_result,
                **({"max_residual_sec": sync_max_residual_sec} if sync_max_residual_sec is not None else {}),
            )
            if sync_result
            else None
        ),
        sync_low_confidence=sync_result.low_confidence if sync_result else None,
        sync_confidence_note=sync_result.confidence_note if sync_result else "",
        nudge_events=list(nudge_events),
        landmark_yield_by_zone=landmark_yield_by_zone or {},
        fallback_invocation_count=fallback_invocation_count,
        rejected_detection_count=rejected_detection_count,
    )


def compute_landmark_yield_by_zone(
    zone_labels: Sequence[Optional[str]], valid_flags: Sequence[bool]
) -> dict:
    """
    Fraction of frames/bouts producing a valid landmark measurement, broken
    out by gradient zone (brief §6 Stage 5 bake-off metric #3 — the metric
    that decides classical-vs-supervised, since non-uniform yield
    reintroduces the position-dependent bias project_brief_v7.md §3 exists
    to eliminate).
    """
    if len(zone_labels) != len(valid_flags):
        raise ValueError("zone_labels and valid_flags must be the same length")
    df = pd.DataFrame({"zone": zone_labels, "valid": valid_flags})
    df = df[df["zone"].notna()]
    if df.empty:
        return {}
    return df.groupby("zone")["valid"].mean().to_dict()
