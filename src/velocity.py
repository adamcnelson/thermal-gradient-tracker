"""Centroid velocity and rolling dispersion for stationary bout detection."""

import numpy as np
import pandas as pd

from .analysis_config import BoutsConfig


def _rolling_mad(series: pd.Series, window: int) -> pd.Series:
    """Rolling median absolute deviation."""
    return series.rolling(window, center=True, min_periods=3).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )


def _rolling_iqr(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, center=True, min_periods=3).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )


def _rolling_sd(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, center=True, min_periods=3).std()


def _dispersion_on_valid_subindex(
    cx_masked: pd.Series, window: int, metric: str
) -> pd.Series:
    """
    Compute rolling dispersion using only valid (non-NaN) frames.

    Drops NaN frames, computes the rolling window on that sub-series, then
    maps results back to the original index.  This prevents a single invalid
    frame from contaminating the dispersion of nearby valid frames.

    Invalid frames in the original index are left as NaN in the output.
    """
    valid_cx = cx_masked.dropna()
    if len(valid_cx) < 3:
        return pd.Series(np.nan, index=cx_masked.index)

    if metric == "mad":
        disp_sub = _rolling_mad(valid_cx, window)
    elif metric == "iqr":
        disp_sub = _rolling_iqr(valid_cx, window)
    else:
        disp_sub = _rolling_sd(valid_cx, window)

    # Reindex back: valid frames get a value, invalid frames stay NaN
    return disp_sub.reindex(cx_masked.index)


def _rolling_median_on_valid_subindex(cx_masked: pd.Series, window: int) -> pd.Series:
    """
    Smooth centroid-x with a rolling median computed on valid (non-NaN) frames only,
    reindexed back so invalid frames stay NaN. Denoises brief per-frame jitter/dropout
    (including a single locally-recovered frame) before it's fed into dispersion, so a
    short noisy patch doesn't fragment a real rest bout into several short ones.
    """
    valid_cx = cx_masked.dropna()
    if len(valid_cx) < 1:
        return pd.Series(np.nan, index=cx_masked.index)
    smoothed_sub = valid_cx.rolling(window, center=True, min_periods=1).median()
    return smoothed_sub.reindex(cx_masked.index)


def compute_velocity(df: pd.DataFrame, config: BoutsConfig) -> pd.DataFrame:
    """
    Add velocity and dispersion columns to a tracking DataFrame.

    Treats invalid frames (qc_flag != 'ok' or mouse_roi_valid != True) as NaN —
    never as zero movement.

    New columns added:
      centroid_x_roll_dispersion  — rolling dispersion of centroid-x (primary)
      velocity_px_s               — Euclidean inter-sample speed (px/s)
      velocity_x_px_s             — x-axis component (px/s)
      velocity_smooth_px_s        — smoothed Euclidean speed
    """
    out = df.copy()
    out = out.sort_values("frame_number").reset_index(drop=True)

    # ── validity mask ──────────────────────────────────────────────────────────
    valid = (out["qc_flag"] == "ok") & (out["mouse_roi_valid"].astype(str) == "True")

    cx = out["mouse_centroid_x"].where(valid) if config.qc_valid_only_for_dispersion else out["mouse_centroid_x"]
    cy = out["mouse_centroid_y"].where(valid)

    # ── smoothed position ──────────────────────────────────────────────────────
    # Rolling-median smoothing on the valid-only sub-series, before dispersion is
    # computed from it. Denoises brief per-frame centroid jitter/dropout (e.g. a
    # single locally-recovered frame, or a couple of low-confidence samples in a
    # low-contrast stretch) so it doesn't spuriously fragment a real rest bout.
    out["mouse_centroid_x_smooth"] = _rolling_median_on_valid_subindex(
        cx, config.position_smoothing_window_samples
    )

    # ── rolling dispersion ─────────────────────────────────────────────────────
    # Computed on the smoothed, valid-only sub-series so that scattered invalid
    # frames (jump, no_mouse_roi, etc.) or noisy individual samples don't
    # contaminate nearby valid frames' windows.
    metric = config.dispersion_metric.lower()
    w = config.dispersion_window_samples
    out["centroid_x_roll_dispersion"] = _dispersion_on_valid_subindex(
        out["mouse_centroid_x_smooth"], w, metric
    )

    # ── velocity ───────────────────────────────────────────────────────────────
    cx_all = out["mouse_centroid_x"].copy().astype(float)
    cy_all = out["mouse_centroid_y"].copy().astype(float)
    t = out["elapsed_time_sec"].astype(float)

    # Zero out invalid frames so diff-based velocity stays NaN across a gap
    cx_all_valid = cx_all.where(valid)
    cy_all_valid = cy_all.where(valid)

    dt = t.diff()  # seconds between consecutive samples
    dx = cx_all_valid.diff()
    dy = cy_all_valid.where(valid).diff()

    # Avoid division by zero (dt == 0 at frame 0 or same timestamp)
    dt_safe = dt.replace(0, np.nan)

    out["velocity_x_px_s"] = dx / dt_safe
    out["velocity_px_s"] = np.sqrt(dx**2 + dy**2) / dt_safe

    # Smooth velocity (rolling median)
    sw = config.velocity_smoothing_window_samples
    out["velocity_smooth_px_s"] = (
        out["velocity_px_s"]
        .rolling(sw, center=True, min_periods=2)
        .median()
    )

    return out
