"""
Stage 3 — temporal synchronization (project_brief_v7.md §6 Stage 3).

Aligns the RGB (webcam) and thermal (.seq) clocks for one session by
cross-correlating motion-energy traces from each modality, then fitting an
affine time map (offset + drift) across multiple windows spanning the
session — rather than trusting a single global lag, since the rig's two
recording devices can drift relative to each other over a session.

Thermal motion energy: the existing tracker's centroid speed
(velocity_smooth_px_s, already computed by src/velocity.py) — not raw
frame-differencing, which the brief calls out as vulnerable to lighting
flicker / illuminator cycling.

RGB motion energy: since there is no RGB tracker yet (Stage 5), this uses
frame-differencing energy, but only ever on frames from the track-matched
crop (top=Back/bottom=Front, confirmed by Adam — see
webcam_preprocessing.py) — never the full combined frame, which would let
the other mouse's motion contaminate the correlation and lock onto a
spurious lag.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate


def thermal_motion_energy(
    tracking_df: pd.DataFrame,
    time_col: str = "elapsed_time_sec",
    velocity_col: str = "velocity_smooth_px_s",
) -> pd.Series:
    """Time-indexed thermal motion-energy trace from the existing tracker's centroid speed."""
    df = tracking_df[[time_col, velocity_col]].dropna()
    return pd.Series(df[velocity_col].to_numpy(), index=df[time_col].to_numpy()).sort_index()


def rgb_motion_energy_from_frames(frames: Sequence[np.ndarray], fps: float) -> pd.Series:
    """
    Frame-differencing motion-energy trace from a sequence of already
    track-matched-crop frames (grayscale or color). frames[i] must already
    be the correct top(Back)/bottom(Front) crop.
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to compute motion energy")
    energies = [0.0]
    prev = frames[0].astype(np.float32)
    for f in frames[1:]:
        cur = f.astype(np.float32)
        energies.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    times = np.arange(len(frames)) / fps
    return pd.Series(energies, index=times)


def _resample_to_common_grid(
    a: pd.Series, b: pd.Series, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both traces onto a shared uniform time grid spanning their overlap."""
    t0 = max(a.index.min(), b.index.min())
    t1 = min(a.index.max(), b.index.max())
    if t1 - t0 < 4 * dt:
        raise ValueError(f"Traces overlap for only {t1 - t0:.2f}s — too short to correlate")
    grid = np.arange(t0, t1, dt)
    a_interp = np.interp(grid, a.index.to_numpy(), a.to_numpy())
    b_interp = np.interp(grid, b.index.to_numpy(), b.to_numpy())
    return grid, a_interp, b_interp


def cross_correlate_lag(a: pd.Series, b: pd.Series, dt: float = 0.05) -> float:
    """
    Return the lag (seconds) such that shifting b's clock BACK by `lag`
    aligns it with a: a(t) ~= b(t + lag). Positive lag means an event a
    recorded at time t was recorded by b at time t + lag (b's clock reads
    later for the same real event). Sub-sample precision via parabolic
    interpolation of the correlation peak.
    """
    _grid, a_i, b_i = _resample_to_common_grid(a, b, dt)
    a_i = a_i - a_i.mean()
    b_i = b_i - b_i.mean()
    if np.allclose(a_i, 0) or np.allclose(b_i, 0):
        raise ValueError("One of the traces is constant; cannot cross-correlate")

    corr = correlate(a_i, b_i, mode="full")
    # Empirically verified (not derived from the formula, which is easy to
    # get backwards): scipy.signal.correlate(a, b, 'full') peaks at index k
    # where shifts[k] = -D when b's feature actually occurs D samples AFTER
    # a's (b delayed relative to a). Negate to get our convention: positive
    # lag = b's clock reads later than a's for the same real-world event.
    shifts = np.arange(-(len(b_i) - 1), len(a_i))
    peak_idx = int(np.argmax(corr))

    if 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0

    peak_shift_samples = shifts[peak_idx] + delta
    return float(-peak_shift_samples * dt)


@dataclass
class WindowedSyncResult:
    window_centers_sec: np.ndarray
    window_lags_sec: np.ndarray
    offset_sec: float
    drift_slope: float  # extra seconds of lag per second of session time
    r_squared: float
    residual_max_sec: float


def fit_windowed_lag(
    thermal: pd.Series,
    rgb: pd.Series,
    n_windows: int = 5,
    window_frac: float = 0.3,
    dt: float = 0.05,
) -> WindowedSyncResult:
    """
    Fit lag independently in n_windows windows spread evenly across the
    session, then regress lag on window-center time to recover an affine
    time map (offset + drift), per brief §6 Stage 3.
    """
    t0 = max(thermal.index.min(), rgb.index.min())
    t1 = min(thermal.index.max(), rgb.index.max())
    span = t1 - t0
    window_len = span * window_frac
    if span <= 0:
        raise ValueError("Thermal and RGB traces do not overlap in time")

    if n_windows == 1:
        centers = np.array([t0 + span / 2])
    else:
        starts = np.linspace(t0, max(t0, t1 - window_len), n_windows)
        centers = starts + window_len / 2

    lags: List[float] = []
    used_centers: List[float] = []
    for c in centers:
        w0, w1 = c - window_len / 2, c + window_len / 2
        th_win = thermal[(thermal.index >= w0) & (thermal.index <= w1)]
        rgb_win = rgb[(rgb.index >= w0) & (rgb.index <= w1)]
        try:
            lag = cross_correlate_lag(th_win, rgb_win, dt=dt)
        except ValueError:
            continue
        lags.append(lag)
        used_centers.append(float(c))

    if len(used_centers) < 2:
        raise ValueError(
            f"Only {len(used_centers)}/{n_windows} windows produced a valid lag "
            "— not enough to fit a drift model"
        )

    centers_arr = np.array(used_centers)
    lags_arr = np.array(lags)
    slope, intercept, r, _p, _se = stats.linregress(centers_arr, lags_arr)
    residuals = lags_arr - (slope * centers_arr + intercept)

    return WindowedSyncResult(
        window_centers_sec=centers_arr,
        window_lags_sec=lags_arr,
        offset_sec=float(intercept),
        drift_slope=float(slope),
        r_squared=float(r**2) if len(centers_arr) > 2 else float("nan"),
        residual_max_sec=float(np.max(np.abs(residuals))),
    )


def passes_acceptance(
    result: WindowedSyncResult,
    thermal_frame_sec: float,
    drift_r2_min: float = 0.9,
    drift_slope_zero_tol: float = 1e-4,
) -> bool:
    """
    Brief §6 Stage 3 proposed (to-be-tuned) acceptance: residual < 1 thermal
    frame across all windows, AND (drift fit R² > 0.9 OR drift ~ zero).
    Failure -> caller should flag the session and drop to thermal-only.
    """
    residual_ok = result.residual_max_sec < thermal_frame_sec
    drift_ok = result.r_squared > drift_r2_min or abs(result.drift_slope) < drift_slope_zero_tol
    return residual_ok and drift_ok
