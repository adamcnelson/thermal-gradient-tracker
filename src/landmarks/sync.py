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

RGB motion energy: prefer rgb_motion_energy_from_tracked_centroids(),
which uses Stage 5's real segment_mouse_rgb tracker — the same
centroid-speed principle as the thermal side. The original real
end-to-end validation of this module (2026-08-13) used
rgb_motion_energy_from_frames() (raw frame-differencing) instead, because
Stage 5 didn't exist yet at the time Stage 3 was built; that run failed
its own acceptance gate (R²=0.23), consistent with the brief's warning
that frame-differencing is flicker-vulnerable. rgb_motion_energy_from_frames()
is kept for cases where segmentation genuinely isn't available, but is no
longer the recommended default. Either way: only ever compute motion
energy on frames from the track-matched crop (top=Back/bottom=Front,
confirmed by Adam — see webcam_preprocessing.py), never the full combined
frame, which would let the other mouse's motion contaminate the
correlation and lock onto a spurious lag.
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
    """
    Time-indexed thermal motion-energy trace from the existing tracker's
    centroid speed. Requires a precomputed velocity_col — NOT present in
    the real tracking CSVs this project actually produces (confirmed
    2026-08-13: `trackingOutputs/*_tracking_every10frames.csv` has no
    velocity_smooth_px_s column). Use
    thermal_motion_energy_from_centroids() below against real data; this
    function is kept for a tracker version that does emit a smoothed
    velocity column directly.
    """
    df = tracking_df[[time_col, velocity_col]].dropna()
    return pd.Series(df[velocity_col].to_numpy(), index=df[time_col].to_numpy()).sort_index()


def thermal_motion_energy_from_centroids(
    tracking_df: pd.DataFrame,
    time_col: str = "elapsed_time_sec",
    x_col: str = "mouse_centroid_x",
    y_col: str = "mouse_centroid_y",
) -> pd.Series:
    """
    Thermal motion-energy trace derived directly from the existing
    tracker's per-frame centroid columns (real fallback for
    thermal_motion_energy(), which needs a velocity column this project's
    tracker doesn't emit). Mirrors
    rgb_motion_energy_from_tracked_centroids()'s centroid-speed principle
    on the thermal side: speed is computed only between consecutive rows
    that both have a valid centroid, so a tracking gap (mouse_roi_valid
    False, pre/post-entry rows) doesn't manufacture a spurious sample.
    """
    df = tracking_df[[time_col, x_col, y_col]].dropna().sort_values(time_col)
    if len(df) < 2:
        raise ValueError("Need at least 2 valid-centroid rows to compute motion energy")
    times = df[time_col].to_numpy()
    centroids = df[[x_col, y_col]].to_numpy()
    dt = np.diff(times)
    dpos = np.diff(centroids, axis=0)
    speed = np.linalg.norm(dpos, axis=1) / dt
    speeds_full = np.concatenate([[0.0], speed])
    return pd.Series(speeds_full, index=times)


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


def rgb_motion_energy_from_tracked_centroids(
    frames: Sequence[np.ndarray],
    background_model,
    fps: float,
    min_area: int,
    max_area: int,
    threshold_sigma: float = 3.0,
) -> pd.Series:
    """
    RGB motion-energy trace from actual tracked-centroid speed (Stage 5's
    segment_mouse_rgb), not raw frame-differencing. This is what the module
    docstring's "since there is no RGB tracker yet" caveat was waiting on —
    Stage 5 now exists, so the RGB side can finally follow the same
    centroid-speed principle the brief specifies for thermal, instead of
    the flicker-vulnerable pixel-diff proxy rgb_motion_energy_from_frames()
    used as a stopgap. That function is kept for cases where segmentation
    genuinely isn't available.

    Frames where segmentation fails are skipped entirely (not zero-filled)
    — speed is computed only between consecutive successfully-tracked
    frames, so a tracking gap doesn't manufacture a spurious near-zero- or
    near-infinite-speed sample at the gap's edges.
    """
    from .rgb_landmarks import segment_mouse_rgb

    centroids = []
    times = []
    for i, frame in enumerate(frames):
        mask = segment_mouse_rgb(frame, background_model, min_area, max_area, threshold_sigma)
        if mask is None:
            continue
        ys, xs = np.where(mask)
        centroids.append((float(xs.mean()), float(ys.mean())))
        times.append(i / fps)

    if len(centroids) < 2:
        raise ValueError("Need at least 2 successfully-tracked frames to compute motion energy")

    centroids_arr = np.array(centroids)
    times_arr = np.array(times)
    dt = np.diff(times_arr)
    dpos = np.diff(centroids_arr, axis=0)
    speed = np.linalg.norm(dpos, axis=1) / dt
    speeds_full = np.concatenate([[0.0], speed])
    return pd.Series(speeds_full, index=times_arr)


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
    low_confidence: bool = False
    confidence_note: str = ""

    @classmethod
    def manual_low_confidence(
        cls, offset_sec: float, drift_slope: float = 0.0, note: str = ""
    ) -> "WindowedSyncResult":
        """
        Construct a result for an offset adopted as a working value rather
        than recovered from a converged fit — e.g. a session (like
        07-30-25_Test_4, 2026-08-18) where every fitting method agreed on
        a rough neighborhood but none produced enough real anchors to
        compute a real residual/R² (fit_lag_from_points() requires >=2
        points; this session repeatedly produced 0-1 real position-anchor
        transitions across three separate attempts — see memory
        project-v7-sync-findings for the full investigation).

        r_squared/residual_max_sec are set to NaN (not fabricated as 0 or
        omitted) since no real fit was computed — passes_acceptance()
        already handles NaN correctly (NaN comparisons are False, so both
        the residual and, unless drift_slope is ~0, the R² branch
        correctly evaluate to "not accepted" rather than silently passing).
        window_centers_sec/window_lags_sec are left empty for the same
        reason: there is no set of independent (time, lag) observations
        behind this value, just a single adopted point estimate.

        low_confidence=True is the actual signal this method exists to
        set — it is threaded through Stage 7's SessionQCReport and
        BoutOutputRow (src/landmarks/outputs.py) so every downstream
        measurement built from this sync result carries the flag, without
        blocking those measurements from being computed (brief §6 Stage 3
        literally offers "flag session, drop to thermal-only" as the
        fallback on failure — this is the "flag" half of that without
        forcing the more drastic "drop to thermal-only" half, a deliberate
        choice: see the Stage 3 status writeup, 2026-08-24).
        """
        return cls(
            window_centers_sec=np.array([]),
            window_lags_sec=np.array([]),
            offset_sec=float(offset_sec),
            drift_slope=float(drift_slope),
            r_squared=float("nan"),
            residual_max_sec=float("nan"),
            low_confidence=True,
            confidence_note=note,
        )


def fit_lag_from_points(times_sec: Sequence[float], lags_sec: Sequence[float]) -> WindowedSyncResult:
    """
    Regress a set of independently-measured (time, lag) observations onto
    an affine time map (offset + drift) via OLS — the fitting step shared
    by fit_windowed_lag() (5 arbitrary fixed-fraction windows) and
    fit_bout_edge_lag() (one observation per real stationary-bout onset,
    2026-08-14 — see that function's docstring for why bout edges are a
    better anchor than blind windows). Factored out so both anchoring
    strategies produce the same WindowedSyncResult shape and go through
    passes_acceptance() identically.
    """
    times_arr = np.asarray(times_sec, dtype=float)
    lags_arr = np.asarray(lags_sec, dtype=float)
    if len(times_arr) < 2:
        raise ValueError(f"Need >=2 (time, lag) points to fit a drift model, got {len(times_arr)}")

    slope, intercept, r, _p, _se = stats.linregress(times_arr, lags_arr)
    residuals = lags_arr - (slope * times_arr + intercept)

    return WindowedSyncResult(
        window_centers_sec=times_arr,
        window_lags_sec=lags_arr,
        offset_sec=float(intercept),
        drift_slope=float(slope),
        r_squared=float(r**2) if len(times_arr) > 2 else float("nan"),
        residual_max_sec=float(np.max(np.abs(residuals))),
    )


def theil_sen_fit(times_sec: Sequence[float], lags_sec: Sequence[float]) -> Tuple[float, float]:
    """
    Robust (outlier-resistant) alternative to fit_lag_from_points()'s OLS
    fit, used throughout this project as an independent cross-check that
    an OLS drift/offset estimate isn't being driven by one bad window/edge
    (see passes_acceptance() docstring for the precedent). Returns
    (slope, intercept) — same affine convention as fit_lag_from_points.
    """
    slope, intercept, _lo, _hi = stats.theilslopes(np.asarray(lags_sec, dtype=float), np.asarray(times_sec, dtype=float))
    return float(slope), float(intercept)


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

    NOTE (2026-08-14): a real batch run across every stationary bout in
    two sessions showed this blind-window anchoring is not reliable
    enough for production use — several windows land mostly inside long
    stationary stretches with little real motion to correlate on, and the
    resulting lag estimate is then dominated by noise, not signal. Prefer
    fit_bout_edge_lag() below, which anchors on real motion transients
    (bout onsets) instead. Kept for comparison/regression-testing.
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

    return fit_lag_from_points(used_centers, lags)


def fit_bout_edge_lag(
    thermal_motion: pd.Series,
    rgb_window_fn,
    bout_start_times_sec: Sequence[float],
    coarse_offset_sec: float,
    edge_window_sec: float = 15.0,
) -> WindowedSyncResult:
    """
    Anchor the affine offset/drift fit on real stationary-bout onsets
    instead of blind fixed-fraction windows (see fit_windowed_lag()'s
    note). Every bout start is, by construction (src/bouts.py's
    classify_stationary), a genuine motion-to-stillness transient in the
    thermal stream — a strong, well-localized correlation feature, unlike
    an arbitrary window that might land entirely inside a long stationary
    stretch with near-zero motion-energy variance on one or both sides.

    thermal_motion : full-session thermal motion-energy trace (e.g. from
        thermal_motion_energy_from_centroids()), indexed by THERMAL clock.
    rgb_window_fn : callable(bout_start_time_sec) -> Optional[pd.Series],
        returning the RGB motion-energy trace for a short window bracketing
        that thermal time (already decoded/tracked by the caller — this
        function does no video I/O itself), indexed by the RGB video's OWN
        native clock (i.e. seconds since that video's frame 0 — do NOT
        pre-shift it). Return None if the window couldn't be produced
        (e.g. falls outside the RGB recording).
    bout_start_times_sec : the bout_start_sec values to anchor on.
    coarse_offset_sec : rough single-number offset estimate (thermal_time +
        coarse_offset_sec =~ rgb_native_time) used ONLY to (a) tell the
        caller's rgb_window_fn roughly where to decode from and (b)
        recenter the RGB window onto thermal-comparable time before
        correlating. Getting this right matters: a real end-to-end bug
        (2026-08-14) fed cross_correlate_lag() a thermal-clock window and
        an RGB window still in raw native-video seconds without ever
        reconciling the two clocks — for Test_4 (true offset ~+3s) the
        two ranges happened to overlap by coincidence and produced a
        result anyway, but for Test_3 (true offset ~-96s) they never
        overlapped at all and every single edge was silently dropped
        (caught by the ValueError->skip below). Recentering by
        coarse_offset_sec here, once, in the one place that does this
        math, avoids that trap for any future offset magnitude.

    Windows a locally-matching slice of thermal_motion around each edge
    (+/- edge_window_sec) to correlate against the caller-supplied RGB
    window. Edges where either side fails to produce a valid
    cross-correlation are skipped (not zero-filled), same policy as
    fit_windowed_lag().
    """
    lags: List[float] = []
    used_times: List[float] = []
    for t_edge in bout_start_times_sec:
        rgb_win_native = rgb_window_fn(t_edge)
        if rgb_win_native is None or len(rgb_win_native) < 2:
            continue
        # Recenter the RGB window onto thermal-comparable time so its index
        # range actually overlaps thermal_motion's, regardless of how large
        # coarse_offset_sec is (see docstring above).
        rgb_win = pd.Series(
            rgb_win_native.to_numpy(), index=rgb_win_native.index.to_numpy() - coarse_offset_sec
        )
        th_win = thermal_motion[
            (thermal_motion.index >= t_edge - edge_window_sec)
            & (thermal_motion.index <= t_edge + edge_window_sec)
        ]
        try:
            fine_lag = cross_correlate_lag(th_win, rgb_win, dt=0.05)
        except ValueError:
            continue
        lags.append(coarse_offset_sec + fine_lag)
        used_times.append(float(t_edge))

    if len(used_times) < 2:
        raise ValueError(
            f"Only {len(used_times)}/{len(bout_start_times_sec)} bout edges produced a valid lag "
            "— not enough to fit a drift model"
        )

    return fit_lag_from_points(used_times, lags)


def passes_acceptance(
    result: WindowedSyncResult,
    thermal_frame_sec: Optional[float] = None,
    max_residual_sec: float = 2.0,
    drift_r2_min: float = 0.5,
    drift_slope_zero_tol: float = 1e-4,
) -> bool:
    """
    Brief §6 Stage 3 acceptance — REVISED 2026-08-13 from the brief's original
    "proposed (to be tuned)" criterion (residual < 1 thermal frame AND drift
    R² > 0.9), after a real end-to-end run on the Test_3 session cleared
    neither threshold and turned out to be measuring the wrong things in both
    cases.

    RESIDUAL (0.125s -> 2.0s default): that flat sub-frame target treats all
    session time as equally sync-sensitive. But Stage 1's own rationale for
    restricting measurement to stationary bouts says the opposite: "a 500 ms
    error costs almost nothing mid-bout, and is fatal mid-run" — sync error
    only matters when it's large enough to misattribute a sample across a
    bout BOUNDARY; inside a bout the animal isn't moving, so position doesn't
    depend on exact timing at all. Real bout durations in this corpus run
    from ~15s to 180s+, so a ~1s residual is a small fraction of any bout's
    margin from its own edges. max_residual_sec's default (2.0s) is
    deliberately not a tightly-derived number — it's "clearly small relative
    to real bout durations," not a value chosen from a distribution of bout
    margins. Revisit if the real corpus turns out to have materially shorter
    bouts than seen so far.

    DRIFT R² (0.9 -> 0.5 default): switching the RGB motion-energy signal
    from raw frame-differencing to Stage 5's real tracked-centroid speed took
    a real run on the Test_3 session from R²=0.23/residual 1.29s to
    R²=0.77/residual 1.06s — confirming frame-differencing (not the fitting
    method) was the original problem. But R² alone is a poor gate here for a
    structural reason, not a tuning one: the real cross-session drift effect
    is small in absolute terms (a few seconds of lag change over ~30
    minutes), so the "variance explained" ratio is inherently noisy with only
    a handful of correlation windows, even for a functionally-correct fit —
    R² measures relative-to-noise fit quality, not absolute error, and
    absolute error is what the residual check above already covers directly.
    The R²=0.77 case was cross-checked with Theil-Sen (a regression method
    robust to any single noisy window) and got essentially the same slope
    (-0.00184 vs -0.00185 from OLS) — independent evidence the fit reflects a
    real trend, not overfitting to noise, despite the moderate R². 0.5 keeps
    R² as a real guard against genuine garbage (no correlation between lag
    and time at all) without demanding textbook-strength linearity from a
    small, low-effect-size real-world sample.

    thermal_frame_sec is kept as an explicit opt-in for reproducing the
    brief's original literal residual criterion; when given, it overrides
    max_residual_sec entirely (i.e. passes_acceptance(result,
    thermal_frame_sec=1/8) still reproduces the original residual behavior
    exactly — pass drift_r2_min=0.9 too for the fully original criterion).
    """
    threshold = thermal_frame_sec if thermal_frame_sec is not None else max_residual_sec
    residual_ok = result.residual_max_sec < threshold
    drift_ok = result.r_squared > drift_r2_min or abs(result.drift_slope) < drift_slope_zero_tol
    return residual_ok and drift_ok


# ── trajectory alignment (project_brief_v8 sync for sessions without human anchors) ──

@dataclass
class TrajectorySyncResult:
    offset_sec: float  # rgb_time = thermal_time * (1 + drift_slope) + offset_sec
    drift_slope: float
    camera_fps: float  # thermal_time = frame_number / camera_fps
    loss: float  # fraction of samples > outlier_px from RGB after median debias
    peak_ratio: float  # loss / best loss > min_separation_sec away; small = sharp
    other_fps_loss: float  # best loss under any other fps candidate
    n_samples: int


def _trajectory_loss(frames, thermal_x, rgb_t, rgb_x, fps, drift, offsets, outlier_px, max_gap_sec, min_samples):
    """Loss for every offset at one (fps, drift): fraction of thermal samples whose x
    differs from the interpolated RGB x by > outlier_px, after removing the median
    difference (absorbs a constant homography bias)."""
    q = (frames / fps * (1 + drift))[None, :] + offsets[:, None]
    i1 = np.clip(np.searchsorted(rgb_t, q), 1, len(rgb_t) - 1)
    valid = (q >= rgb_t[0]) & (q <= rgb_t[-1]) & ((rgb_t[i1] - rgb_t[i1 - 1]) <= max_gap_sec)
    d = np.where(valid, np.interp(q, rgb_t, rgb_x) - thermal_x[None, :], np.nan)
    with np.errstate(invalid="ignore"):
        dev = np.abs(d - np.nanmedian(d, axis=1, keepdims=True))
        loss = np.nansum(dev > outlier_px, axis=1) / valid.sum(axis=1)
    n = valid.sum(axis=1)
    loss[n < min_samples] = np.nan
    return loss, n


def estimate_sync_from_trajectories(
    frames: np.ndarray, thermal_x: np.ndarray, rgb_t: np.ndarray, rgb_x: np.ndarray,
    fps_candidates: Sequence[float] = (8.0, 10.0), offset_range_sec: Tuple[float, float] = (-300.0, 300.0),
    drift_range: float = 0.03, outlier_px: float = 15.0, max_gap_sec: float = 3.0,
    min_samples: int = 200, min_separation_sec: float = 15.0,
) -> TrajectorySyncResult:
    """
    Blind RGB<->thermal sync from two independent mouse x-trajectories in thermal
    pixel space: thermal-native centroids (by frame number, so a wrong fps baked
    into a tracking CSV can't bias it) vs the RGB track warped through the
    homography. Grid-searches offset x camera fps x drift, then refines.

    x only: the lane is ~412px long but ~43px tall, so y carries little signal.
    The loss counts outliers rather than a median error -- a median is dominated
    by stationary stretches, which can't distinguish offsets within a bout.

    drift_range is deliberately much wider than real clock drift (~0): a nonzero
    recovered drift means a timing bug (e.g. inaccurate video seeking produced
    drift=-0.0165 on Test_7, 2026-10-01), which callers should flag, not trust.
    Validated on Test_3/4/7 (6 lanes): within ~1s of human-anchored offsets.
    """
    frames, thermal_x = np.asarray(frames, float), np.asarray(thermal_x, float)
    order = np.argsort(rgb_t)
    rgb_t, rgb_x = np.asarray(rgb_t, float)[order], np.asarray(rgb_x, float)[order]
    args = (outlier_px, max_gap_sec, min_samples)

    def search(fps_list, drifts, offsets):
        best = (np.inf, None, None, None, None)
        for fps in fps_list:
            for drift in drifts:
                loss, n = _trajectory_loss(frames, thermal_x, rgb_t, rgb_x, fps, drift, offsets, *args)
                if np.all(np.isnan(loss)):
                    continue
                b = int(np.nanargmin(loss))
                if loss[b] < best[0]:
                    best = (float(loss[b]), fps, drift, float(offsets[b]), (loss, int(n[b])))
        return best

    coarse_offsets = np.arange(offset_range_sec[0], offset_range_sec[1], 0.5)
    coarse_drifts = np.arange(-drift_range, drift_range + 1e-9, 0.0025)
    loss, fps, drift, offset, _ = search(fps_candidates, coarse_drifts, coarse_offsets)
    if fps is None:
        raise ValueError("No offset had enough overlapping samples to score")
    other = [f for f in fps_candidates if f != fps]
    other_loss = search(other, coarse_drifts, coarse_offsets)[0] if other else float("nan")

    fine_offsets = np.arange(offset - 2.0, offset + 2.0, 0.05)
    fine_drifts = np.arange(drift - 0.0025, drift + 0.0025 + 1e-9, 0.00025)
    loss, fps, drift, offset, _ = search([fps], fine_drifts, fine_offsets)

    # sharpness: compare against the best basin elsewhere at the winning (fps, drift)
    curve, _ = _trajectory_loss(frames, thermal_x, rgb_t, rgb_x, fps, drift, coarse_offsets, *args)
    far = np.abs(coarse_offsets - offset) > min_separation_sec
    runner = np.nanmin(curve[far]) if np.any(far & ~np.isnan(curve)) else np.nan
    n = _trajectory_loss(frames, thermal_x, rgb_t, rgb_x, fps, drift, np.array([offset]), *args)[1][0]
    return TrajectorySyncResult(
        offset_sec=offset, drift_slope=float(drift), camera_fps=float(fps), loss=loss,
        peak_ratio=float(loss / runner) if runner and runner > 0 else float("nan"),
        other_fps_loss=float(other_loss), n_samples=int(n),
    )


def consolidate_session_sync(
    lanes: pd.DataFrame, max_lane_spread_sec: float = 1.0, large_offset_sec: float = 15.0,
) -> pd.DataFrame:
    """
    Collapse per-lane trajectory sync estimates into one sync per session.

    Front.seq/Back.seq are the same thermal recording against the same RGB video,
    so both lanes estimate the same sync -- their agreement is an independent check.
    Uses the mean of the unflagged lanes (all lanes if none are clean).

    low_confidence: no clean lane, lanes disagree by > max_lane_spread_sec, or the
    lanes picked different fps. review_reasons (not low_confidence) marks things a
    human should eyeball: a single scored lane, or |offset| > large_offset_sec
    (atypical across this corpus, though possibly real).

    `lanes` needs columns: session, lane, offset_sec, drift_slope, camera_fps, flags.
    """
    rows = []
    lanes = lanes.dropna(subset=["offset_sec"]).copy()
    lanes["flags"] = lanes["flags"].fillna("")
    for session, g in lanes.groupby("session", sort=True):
        clean = g[g["flags"] == ""]
        use = clean if len(clean) else g
        spread = float(g["offset_sec"].max() - g["offset_sec"].min()) if len(g) > 1 else float("nan")
        reasons = []
        if not len(clean):
            reasons.append("no_clean_lane")
        if spread > max_lane_spread_sec:
            reasons.append("lane_disagreement")
        if g["camera_fps"].nunique() > 1:
            reasons.append("lane_fps_mismatch")
        offset = float(use["offset_sec"].mean())
        review = []
        if len(g) == 1:
            review.append("single_lane")
        if abs(offset) > large_offset_sec:
            review.append("large_offset")
        lane_desc = "; ".join(
            f"{r.lane}={r.offset_sec:+.2f}s/drift{r.drift_slope:+.4f}" + (f" [{r.flags}]" if r.flags else "")
            for r in g.itertuples()
        )
        rows.append(dict(
            session=session, offset_sec=round(offset, 3),
            drift_slope=round(float(use["drift_slope"].mean()), 5),
            camera_fps=float(use["camera_fps"].iloc[0]), n_lanes=len(g), n_clean_lanes=len(clean),
            lane_spread_sec=spread, low_confidence=bool(reasons),
            low_confidence_reasons=";".join(reasons), review_reasons=";".join(review), lanes=lane_desc,
        ))
    return pd.DataFrame(rows)
