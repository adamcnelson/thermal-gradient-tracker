"""Stationary bout detection, merging, and metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── classification ─────────────────────────────────────────────────────────────

def classify_stationary(
    df: pd.DataFrame,
    disp_thresh: float,
    vel_thresh: Optional[float] = None,
) -> pd.Series:
    """
    Return a boolean Series: True = sample is stationary.

    Primary criterion: centroid_x_roll_dispersion < disp_thresh.
    Secondary (guard): velocity_smooth_px_s < vel_thresh (if thresh provided and column exists).
    Invalid/NaN rows are never classified as stationary.
    """
    valid = df["qc_flag"] == "ok"

    disp = df.get("centroid_x_roll_dispersion", pd.Series(np.nan, index=df.index))
    stat = valid & disp.notna() & (disp < disp_thresh)

    if vel_thresh is not None and "velocity_smooth_px_s" in df.columns:
        vel = df["velocity_smooth_px_s"]
        stat = stat & (vel.isna() | (vel < vel_thresh))

    return stat.rename("stationary")


def classify_frame_state(
    df: pd.DataFrame,
    disp_thresh: float,
    vel_thresh: Optional[float] = None,
) -> pd.Series:
    """
    Return a 3-valued categorical Series: "stationary", "moving", or
    "unknown" — a finer-grained sibling of classify_stationary() (same
    stationary/not-stationary boundary) that additionally distinguishes
    WHY a sample isn't stationary.

    Real bug this exists to let rle_bouts() fix (2026-08-25, found via
    project_v7's bout-fragmentation investigation): classify_stationary()
    treats "confirmed moving" and "no real measurement at all"
    (qc_flag != "ok" — e.g. no_mouse_roi, jump, pre_entry) identically as
    "not stationary". A single brief no_mouse_roi dropout in the middle of
    an otherwise long, genuinely stationary bout then fragments that bout
    at the RLE step, before rle_bouts()'s gap-merge even runs (real
    measured impact: Test_4 loses no_mouse_roi on 16% of frames, Test_3
    1.6%; see [[project-v7-bout-fragmentation-root-cause]] memory).
    "moving" (a real measurement that fails the dispersion/velocity
    threshold — confirmed motion) and "unknown" (no real measurement
    exists) are NOT the same evidence and shouldn't be treated the same
    way when deciding whether to bridge a gap between two stationary
    runs: bridging through confirmed motion would fabricate stationarity
    the data actively contradicts, but bridging through a brief unknown
    gap (surrounded by real stationary evidence on both sides) does not
    fabricate anything — it just declines to let an absence of
    measurement masquerade as evidence of movement.
    """
    valid = df["qc_flag"] == "ok"
    disp = df.get("centroid_x_roll_dispersion", pd.Series(np.nan, index=df.index))
    stat = valid & disp.notna() & (disp < disp_thresh)
    if vel_thresh is not None and "velocity_smooth_px_s" in df.columns:
        vel = df["velocity_smooth_px_s"]
        stat = stat & (vel.isna() | (vel < vel_thresh))

    state = pd.Series("moving", index=df.index)
    state[~valid] = "unknown"
    state[stat] = "stationary"
    return state.rename("frame_state")


# ── run-length encoding ────────────────────────────────────────────────────────

def _rle(mask: pd.Series) -> List[Tuple[bool, int, int]]:
    """Return list of (value, start_idx, end_idx) for each run."""
    runs = []
    if mask.empty:
        return runs
    vals = mask.values
    n = len(vals)
    i = 0
    while i < n:
        v = vals[i]
        j = i
        while j < n and vals[j] == v:
            j += 1
        runs.append((bool(v), i, j - 1))
        i = j
    return runs


def rle_bouts(
    stationary_mask: pd.Series,
    elapsed_time_sec: pd.Series,
    min_bout_sec: float,
    max_gap_sec: float,
    frame_state: Optional[pd.Series] = None,
    max_unknown_gap_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Detect stationary bouts from a boolean mask, merge short gaps, filter short bouts.

    frame_state / max_unknown_gap_sec (both optional, default off —
    existing callers see no behavior change): pass classify_frame_state()'s
    3-valued output to let a gap merge under the more permissive
    max_unknown_gap_sec instead of max_gap_sec when EVERY sample inside
    that gap is "unknown" (no real measurement — e.g. a brief
    no_mouse_roi dropout), not "moving" (a real measurement showing
    confirmed motion). A gap containing even one "moving" sample always
    uses max_gap_sec, same as before — this only extends how long an
    UNMEASURED stretch can be bridged, never how long a MOVING stretch
    can be. See classify_frame_state()'s docstring for the real bug this
    fixes (project_v7 bout-fragmentation investigation, 2026-08-25).

    Returns a DataFrame with columns:
      bout_index, bout_start_idx, bout_end_idx,
      bout_start_sec, bout_end_sec, bout_duration_sec, n_samples
    """
    mask = stationary_mask.reset_index(drop=True)
    t = elapsed_time_sec.reset_index(drop=True)
    state = frame_state.reset_index(drop=True) if frame_state is not None else None

    runs = _rle(mask)

    # Collect raw stationary epochs
    epochs: List[Dict] = []
    for val, s, e in runs:
        if val:
            epochs.append({
                "start_idx": s,
                "end_idx": e,
                "start_sec": float(t.iloc[s]),
                "end_sec": float(t.iloc[e]),
            })

    if not epochs:
        return pd.DataFrame(columns=[
            "bout_index", "bout_start_idx", "bout_end_idx",
            "bout_start_sec", "bout_end_sec", "bout_duration_sec", "n_samples",
        ])

    # ── merge gaps ─────────────────────────────────────────────────────────────
    merged = [epochs[0].copy()]
    for ep in epochs[1:]:
        gap = ep["start_sec"] - merged[-1]["end_sec"]
        tolerance = max_gap_sec
        if state is not None and max_unknown_gap_sec is not None:
            gap_states = state.iloc[merged[-1]["end_idx"] + 1 : ep["start_idx"]]
            if len(gap_states) > 0 and bool((gap_states == "unknown").all()):
                tolerance = max(max_gap_sec, max_unknown_gap_sec)
        if gap <= tolerance:
            merged[-1]["end_idx"] = ep["end_idx"]
            merged[-1]["end_sec"] = ep["end_sec"]
        else:
            merged.append(ep.copy())

    # ── filter short bouts ─────────────────────────────────────────────────────
    records = []
    for i, ep in enumerate(merged):
        dur = ep["end_sec"] - ep["start_sec"]
        if dur >= min_bout_sec:
            n = ep["end_idx"] - ep["start_idx"] + 1
            records.append({
                "bout_index": i,
                "bout_start_idx": ep["start_idx"],
                "bout_end_idx": ep["end_idx"],
                "bout_start_sec": ep["start_sec"],
                "bout_end_sec": ep["end_sec"],
                "bout_duration_sec": dur,
                "n_samples": n,
            })

    # Re-index after filtering
    for new_i, r in enumerate(records):
        r["bout_index"] = new_i

    return pd.DataFrame(records)


# ── per-bout metrics ───────────────────────────────────────────────────────────

def compute_bout_metrics(
    df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build per-bout table with temperature and position summaries.

    metadata dict may include: video_file, track, lane, mouse_id, virus, injection, phase, etc.
    """
    meta = metadata or {}
    rows = []
    df_reset = df.reset_index(drop=True)

    for _, bout in bouts_df.iterrows():
        s = int(bout["bout_start_idx"])
        e = int(bout["bout_end_idx"]) + 1
        seg = df_reset.iloc[s:e]

        row: dict = {**meta}
        row["bout_index"] = int(bout["bout_index"])
        row["bout_start_sec"] = bout["bout_start_sec"]
        row["bout_end_sec"] = bout["bout_end_sec"]
        row["bout_duration_sec"] = bout["bout_duration_sec"]
        row["n_samples"] = bout["n_samples"]

        for col, agg_name, agg_fn in [
            ("floor_temp_mean", "floor_temp_mean_bout", "mean"),
            ("floor_temp_median", "floor_temp_median_bout", "mean"),
            ("mouse_surface_temp_mean", "mouse_surface_temp_mean_bout", "mean"),
            ("mouse_surface_temp_median", "mouse_surface_temp_median_bout", "mean"),
            ("mouse_minus_floor_temp_mean", "mouse_minus_floor_temp_mean_bout", "mean"),
            ("mouse_centroid_x", "mean_centroid_x", "mean"),
        ]:
            if col in seg.columns:
                vals = pd.to_numeric(seg[col], errors="coerce").dropna()
                row[agg_name] = float(vals.mean()) if len(vals) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_bout_summary(
    df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    metadata: Optional[dict] = None,
    trial_length_reference_sec: float = 3000.0,
    lut_arena_time: Optional[str] = None,
) -> dict:
    """Per-file summary statistics."""
    meta = metadata or {}

    valid_t = pd.to_numeric(df["elapsed_time_sec"], errors="coerce")
    trial_length = float(valid_t.dropna().max()) if not valid_t.dropna().empty else np.nan

    n_bouts = len(bouts_df)
    total_stat = float(bouts_df["bout_duration_sec"].sum()) if n_bouts else 0.0
    frac_stat = total_stat / trial_length if (trial_length and trial_length > 0) else np.nan
    mean_dur = float(bouts_df["bout_duration_sec"].mean()) if n_bouts else np.nan
    median_dur = float(bouts_df["bout_duration_sec"].median()) if n_bouts else np.nan

    # Preferred floor temp = mean of per-bout mean floor temps
    if n_bouts and "floor_temp_mean_bout" in bouts_df.columns:
        pref_temp = float(bouts_df["floor_temp_mean_bout"].mean())
    elif n_bouts:
        # Compute inline
        metrics = compute_bout_metrics(df, bouts_df)
        pref_temp = float(metrics["floor_temp_mean_bout"].mean()) if "floor_temp_mean_bout" in metrics else np.nan
    else:
        pref_temp = np.nan

    rate_per_1000s = (n_bouts / trial_length * 1000.0) if (trial_length and trial_length > 0) else np.nan

    summary = {
        **meta,
        "trial_length_sec": trial_length,
        "trial_length_lut": lut_arena_time,
        "n_stationary_bouts": n_bouts,
        "total_stationary_time_sec": total_stat,
        "fraction_time_stationary": frac_stat,
        "mean_bout_duration_sec": mean_dur,
        "median_bout_duration_sec": median_dur,
        "bouts_per_1000s": rate_per_1000s,
        "mean_preferred_floor_temp": pref_temp,
    }
    return summary

