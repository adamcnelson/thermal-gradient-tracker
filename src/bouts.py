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
) -> pd.DataFrame:
    """
    Detect stationary bouts from a boolean mask, merge short gaps, filter short bouts.

    Returns a DataFrame with columns:
      bout_index, bout_start_idx, bout_end_idx,
      bout_start_sec, bout_end_sec, bout_duration_sec, n_samples
    """
    mask = stationary_mask.reset_index(drop=True)
    t = elapsed_time_sec.reset_index(drop=True)

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
        if gap <= max_gap_sec:
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

