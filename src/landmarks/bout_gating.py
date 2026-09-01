"""
Stage 1 — stationary-bout gating (project_brief_v7.md §6 Stage 1).

All landmark measurement is restricted to stationary bouts. Feature
extraction is an annotation layer on the *existing* bout output, not a
parallel per-frame pipeline — this module adds no new bout-detection logic.
src/bouts.py and src/tracking.py are reused as-is and are not imported or
modified here; this only selects, for each already-detected stationary
bout, the per-frame tracking rows that fall inside it, so Stage 5/6 can
iterate bout-by-bout instead of frame-by-frame.

Schema this expects (matches the real pipeline's output, e.g.
bouts/master_tracking_with_metadata.csv joined against bouts/bout_table.csv):
  tracking_df : one row per sampled frame, with at least video_file,
                elapsed_time_sec, qc_flag, mouse_roi_valid
  bouts_df    : one row per bout, with at least video_file, bout_index,
                bout_start_sec, bout_end_sec (src/bouts.py's
                compute_bout_metrics() output)
"""

from typing import Iterator, Tuple

import pandas as pd


def select_bout_frames(
    tracking_df: pd.DataFrame,
    bout: pd.Series,
    video_col: str = "video_file",
    time_col: str = "elapsed_time_sec",
) -> pd.DataFrame:
    """
    Return the tracking rows belonging to one bout.

    Filters on elapsed_time_sec within [bout_start_sec, bout_end_sec]
    (inclusive) and, if present, video_file equality. Also drops
    qc_flag != "ok" / mouse_roi_valid == False rows found inside the
    window: bout boundaries are defined by *stationarity*
    (src/bouts.py classify_stationary), not by tracking validity, so a
    bout can still contain an occasional no_mouse_roi/jump frame.
    """
    df = tracking_df
    if video_col in df.columns and video_col in bout.index:
        df = df[df[video_col] == bout[video_col]]

    mask = (df[time_col] >= bout["bout_start_sec"]) & (df[time_col] <= bout["bout_end_sec"])
    if "qc_flag" in df.columns:
        mask &= df["qc_flag"] == "ok"
    if "mouse_roi_valid" in df.columns:
        mask &= df["mouse_roi_valid"].astype(bool)

    return df[mask]


def filter_bouts_after_entry(
    bouts_df: pd.DataFrame, entry_time_sec: float, start_col: str = "bout_start_sec"
) -> pd.DataFrame:
    """
    Drop any bout that starts before the real mouse-entry time (see
    entry_detection.py, 2026-08-24).

    The existing (pre-v7) thermal tracker's own pre_entry->ok qc_flag
    transition — which every bout in bouts_df is already filtered through
    upstream, before this module ever sees it — was found, on real Test_3
    data, to fire well before the animal is actually present: it marks
    thermal-clock 47.5s as "tracking start," but the RGB video shows the
    arena is empty until a hand/arm sweeps in and drops the mouse at
    RGB-clock ~33s, and the independently well-established sync offset
    for that session implies those two facts are inconsistent by roughly
    80 seconds. In practice this means at least one early bout (Test_3's
    bout 0, an otherwise-unexplained "hot zone" bout with unusually high
    floor temp that failed to produce a measurement in every real batch
    run of this pipeline) is very likely a spurious artifact of the
    existing tracker's own entry heuristic, not real early-session animal
    behavior.

    entry_time_sec should come from entry_detection.find_entry_frame_index_thermal()
    (converted to a time via the same tracking CSV's elapsed_time_sec),
    not derived by converting an RGB-side entry time across the sync
    offset — that would reintroduce exactly the offset uncertainty this
    function exists to sidestep, since bouts already live on thermal
    clock.
    """
    return bouts_df[bouts_df[start_col] >= entry_time_sec].reset_index(drop=True)


def iter_stationary_bouts(
    tracking_df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    video_col: str = "video_file",
    time_col: str = "elapsed_time_sec",
) -> Iterator[Tuple[pd.Series, pd.DataFrame]]:
    """
    Yield (bout_row, frames_in_bout) for every bout that has at least one
    qc-valid frame inside its window. Bouts with zero valid frames (all
    frames inside the stationary window happened to fail tracking QC) are
    skipped rather than yielded empty, since landmark extraction has
    nothing to annotate for them.
    """
    for _, bout in bouts_df.iterrows():
        frames = select_bout_frames(tracking_df, bout, video_col=video_col, time_col=time_col)
        if len(frames):
            yield bout, frames
