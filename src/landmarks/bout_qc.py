"""
Stage 7 diagnostic figures — v7 per-bout thermal measurement QC plots
(project_brief_v7.md §1 Objective, §6 Stage 7).

Adapts (extends, does not modify) the existing legacy bout diagnostic
figure (src/bout_qc.py's save_bout_diagnostic(), used by the pre-v7
production pipeline across the full session corpus) by adding two new
panels showing the v7 landmark-specific measurements the brief's
Objective actually asks for:

- warm spot (primary readout #1 — "the consistently warmest anterior
  surface region... a proxy for a bodily region that is reliably
  warmest")
- tail-base ΔT (primary readout #2 — "a vasomotor readout, reported as
  ΔT above local floor temperature")
- whole dorsal-surface mean/median (Adam's added secondary measurement,
  2026-08-13 — not in the brief's original two readouts, but requested
  alongside them)

Per brief §5 ("New code lives in a new subpackage... No edits to existing
modules except a single opt-in hook"), this is a fully standalone module,
not a modification of src/bout_qc.py — that file's legacy 5-panel figure
continues to be produced exactly as before, completely unaffected. The
small amount of panel-drawing logic shared conceptually with
src/bout_qc.py (bout shading, the ethogram bar) is intentionally
duplicated here rather than imported, matching the same
reused-as-is-never-imported-into precedent already established for every
other pre-v7 module this project touches (see src/landmarks/bout_gating.py's
own docstring).

v7 measurements only exist within stationary bouts by design (brief §6
Stage 1) — unlike the legacy figure's continuous per-frame floor/mouse
temp trace, the new panels plot each bout's measurement as a horizontal
segment spanning that bout's own [start, end] window, not a continuous
line, since there is genuinely no v7 measurement in between bouts.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WARM_SPOT_COLOR = "#C97A2B"
DORSAL_COLOR = "#2E6E8E"
TAIL_COLOR = "#8E2E6E"
FLOOR_REF_COLOR = "#9AA5AF"
QC_VALID_COLOR = "#2F7D5C"
QC_INVALID_COLOR = "#9AA5AF"
RGB_TRACK_COLOR = "#1B7A3D"


def _shade_bouts(ax, bouts_df: pd.DataFrame, start_col: str = "bout_start_sec", end_col: str = "bout_end_sec") -> None:
    for _, bout in bouts_df.iterrows():
        ax.axvspan(bout[start_col], bout[end_col], alpha=0.25, color="steelblue", lw=0)


def _draw_ethogram(ax, t: pd.Series, bouts_df: pd.DataFrame) -> None:
    t_min = float(t.dropna().min()) if not t.dropna().empty else 0.0
    t_max = float(t.dropna().max()) if not t.dropna().empty else 1.0
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1)
    ax.axhspan(0, 1, color="lightgray")
    for _, bout in bouts_df.iterrows():
        ax.axvspan(bout["bout_start_sec"], bout["bout_end_sec"], 0, 1, color="mediumseagreen", alpha=0.85)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["stationary"], fontsize=7)


def _plot_bout_span_series(ax, bout_output_df: pd.DataFrame, value_col: str, color: str, lw: float = 2.5) -> None:
    """One horizontal segment per bout at height=value, spanning [bout_start_thermal_sec,
    bout_end_thermal_sec]. Rows where value_col is NaN are skipped entirely (not drawn
    as zero/interpolated) -- e.g. warm_spot_temp_c and tail_delta_t_c are NaN for every
    curled-posture bout, which is a real, meaningful absence, not missing data to fill in."""
    for _, row in bout_output_df.iterrows():
        val = row[value_col]
        if pd.isna(val):
            continue
        ax.plot(
            [row["bout_start_thermal_sec"], row["bout_end_thermal_sec"]], [val, val],
            color=color, lw=lw, solid_capstyle="round", zorder=3,
        )


def _sync_confidence_alpha(
    rt: pd.Series, anchor_sec: Optional[float], min_alpha: float = 0.06, decay_sec: float = 400.0,
) -> np.ndarray:
    """
    Per-point alpha for the RGB track panels, fading with distance (in
    thermal-clock seconds) from `anchor_sec` -- the one point in the
    session where RGB/thermal alignment is actually verified (the entry
    event: a real, sharp motion transient independently confirmed to land
    within ~1s of the known thermal entry time in both sessions, see
    [[project-v7-...]] memory / this module's own module docstring).

    Real finding (2026-08-25): the fixed single-offset sync this project
    uses (no drift term) visibly degrades away from that anchor -- e.g.
    Test_4's RGB-vs-thermal position correlation goes from a weak +0.08
    near the start of the session to -0.40 (actively backwards) by
    t~1500-2000s. FOUR different attempts to fit a drift-corrected model
    (global 2D offset+drift grid search, windowed local correlation, and
    the project's own established bout-edge-anchor method from
    landmarks.sync.fit_bout_edge_lag) all failed to converge on a
    trustworthy answer -- every method gave wildly scattered per-window
    lag estimates (R^2 ~ 0), most likely because the mouse's motion is
    confined to a short, narrow track and fairly repetitive, which makes
    position/velocity correlation fundamentally ambiguous at the
    precision needed to detect real drift. Rather than pick one of those
    low-confidence drift estimates and present it as fact, or leave the
    plot implying uniform full-session precision it doesn't have, RGB
    track points are drawn at full opacity near the anchor and fade
    (exponentially, floor at min_alpha) with distance from it -- an
    honest visual signal that DISTANT RGB positions/velocities are
    real data, but their exact thermal-clock alignment is increasingly
    unverified, not that the data itself is wrong.
    """
    rt = pd.to_numeric(rt, errors="coerce")
    if anchor_sec is None:
        return np.full(len(rt), 0.5)
    dist = (rt - anchor_sec).abs().to_numpy()
    alpha = min_alpha + (1.0 - min_alpha) * np.exp(-dist / decay_sec)
    return np.nan_to_num(alpha, nan=min_alpha)


def _mark_qc_valid(ax, bout_output_df: pd.DataFrame, value_col: str) -> None:
    """Small marker at each bout's midpoint, filled green if qc_valid else hollow gray --
    shows which bouts pass the brief §6 Stage 6 gating (posture, |ΔT| threshold, sync/
    homography), independent of whether the underlying value itself was measurable."""
    for _, row in bout_output_df.iterrows():
        val = row[value_col]
        if pd.isna(val):
            continue
        mid = (row["bout_start_thermal_sec"] + row["bout_end_thermal_sec"]) / 2
        if row["qc_valid"]:
            ax.scatter([mid], [val], s=18, color=QC_VALID_COLOR, zorder=4, edgecolor="none")
        else:
            ax.scatter([mid], [val], s=18, facecolor="none", edgecolor=QC_INVALID_COLOR, linewidth=1.0, zorder=4)


def save_v7_bout_diagnostic(
    tracking_df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    bout_output_df: pd.DataFrame,
    output_path: str,
    title_extra: str = "",
    rgb_track_df: Optional[pd.DataFrame] = None,
    rgb_sync_anchor_thermal_sec: Optional[float] = None,
) -> None:
    """
    Save the v7-extended multi-panel bout diagnostic figure for one session.

    Panels 1-5 mirror src/bout_qc.py's legacy figure exactly (position,
    dispersion, velocity, ethogram, legacy whole-body temp) for direct
    before/after comparison. Panels 6-7 are new:
      6. Dorsal-surface temperatures — floor_temp (thin reference line) +
         warm_spot_temp_c + dorsal_mean_c (solid) + dorsal_median_c (thin
         dashed), one horizontal segment per bout.
      7. Tail-base ΔT — tail_delta_t_c per bout, own y-scale (a
         temperature difference, not an absolute temperature), zero
         reference line per the brief's own framing ("record the sign of
         ΔT as a variable — a mouse reading cooler than the floor... is a
         real physiological observation, not an error").

    tracking_df : the session's tracking CSV (same input as the legacy
        figure) — used for panels 1, 2, 3, 5. The thermal centroid_x panel
        here is drawn on the thermal video's own native x-axis (not
        flipped/rescaled) — matching it visually against rgb_track_df's
        centroid is what rgb_track_df's homography-warped column is for,
        not a transform of this column.
    bouts_df : the RAW (or pre-entry-filtered — see
        landmarks.bout_gating.filter_bouts_after_entry) bout table with
        bout_start_sec/bout_end_sec, used for shading in every panel.
    bout_output_df : Stage 7's per-bout output table (e.g.
        landmark_outputs/<session>_bout_output.csv), providing
        bout_start_thermal_sec/bout_end_thermal_sec/warm_spot_temp_c/
        dorsal_mean_c/dorsal_median_c/tail_delta_t_c/qc_valid.
    rgb_track_df : optional, independent RGB-derived centroid/velocity
        track (see scripts computing e.g. <session>_rgb_track.csv via
        segment_mouse_rgb/classify_mouse_blob at ~1s stride over the full
        session) — a genuinely separate measurement from tracking_df's
        thermal-native centroid, used to cross-check the bout detector's
        stationary-bout calls against RGB evidence independently of the
        thermal track. Expected columns: rgb_time_sec (RGB clock),
        rgb_centroid_x_thermal_aligned (the RGB centroid warped into
        thermal pixel space via the session's own homography, so it's
        directly comparable to tracking_df's mouse_centroid_x — this is
        the "make the axes match" step, done via calibrated homography
        rather than a manual flip/rescale), rgb_velocity_px_s. When None,
        the two new RGB panels are omitted and the figure is identical to
        before this argument was added.
    rgb_sync_anchor_thermal_sec : thermal-clock time of the one point in
        the session where RGB/thermal sync is actually verified (the
        entry event -- see _sync_confidence_alpha()'s docstring for why
        this project cannot yet fit a reliable full-session drift
        correction). RGB track points fade with distance from this
        anchor rather than being drawn at uniform, falsely-precise
        opacity. When None, RGB points are drawn at a flat mid-opacity
        (no confidence claim either way).
    """
    has_temp = "floor_temp_mean" in tracking_df.columns and "mouse_surface_temp_mean" in tracking_df.columns
    has_rgb = rgb_track_df is not None and len(rgb_track_df) > 0
    n_panels = 6 + int(has_temp) + int(has_rgb) * 2
    height_ratios = [3] + ([2] if has_rgb else []) + [2, 2] + ([2] if has_rgb else []) + [1] + ([2] if has_temp else []) + [2, 2]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 2.8 * n_panels), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    t = pd.to_numeric(tracking_df["elapsed_time_sec"], errors="coerce")
    cx = pd.to_numeric(tracking_df["mouse_centroid_x"], errors="coerce")

    idx = 0

    # Panel 1: centroid-x (same as legacy, thermal-native)
    ax = axes[idx]
    ax.scatter(t, cx, s=2, c="k", alpha=0.4, rasterized=True)
    _shade_bouts(ax, bouts_df)
    ax.set_ylabel("centroid_x (px, thermal)")
    ax.set_title(f"v7 landmark-specific thermal measurement QC{' — ' + title_extra if title_extra else ''}")
    idx += 1

    # Panel (NEW, only if rgb_track_df given): RGB centroid_x, warped into
    # thermal pixel space via homography -- placed directly below the
    # thermal centroid panel above for a direct visual cross-check.
    if has_rgb:
        ax = axes[idx]
        rt = pd.to_numeric(rgb_track_df["rgb_time_sec"], errors="coerce")
        rcx = pd.to_numeric(rgb_track_df["rgb_centroid_x_thermal_aligned"], errors="coerce")
        alpha = _sync_confidence_alpha(rt, rgb_sync_anchor_thermal_sec)
        colors = mcolors.to_rgba_array(RGB_TRACK_COLOR, alpha=alpha)
        ax.scatter(rt, rcx, s=3, c=colors, rasterized=True)
        if rgb_sync_anchor_thermal_sec is not None:
            ax.axvline(rgb_sync_anchor_thermal_sec, color=RGB_TRACK_COLOR, lw=0.8, ls=":", alpha=0.6, zorder=1)
        _shade_bouts(ax, bouts_df)
        ax.set_ylabel("RGB centroid_x\n(px, thermal-aligned)")
        idx += 1

    # Panel: dispersion (same as legacy)
    ax = axes[idx]
    if "centroid_x_roll_dispersion" in tracking_df.columns:
        disp = pd.to_numeric(tracking_df["centroid_x_roll_dispersion"], errors="coerce")
        ax.plot(t, disp, lw=0.8, color="darkorange")
    _shade_bouts(ax, bouts_df)
    ax.set_ylabel("disp (px MAD)")
    idx += 1

    # Panel: velocity (same as legacy, thermal-native)
    ax = axes[idx]
    if "velocity_smooth_px_s" in tracking_df.columns:
        vel = pd.to_numeric(tracking_df["velocity_smooth_px_s"], errors="coerce")
        ax.plot(t, vel, lw=0.8, color="purple")
    _shade_bouts(ax, bouts_df)
    ax.set_ylabel("vel (px/s, thermal)")
    idx += 1

    # Panel (NEW, only if rgb_track_df given): RGB velocity -- placed
    # directly below the thermal velocity panel above.
    if has_rgb:
        ax = axes[idx]
        rt = pd.to_numeric(rgb_track_df["rgb_time_sec"], errors="coerce")
        rvel = pd.to_numeric(rgb_track_df["rgb_velocity_px_s"], errors="coerce")
        # Scatter, not a continuous line -- a line plot can't vary per-segment
        # opacity, and a hard color cutoff would misrepresent the smooth,
        # continuous falloff in alignment confidence described above.
        alpha = _sync_confidence_alpha(rt, rgb_sync_anchor_thermal_sec)
        colors = mcolors.to_rgba_array(RGB_TRACK_COLOR, alpha=alpha)
        ax.scatter(rt, rvel, s=4, c=colors, rasterized=True)
        if rgb_sync_anchor_thermal_sec is not None:
            ax.axvline(rgb_sync_anchor_thermal_sec, color=RGB_TRACK_COLOR, lw=0.8, ls=":", alpha=0.6, zorder=1)
        _shade_bouts(ax, bouts_df)
        ax.set_ylabel("RGB vel (px/s)\n(fades = sync unverified)")
        idx += 1

    # Panel: ethogram (same as legacy)
    ax = axes[idx]
    _draw_ethogram(ax, t, bouts_df)
    ax.set_ylabel("state")
    ax.set_yticks([])
    idx += 1

    # Panel: legacy whole-body temp (same as legacy, kept for direct comparison)
    if has_temp:
        ax = axes[idx]
        ft = pd.to_numeric(tracking_df["floor_temp_mean"], errors="coerce")
        mt = pd.to_numeric(tracking_df["mouse_surface_temp_mean"], errors="coerce")
        ax.plot(t, ft, lw=0.8, color="royalblue", label="floor_temp")
        ax.plot(t, mt, lw=0.8, color="tomato", label="mouse_temp (legacy whole-body)")
        _shade_bouts(ax, bouts_df)
        ax.set_ylabel("legacy temp (°C)")
        ax.legend(fontsize=8, loc="upper right")
        idx += 1

    # Panel (NEW): dorsal-surface temperatures
    ax = axes[idx]
    idx += 1
    if has_temp:
        ft = pd.to_numeric(tracking_df["floor_temp_mean"], errors="coerce")
        ax.plot(t, ft, lw=0.6, color=FLOOR_REF_COLOR, alpha=0.7, label="floor_temp (reference)")
    _shade_bouts(ax, bouts_df)
    for _, row in bout_output_df.iterrows():
        if pd.isna(row["dorsal_median_c"]):
            continue
        ax.plot(
            [row["bout_start_thermal_sec"], row["bout_end_thermal_sec"]],
            [row["dorsal_median_c"], row["dorsal_median_c"]],
            color=DORSAL_COLOR, lw=1.0, ls="--", alpha=0.6, zorder=2,
        )
    _plot_bout_span_series(ax, bout_output_df, "dorsal_mean_c", DORSAL_COLOR, lw=2.5)
    _plot_bout_span_series(ax, bout_output_df, "warm_spot_temp_c", WARM_SPOT_COLOR, lw=2.5)
    _mark_qc_valid(ax, bout_output_df, "warm_spot_temp_c")
    ax.set_ylabel("dorsal temp (°C)")
    legend_handles = [
        mlines.Line2D([0], [0], color=FLOOR_REF_COLOR, lw=1.2, label="floor_temp (reference)"),
        mlines.Line2D([0], [0], color=DORSAL_COLOR, lw=2.5, label="dorsal mean (whole surface)"),
        mlines.Line2D([0], [0], color=DORSAL_COLOR, lw=1.0, ls="--", label="dorsal median"),
        mlines.Line2D([0], [0], color=WARM_SPOT_COLOR, lw=2.5, label="warm spot (anterior 95th pct)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right")

    # Panel (NEW): tail-base ΔT
    ax = axes[idx]
    _shade_bouts(ax, bouts_df)
    ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=1)
    _plot_bout_span_series(ax, bout_output_df, "tail_delta_t_c", TAIL_COLOR, lw=2.5)
    _mark_qc_valid(ax, bout_output_df, "tail_delta_t_c")
    ax.set_ylabel("tail ΔT (°C)")
    ax.legend(
        handles=[
            mlines.Line2D([0], [0], color=TAIL_COLOR, lw=2.5, label="tail ΔT (vs. local floor)"),
            mlines.Line2D([0], [0], marker="o", color=QC_VALID_COLOR, lw=0, label="qc_valid"),
            mlines.Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=QC_INVALID_COLOR, lw=0, label="qc failed"),
        ],
        fontsize=7.5, loc="upper right",
    )

    axes[-1].set_xlabel("Time (s, thermal clock)")

    patch = mpatches.Patch(color="steelblue", alpha=0.4, label="stationary bout")
    fig.legend(handles=[patch], loc="upper left", fontsize=8, bbox_to_anchor=(0.01, 0.995))

    if has_rgb and rgb_sync_anchor_thermal_sec is not None:
        fig.text(
            0.01, 0.985,
            "RGB panels: dotted line = verified sync anchor (real motion event, confirmed "
            "aligned within ~1s); opacity fades with distance from it -- this project could not "
            "fit a reliable full-session drift correction (see _sync_confidence_alpha() docstring), "
            "so faint points are real data whose exact time-alignment is unverified, not bad data.",
            fontsize=6.5, color="#555555", ha="left", va="top",
        )

    fig.tight_layout()

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), dpi=120, bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
