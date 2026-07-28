"""Ethogram and diagnostic figures for stationary bout QC."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_config import BoutsConfig


def save_bout_diagnostic(
    df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    output_path: str,
    config: BoutsConfig,
    title_extra: str = "",
) -> None:
    """
    Save a multi-panel diagnostic figure for one tracking file's bout detection.

    Panels:
      1. mouse_centroid_x vs time, bouts shaded
      2. centroid_x_roll_dispersion vs time + threshold
      3. velocity_smooth_px_s vs time + threshold
      4. Stationary / moving ethogram bar
      5. floor_temp + mouse_surface_temp vs time (if available)
    """
    has_temp = "floor_temp_mean" in df.columns and "mouse_surface_temp_mean" in df.columns
    n_panels = 5 if has_temp else 4

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 2.8 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": ([3, 2, 2, 1, 2] if has_temp else [3, 2, 2, 1])},
    )

    t = pd.to_numeric(df["elapsed_time_sec"], errors="coerce")
    cx = pd.to_numeric(df["mouse_centroid_x"], errors="coerce")

    # ── shade bout intervals ───────────────────────────────────────────────────
    def _shade_bouts(ax):
        for _, bout in bouts_df.iterrows():
            ax.axvspan(bout["bout_start_sec"], bout["bout_end_sec"],
                       alpha=0.25, color="steelblue", lw=0)

    # Panel 1: centroid-x
    ax = axes[0]
    ax.scatter(t, cx, s=2, c="k", alpha=0.4, rasterized=True)
    _shade_bouts(ax)
    ax.set_ylabel("centroid_x (px)")
    ax.set_title(f"Mouse position along gradient{' — ' + title_extra if title_extra else ''}")

    # Panel 2: dispersion
    ax = axes[1]
    if "centroid_x_roll_dispersion" in df.columns:
        disp = pd.to_numeric(df["centroid_x_roll_dispersion"], errors="coerce")
        ax.plot(t, disp, lw=0.8, color="darkorange")
        thresh = config.stationary_dispersion_threshold_px
        if thresh is not None:
            ax.axhline(thresh, color="red", lw=1.2, ls="--", label=f"thresh={thresh:.1f} px")
            ax.legend(fontsize=8)
    _shade_bouts(ax)
    ax.set_ylabel("disp (px MAD)")

    # Panel 3: velocity
    ax = axes[2]
    if "velocity_smooth_px_s" in df.columns:
        vel = pd.to_numeric(df["velocity_smooth_px_s"], errors="coerce")
        ax.plot(t, vel, lw=0.8, color="purple")
        vthresh = config.stationary_velocity_threshold_px_per_s
        if vthresh is not None:
            ax.axhline(vthresh, color="red", lw=1.2, ls="--", label=f"thresh={vthresh:.1f} px/s")
            ax.legend(fontsize=8)
    _shade_bouts(ax)
    ax.set_ylabel("vel (px/s)")

    # Panel 4: ethogram bar
    ax = axes[3]
    _draw_ethogram(ax, df, bouts_df, t)
    ax.set_ylabel("state")
    ax.set_yticks([])

    # Panel 5: temperatures
    if has_temp:
        ax = axes[4]
        ft = pd.to_numeric(df["floor_temp_mean"], errors="coerce")
        mt = pd.to_numeric(df["mouse_surface_temp_mean"], errors="coerce")
        ax.plot(t, ft, lw=0.8, color="royalblue", label="floor_temp")
        ax.plot(t, mt, lw=0.8, color="tomato", label="mouse_temp")
        _shade_bouts(ax)
        ax.set_ylabel("temp (°C)")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (s)")

    patch = mpatches.Patch(color="steelblue", alpha=0.4, label="stationary bout")
    fig.legend(handles=[patch], loc="upper right", fontsize=8)

    fig.tight_layout()

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), dpi=120, bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def _draw_ethogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    t: pd.Series,
) -> None:
    """Draw a binary stationary/moving bar in the given axes."""
    t_min = float(t.dropna().min()) if not t.dropna().empty else 0.0
    t_max = float(t.dropna().max()) if not t.dropna().empty else 1.0

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1)

    # Fill whole bar as "moving" (gray)
    ax.axhspan(0, 1, color="lightgray")

    # Overlay stationary bouts
    for _, bout in bouts_df.iterrows():
        ax.axvspan(bout["bout_start_sec"], bout["bout_end_sec"], 0, 1,
                   color="mediumseagreen", alpha=0.85)

    ax.set_yticks([0.5])
    ax.set_yticklabels(["stationary"], fontsize=7)
