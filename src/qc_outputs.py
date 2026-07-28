"""
QC output generation: overlay images, summary CSV, and diagnostic plots.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .seq_io import adu_to_display

log = logging.getLogger("thermal-tracker")


def save_overlay_image(
    pixel_data: np.ndarray,
    mouse_mask: Optional[np.ndarray],
    mouse_roi: Optional[Tuple[float, float, float]],
    floor_roi: Optional[Tuple[float, float, float]],
    frame_number: int,
    mouse_temp: Optional[float],
    floor_temp: Optional[float],
    output_path: str,
    colormap: str = "inferno",
) -> None:
    """
    Save a false-color overlay image with mouse mask, mouse ROI, and floor ROI.

    mouse_roi / floor_roi: (cx, cy, radius) tuples, or None if invalid.
    """
    gray = adu_to_display(pixel_data)
    cmap = plt.get_cmap(colormap)
    rgb = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw mouse mask outline
    if mouse_mask is not None:
        contours, _ = cv2.findContours(
            mouse_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(bgr, contours, -1, (0, 255, 255), 1)  # cyan

    # Draw mouse ROI circle
    if mouse_roi is not None:
        cx, cy, r = mouse_roi
        cv2.circle(bgr, (int(cx), int(cy)), int(r), (0, 255, 255), 1)  # cyan
        cv2.circle(bgr, (int(cx), int(cy)), 2, (0, 255, 255), -1)

    # Draw floor ROI circle
    if floor_roi is not None:
        cx, cy, r = floor_roi
        cv2.circle(bgr, (int(cx), int(cy)), int(r), (0, 255, 0), 1)  # green
        cv2.circle(bgr, (int(cx), int(cy)), 2, (0, 255, 0), -1)

    # Overlay text
    lines = [f"Frame {frame_number}"]
    if mouse_temp is not None:
        lines.append(f"Mouse: {mouse_temp:.1f} C")
    if floor_temp is not None:
        lines.append(f"Floor: {floor_temp:.1f} C")
    if mouse_temp is not None and floor_temp is not None:
        lines.append(f"Delta: {mouse_temp - floor_temp:+.1f} C")

    for i, line in enumerate(lines):
        cv2.putText(
            bgr, line,
            (4, 14 + i * 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(output_path, bgr)


def save_qc_summary(
    df: pd.DataFrame,
    output_path: str,
    entry_info: Optional[dict] = None,
) -> None:
    """Save a one-row-per-file QC summary CSV."""
    n_pre = int((df["qc_flag"] == "pre_entry").sum())
    tracking_df = df[df["qc_flag"] != "pre_entry"]
    n = len(tracking_df)
    n_mouse = int(tracking_df["mouse_roi_valid"].sum()) if n else 0
    n_floor = int(tracking_df["floor_roi_valid"].sum()) if n else 0
    failed = int((tracking_df["qc_flag"] != "ok").sum()) if n else 0

    ei = entry_info or {}
    summary = {
        "tracking_start_frame": ei.get("tracking_start_frame", 0),
        "elapsed_before_tracking_sec": ei.get("elapsed_before_tracking_sec", 0.0),
        "entry_detection_method": ei.get("method", "unknown"),
        "entry_detection_warning": ei.get("warning", ""),
        "n_pre_entry_frames": n_pre,
        "n_tracked_frames": n,
        "n_valid_mouse_roi": n_mouse,
        "pct_valid_mouse_roi": round(100 * n_mouse / max(n, 1), 1),
        "n_valid_floor_roi": n_floor,
        "pct_valid_floor_roi": round(100 * n_floor / max(n, 1), 1),
        "n_failed_frames": failed,
        "pct_failed": round(100 * failed / max(n, 1), 1),
        "median_tracking_confidence": round(
            float(tracking_df["tracking_confidence"].median()) if n else float("nan"), 3
        ),
    }

    flag_counts = df["qc_flag"].value_counts().to_dict()
    summary["qc_flag_counts"] = str(flag_counts)

    pd.DataFrame([summary]).to_csv(output_path, index=False)
    log.info(f"  QC summary saved: {output_path}")


def save_diagnostic_plots(df: pd.DataFrame, output_prefix: str) -> None:
    """
    Save a multi-panel diagnostic figure.

    Panels:
    1. Mouse centroid X position over time (shows movement along gradient)
    2. Floor temperature over time (raw ADU)
    3. Mouse surface temperature over time (raw ADU)
    4. Mouse-minus-floor temperature over time
    5. Histogram of floor temperatures (thermal preference distribution)
    """
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    t = df["elapsed_time_sec"].values
    valid_mouse = df["mouse_roi_valid"].values.astype(bool)
    valid_floor = df["floor_roi_valid"].values.astype(bool)
    valid_both = valid_mouse & valid_floor

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    fig.suptitle(str(prefix.name), fontsize=10)

    # Panel 1: mouse X position
    ax = axes[0]
    if valid_mouse.any():
        ax.plot(t[valid_mouse], df["mouse_centroid_x"].values[valid_mouse], ".", ms=2, alpha=0.6)
    ax.set_ylabel("Mouse centroid X (px)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mouse position along gradient")

    # Panel 2: floor temperature
    ax = axes[1]
    if valid_floor.any():
        ax.plot(t[valid_floor], df["floor_temp_mean"].values[valid_floor], ".", ms=2, alpha=0.6,
                color="tab:orange")
    ax.set_ylabel("Floor temp (°C)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Floor/location temperature (same-footprint historical background)")

    # Panel 3: mouse temperature
    ax = axes[2]
    if valid_mouse.any():
        ax.plot(t[valid_mouse], df["mouse_surface_temp_mean"].values[valid_mouse], ".", ms=2,
                alpha=0.6, color="tab:red")
    ax.set_ylabel("Mouse surface temp (°C)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mouse surface temperature")

    # Panel 4: delta
    ax = axes[3]
    if valid_both.any():
        ax.plot(t[valid_both], df["mouse_minus_floor_temp_mean"].values[valid_both], ".", ms=2,
                alpha=0.6, color="tab:purple")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Mouse − Floor (°C)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mouse minus floor temperature")

    # Panel 5: floor temperature histogram
    ax = axes[4]
    if valid_floor.any():
        floor_vals = df["floor_temp_mean"].values[valid_floor]
        ax.hist(floor_vals, bins=40, color="tab:orange", alpha=0.7)
    ax.set_xlabel("Floor temp (°C)")
    ax.set_ylabel("Frame count")
    ax.set_title("Distribution of occupied floor temperatures")

    fig.tight_layout()
    plot_path = str(prefix) + "_diagnostic.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Diagnostic plot saved: {plot_path}")
