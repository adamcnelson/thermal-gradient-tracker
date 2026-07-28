"""
Extract temperature statistics from circular ROIs.

The pipeline converts raw uint16 frames to Celsius using raw_to_celsius()
from seq_io before calling these functions, so all values returned here are
in degrees Celsius (float32). raw_to_celsius() applies the FLIR Planck
equation using per-file calibration constants read via exiftool; see
src/seq_io.py for the formula and its linear fallback (used only when
exiftool/Planck constants are unavailable).
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .roi_geometry import circular_mask


def extract_roi_stats(
    pixel_data: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Dict:
    """
    Extract summary statistics from all pixels inside a circular ROI.

    Parameters
    ----------
    pixel_data : 2D array of thermal values (uint16 ADU or calibrated float)
    cx, cy : ROI center coordinates (image convention: x=col, y=row)
    radius : ROI radius in pixels

    Returns
    -------
    dict with keys: mean, median, min, max, pixel_count
    All values are float; pixel_count is int.
    """
    mask = circular_mask(cx, cy, radius, pixel_data.shape)
    pixels = pixel_data[mask].astype(np.float64)

    if pixels.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "pixel_count": 0,
        }

    return {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "pixel_count": int(pixels.size),
    }


def build_output_row(
    *,
    video_file: str,
    frame_number: int,
    elapsed_time_sec: Optional[float],
    sampling_interval_frames: int,
    # Mouse detection outputs
    mouse_centroid_x: Optional[float],
    mouse_centroid_y: Optional[float],
    mouse_area_px: Optional[int],
    mouse_bbox_x: Optional[int],
    mouse_bbox_y: Optional[int],
    mouse_bbox_width: Optional[int],
    mouse_bbox_height: Optional[int],
    tracking_confidence: float,
    mouse_roi_valid: bool,
    # Mouse ROI measurements
    mouse_roi_center_x: Optional[float],
    mouse_roi_center_y: Optional[float],
    mouse_roi_radius_px: float,
    mouse_stats: Optional[Dict],
    # Floor ROI measurements
    floor_roi_valid: bool,
    floor_roi_center_x: Optional[float],
    floor_roi_center_y: Optional[float],
    floor_roi_radius_px: float,
    floor_stats: Optional[Dict],
    floor_roi_shift_direction: Optional[str],
    floor_roi_shift_distance_px: Optional[float],
    # QC
    qc_flag: str,
    qc_notes: str,
    detection_method: str = "none",
) -> Dict:
    """Assemble one CSV row from per-frame tracking results."""

    def _get(d, key):
        return d[key] if d is not None else np.nan

    mouse_mean = _get(mouse_stats, "mean")
    floor_mean = _get(floor_stats, "mean")
    delta = (mouse_mean - floor_mean) if (mouse_roi_valid and floor_roi_valid) else np.nan

    return {
        "video_file": video_file,
        "frame_number": frame_number,
        "elapsed_time_sec": elapsed_time_sec if elapsed_time_sec is not None else np.nan,
        "sampling_interval_frames": sampling_interval_frames,
        # Mouse surface temperature
        "mouse_surface_temp_mean": _get(mouse_stats, "mean"),
        "mouse_surface_temp_median": _get(mouse_stats, "median"),
        "mouse_surface_temp_min": _get(mouse_stats, "min"),
        "mouse_surface_temp_max": _get(mouse_stats, "max"),
        # Floor temperature
        "floor_temp_mean": _get(floor_stats, "mean"),
        "floor_temp_median": _get(floor_stats, "median"),
        "floor_temp_min": _get(floor_stats, "min"),
        "floor_temp_max": _get(floor_stats, "max"),
        # Derived
        "mouse_minus_floor_temp_mean": delta,
        # Mouse ROI geometry
        "mouse_roi_center_x": mouse_roi_center_x if mouse_roi_center_x is not None else np.nan,
        "mouse_roi_center_y": mouse_roi_center_y if mouse_roi_center_y is not None else np.nan,
        "mouse_roi_radius_px": mouse_roi_radius_px,
        # Floor ROI geometry
        "floor_roi_center_x": floor_roi_center_x if floor_roi_center_x is not None else np.nan,
        "floor_roi_center_y": floor_roi_center_y if floor_roi_center_y is not None else np.nan,
        "floor_roi_radius_px": floor_roi_radius_px,
        "floor_roi_shift_direction": floor_roi_shift_direction if floor_roi_shift_direction else "NA",
        "floor_roi_shift_distance_px": floor_roi_shift_distance_px if floor_roi_shift_distance_px is not None else np.nan,
        # Mouse tracking
        "mouse_centroid_x": mouse_centroid_x if mouse_centroid_x is not None else np.nan,
        "mouse_centroid_y": mouse_centroid_y if mouse_centroid_y is not None else np.nan,
        "mouse_area_px": mouse_area_px if mouse_area_px is not None else np.nan,
        "mouse_bbox_x": mouse_bbox_x if mouse_bbox_x is not None else np.nan,
        "mouse_bbox_y": mouse_bbox_y if mouse_bbox_y is not None else np.nan,
        "mouse_bbox_width": mouse_bbox_width if mouse_bbox_width is not None else np.nan,
        "mouse_bbox_height": mouse_bbox_height if mouse_bbox_height is not None else np.nan,
        "tracking_confidence": tracking_confidence,
        "mouse_roi_valid": mouse_roi_valid,
        "floor_roi_valid": floor_roi_valid,
        "detection_method": detection_method,
        "qc_flag": qc_flag,
        "qc_notes": qc_notes,
    }
