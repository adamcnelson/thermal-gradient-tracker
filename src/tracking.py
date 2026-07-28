"""
Frame-by-frame tracking pipeline.

Processes one .seq file end-to-end:
  1. Build a background model from sampled frames (used for both segmentation
     and, per Adam's direction, "floor" temperature — see below).
  2. For each sampled frame, detect the mouse via the global adaptive
     threshold; if that finds nothing and a last-known centroid exists, retry
     with local-fallback recovery (mouse_segmentation.local_fallback_segment)
     — this is what recovers the mouse when its surface temperature nearly
     matches the floor and the contrast is too faint to clear a whole-frame
     threshold, but still detectable within a small local window.
  3. Place mouse surface ROI (largest inscribed circle in mouse mask).
  4. Extract mouse surface temperature from the live frame at that ROI, and
     "floor"/location temperature from the historical background at the SAME
     footprint — robust to local gradient irregularities that make a nearby
     live-frame reading unrepresentative of the mouse's actual position.
  5. Return one output row per sampled frame.

Temporal continuity: the mouse's centroid from frame t constrains the
search region in frame t + sampling_interval, so transient false positives
are unlikely to hijack tracking. Sudden centroid jumps are flagged in qc_notes.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .arena_mask import ArenaMask, TrackingConfig
from .mouse_segmentation import (
    BackgroundModel,
    segment_mouse,
    local_fallback_segment,
    compute_mouse_bbox,
)
from .roi_geometry import find_mouse_roi_center
from .seq_io import SeqReader, raw_to_celsius, read_planck_constants
from .temperature_extraction import extract_roi_stats, build_output_row

log = logging.getLogger("thermal-tracker")

# If the mouse centroid moves more than this many pixels between sampled frames,
# flag it as a potential jump.
JUMP_THRESHOLD_PX = 80


def _parse_filename_metadata(path: Path) -> dict:
    """
    Extract experimental labels from the filename.
    Expected pattern: DATE_ID1_SEX1_ID2_SEX2_EXPERIMENT_TRACK.seq
    Example: 07-28-25_4540_B_4541_F_Test3-004_Front.seq
    """
    import re

    stem = path.stem
    meta = {"track_label": None, "animal1_id": None, "animal2_id": None, "experiment": None}

    # Track label: last underscore-separated token
    parts = stem.split("_")
    if parts:
        meta["track_label"] = parts[-1]

    # Try to match the full pattern
    pattern = r"(\d{2}-\d{2}-\d{2})_(\d+)_([A-Z])_(\d+)_([A-Z])_([\w-]+)_(\w+)"
    m = re.match(pattern, stem)
    if m:
        meta["date"] = m.group(1)
        meta["animal1_id"] = m.group(2)
        meta["animal1_sex"] = m.group(3)
        meta["animal2_id"] = m.group(4)
        meta["animal2_sex"] = m.group(5)
        meta["experiment"] = m.group(6)
        meta["track_label"] = m.group(7)

    return meta


def detect_tracking_start(
    reader: "SeqReader",
    bg_model: "BackgroundModel",
    config: TrackingConfig,
    total_frames: int,
) -> Tuple[int, dict]:
    """
    Scan forward to find the first frame where the mouse is stably present.

    Returns (tracking_start_frame, info_dict).

    The scan steps at sampling_interval_frames and counts consecutive frames that
    simultaneously satisfy:
      - A valid mouse mask is detected (segmentation not None)
      - Segmentation confidence >= tracking_start_confidence_threshold
      - Disturbance score <= max_entry_disturbance_score
        (disturbance is high when total foreground >> expected mouse size,
         e.g. when a researcher's hand/arm is in the frame)

    tracking_start_frame is the first frame of the stable window (not the last).
    All sampled frames before tracking_start_frame get qc_flag = "pre_entry".

    If manual_tracking_start_frame is set, returns that directly.
    If auto_detect_tracking_start is False, returns frame 0.
    If the stable window is never found, returns 0 with a warning.
    """
    fps = config.camera_fps

    if config.manual_tracking_start_frame is not None:
        start = int(config.manual_tracking_start_frame)
        return start, {
            "method": "manual",
            "tracking_start_frame": start,
            "elapsed_before_tracking_sec": start / fps,
        }

    if not config.auto_detect_tracking_start:
        return 0, {"method": "disabled", "tracking_start_frame": 0,
                   "elapsed_before_tracking_sec": 0.0}

    step = config.sampling_interval_frames
    conf_thresh = config.tracking_start_confidence_threshold
    dist_thresh = config.max_entry_disturbance_score
    required = config.min_stable_mouse_frames

    # Disturbance denominator: total foreground area expected for 4× max mouse
    dist_denom = max(config.max_mouse_area_px * 4, 1)

    consecutive = 0
    window_start = 0
    last_centroid: Optional[Tuple[float, float]] = None

    for frame_idx, pixel_data in reader.frames():
        if frame_idx % step != 0:
            continue

        fg_score = bg_model.foreground_score(pixel_data)
        threshold = bg_model.adaptive_threshold(fg_score, config.segmentation_threshold_sigma)
        disturbance = min(1.0, float(np.sum(fg_score > threshold)) / dist_denom)

        mouse_mask, centroid, confidence, _ = segment_mouse(
            pixel_data,
            bg_model,
            min_area=config.min_mouse_area_px,
            max_area=config.max_mouse_area_px,
            threshold_sigma=config.segmentation_threshold_sigma,
            last_centroid=last_centroid,
            search_radius=None,
        )

        stable = (
            mouse_mask is not None
            and confidence >= conf_thresh
            and disturbance <= dist_thresh
        )

        if stable:
            if consecutive == 0:
                window_start = frame_idx
            consecutive += 1
            last_centroid = centroid
        else:
            consecutive = 0
            last_centroid = None

        if consecutive >= required:
            info = {
                "method": "auto",
                "tracking_start_frame": window_start,
                "elapsed_before_tracking_sec": round(window_start / fps, 1),
                "stable_window_frames": required,
            }
            log.info(
                f"  Mouse entry detected at frame {window_start} "
                f"(~{info['elapsed_before_tracking_sec']}s elapsed)"
            )
            return window_start, info

    log.warning(
        "  Could not detect a stable mouse-entry window. "
        f"Tracking will start from frame 0. "
        "Check segmentation parameters or set manual_tracking_start_frame."
    )
    return 0, {
        "method": "fallback",
        "tracking_start_frame": 0,
        "elapsed_before_tracking_sec": 0.0,
        "warning": "Stable entry window not found; defaulted to frame 0",
    }


def _pre_entry_row(
    video_file: str,
    frame_number: int,
    elapsed_time_sec: float,
    sampling_interval: int,
    mouse_roi_radius: float,
    floor_roi_radius: float,
) -> dict:
    """Build an NA output row for a frame before tracking_start_frame."""
    return build_output_row(
        video_file=video_file,
        frame_number=frame_number,
        elapsed_time_sec=elapsed_time_sec,
        sampling_interval_frames=sampling_interval,
        mouse_centroid_x=None, mouse_centroid_y=None,
        mouse_area_px=None, mouse_bbox_x=None, mouse_bbox_y=None,
        mouse_bbox_width=None, mouse_bbox_height=None,
        tracking_confidence=0.0, mouse_roi_valid=False,
        mouse_roi_center_x=None, mouse_roi_center_y=None,
        mouse_roi_radius_px=mouse_roi_radius, mouse_stats=None,
        floor_roi_valid=False,
        floor_roi_center_x=None, floor_roi_center_y=None,
        floor_roi_radius_px=floor_roi_radius, floor_stats=None,
        floor_roi_shift_direction=None, floor_roi_shift_distance_px=None,
        qc_flag="pre_entry",
        qc_notes="Before tracking start frame",
    )


def track_file(
    seq_path: str,
    config: TrackingConfig,
    arena_mask: ArenaMask,
    output_dir: str,
    overwrite: bool = False,
    save_qc_images: bool = True,
    n_qc_images: int = 20,
) -> Tuple[pd.DataFrame, Path, dict]:
    """
    Run the full tracking pipeline on one cropped .seq file.

    Parameters
    ----------
    seq_path : path to the input .seq file
    config : TrackingConfig with all pipeline parameters
    arena_mask : ArenaMask for this video geometry
    output_dir : directory where CSV and QC files will be saved
    overwrite : if False, raise FileExistsError if output CSV exists
    save_qc_images : whether to save overlay preview images
    n_qc_images : how many evenly-spaced QC images to save

    Returns
    -------
    (DataFrame of per-frame measurements, Path to output CSV)
    """
    from .qc_outputs import save_overlay_image

    seq_path = Path(seq_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = config.sampling_interval_frames
    output_name = f"{seq_path.stem}_tracking_every{interval}frames.csv"
    csv_path = out_dir / output_name

    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use --overwrite to replace): {csv_path}"
        )

    file_meta = _parse_filename_metadata(seq_path)
    log.info(f"Processing: {seq_path.name}")

    # Read Planck calibration constants once (requires exiftool)
    planck = read_planck_constants(str(seq_path))
    if planck is not None:
        log.info(
            f"  Planck constants: R1={planck['R1']:.3f}  R2={planck['R2']:.6f}"
            f"  B={planck['B']:.1f}  O={planck['O']}  F={planck['F']}"
        )
    else:
        log.warning(
            "  exiftool not found or Planck constants missing — "
            "temperatures will be INCORRECT. Install exiftool: brew install exiftool"
        )

    with SeqReader(str(seq_path)) as reader:
        frame_shape = reader.frame_shape
        total_frames = reader.count_frames()
        log.info(f"  Frame shape: {frame_shape[1]}x{frame_shape[0]}  |  ~{total_frames} frames")

        arena_mask.validate_shape(frame_shape)

        # Build background model. Sampled evenly across the whole video, this
        # serves double duty: foreground diffing for segmentation, and — per
        # Adam's direction — the source for "floor" temperature. Reading the
        # historical median at the mouse's own footprint (rather than a live
        # ROI shifted to a nearby position) is robust to local irregularities
        # in the gradient (warps/bubbles/arches) that make a nearby spot's
        # temperature not representative of the mouse's actual position.
        log.info(f"  Building background model from {config.background_n_frames} frames…")
        bg_model = BackgroundModel.build(
            reader, config.background_n_frames, config.random_seed
        )
        celsius_background = raw_to_celsius(bg_model.background, planck)

        # Detect mouse entry — scan forward until stable presence is confirmed
        log.info("  Scanning for mouse entry…")
        tracking_start_frame, entry_info = detect_tracking_start(
            reader, bg_model, config, total_frames
        )

        # Determine which frames to sample
        sampled_indices = set(range(0, total_frames, interval))
        n_sampled = len(sampled_indices)
        n_pre_entry = sum(1 for f in sampled_indices if f < tracking_start_frame)
        log.info(
            f"  Sampling {n_sampled} frames (every {interval} frames) — "
            f"{n_pre_entry} pre-entry (NA), "
            f"{n_sampled - n_pre_entry} to track"
        )

        # Evenly-spaced frames for QC images (only from tracking region)
        qc_indices: set = set()
        if save_qc_images and n_qc_images > 0:
            tracking_indices = sorted(
                f for f in sampled_indices if f >= tracking_start_frame
            )
            qc_step = max(1, len(tracking_indices) // n_qc_images)
            qc_indices = set(tracking_indices[::qc_step])

        qc_image_dir = out_dir / "qc_images" / seq_path.stem
        if save_qc_images:
            qc_image_dir.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        last_centroid: Optional[Tuple[float, float]] = None
        search_radius = min(frame_shape[0], frame_shape[1]) * 0.3  # 30% of frame size

        processed = 0
        for frame_idx, pixel_data in reader.frames():
            if frame_idx not in sampled_indices:
                continue

            elapsed = frame_idx / config.camera_fps

            # Emit NA row for frames before the mouse entered the arena
            if frame_idx < tracking_start_frame:
                rows.append(_pre_entry_row(
                    video_file=seq_path.name,
                    frame_number=frame_idx,
                    elapsed_time_sec=elapsed,
                    sampling_interval=interval,
                    mouse_roi_radius=float(config.mouse_roi_radius_px),
                    floor_roi_radius=float(config.floor_roi_radius_px),
                ))
                continue

            qc_flag = "ok"
            qc_notes = ""

            # Calibrated frame for temperature measurements (°C).
            # Segmentation uses raw uint16 (background model is also uint16).
            celsius_frame = raw_to_celsius(pixel_data, planck)

            # --- Segment mouse ---
            mouse_mask, centroid_xy, confidence, debug_info = segment_mouse(
                pixel_data,
                bg_model,
                min_area=config.min_mouse_area_px,
                max_area=config.max_mouse_area_px,
                threshold_sigma=config.segmentation_threshold_sigma,
                last_centroid=last_centroid,
                search_radius=search_radius,
            )
            detection_method = "global" if mouse_mask is not None else "none"

            # --- Local-fallback recovery ---
            # The global adaptive threshold misses the mouse when its surface
            # temperature nearly matches the floor (contrast too faint to clear
            # a whole-frame mean+sigma cut). Re-search a small window around the
            # last known centroid at a lower, locally-computed threshold — the
            # same faint contrast is a much larger fraction of the local
            # variance there.
            if mouse_mask is None and last_centroid is not None and config.enable_local_fallback:
                fg_score = bg_model.foreground_score(pixel_data)
                mouse_mask, centroid_xy, fallback_debug = local_fallback_segment(
                    fg_score,
                    last_centroid=last_centroid,
                    search_radius_px=config.local_fallback_search_radius_px,
                    threshold_percentile=config.local_fallback_threshold_percentile,
                    min_area=config.local_fallback_min_area_px,
                    max_area=config.max_mouse_area_px,
                )
                if mouse_mask is not None:
                    detection_method = "local_fallback"
                    confidence = 0.5  # lower than a normal global detection; recovery is less certain
                    debug_info = {**debug_info, **fallback_debug}

            mouse_roi_valid = False
            floor_roi_valid = False
            bbox = (None, None, None, None)
            mouse_roi_cx = mouse_roi_cy = None
            floor_cx = floor_cy = None
            floor_dir = None
            floor_dist = None
            mouse_stats = None
            floor_stats = None

            if mouse_mask is None:
                qc_flag = "no_mouse"
                qc_notes = "Mouse not detected"
            else:
                area = int(np.sum(mouse_mask))
                bbox = compute_mouse_bbox(mouse_mask)
                cx, cy = centroid_xy

                # Flag sudden jumps
                if last_centroid is not None:
                    jump = np.sqrt((cx - last_centroid[0]) ** 2 + (cy - last_centroid[1]) ** 2)
                    if jump > JUMP_THRESHOLD_PX:
                        qc_flag = "jump"
                        qc_notes = f"Centroid jumped {jump:.1f}px"

                last_centroid = (cx, cy)

                if detection_method == "local_fallback":
                    qc_notes += " | Recovered via local-fallback (low-contrast mouse-floor match)"

                # --- Place mouse ROI ---
                roi_center = find_mouse_roi_center(mouse_mask, config.mouse_roi_radius_px)
                if roi_center is None:
                    qc_flag = qc_flag if qc_flag != "ok" else "no_mouse_roi"
                    qc_notes += " | Mouse ROI does not fit inside mask"
                else:
                    mouse_roi_cx, mouse_roi_cy = roi_center
                    mouse_stats = extract_roi_stats(
                        celsius_frame, mouse_roi_cx, mouse_roi_cy, config.mouse_roi_radius_px
                    )
                    mouse_roi_valid = True

                    # --- "Floor" / location temperature: same footprint as the mouse ROI,
                    # read from the historical background instead of a live adjacent ROI.
                    # This is the temperature the gradient typically reads at the mouse's
                    # current position — robust to local gradient irregularities (warps,
                    # bubbles, arches) that make a nearby live spot not representative.
                    floor_cx, floor_cy = mouse_roi_cx, mouse_roi_cy
                    floor_dir, floor_dist = "same_footprint", 0.0
                    floor_stats = extract_roi_stats(
                        celsius_background, floor_cx, floor_cy, config.floor_roi_radius_px
                    )
                    floor_roi_valid = floor_stats["pixel_count"] > 0
                    if not floor_roi_valid:
                        qc_notes += " | No valid floor ROI found"
                        if qc_flag == "ok":
                            qc_flag = "no_floor_roi"

            # Assemble output row
            row = build_output_row(
                video_file=seq_path.name,
                frame_number=frame_idx,
                elapsed_time_sec=elapsed,
                sampling_interval_frames=interval,
                mouse_centroid_x=centroid_xy[0] if centroid_xy else None,
                mouse_centroid_y=centroid_xy[1] if centroid_xy else None,
                mouse_area_px=int(np.sum(mouse_mask)) if mouse_mask is not None else None,
                mouse_bbox_x=bbox[0],
                mouse_bbox_y=bbox[1],
                mouse_bbox_width=bbox[2],
                mouse_bbox_height=bbox[3],
                tracking_confidence=confidence,
                mouse_roi_valid=mouse_roi_valid,
                mouse_roi_center_x=mouse_roi_cx,
                mouse_roi_center_y=mouse_roi_cy,
                mouse_roi_radius_px=float(config.mouse_roi_radius_px),
                mouse_stats=mouse_stats,
                floor_roi_valid=floor_roi_valid,
                floor_roi_center_x=floor_cx,
                floor_roi_center_y=floor_cy,
                floor_roi_radius_px=float(config.floor_roi_radius_px),
                floor_stats=floor_stats,
                floor_roi_shift_direction=floor_dir,
                floor_roi_shift_distance_px=floor_dist,
                detection_method=detection_method,
                qc_flag=qc_flag,
                qc_notes=qc_notes.strip(" |"),
            )
            rows.append(row)

            # Save QC overlay image for selected frames
            if save_qc_images and frame_idx in qc_indices:
                qc_img_path = qc_image_dir / f"frame_{frame_idx:06d}.png"
                save_overlay_image(
                    pixel_data=pixel_data,
                    mouse_mask=mouse_mask,
                    mouse_roi=(mouse_roi_cx, mouse_roi_cy, config.mouse_roi_radius_px)
                    if mouse_roi_valid else None,
                    floor_roi=(floor_cx, floor_cy, config.floor_roi_radius_px)
                    if floor_roi_valid else None,
                    frame_number=frame_idx,
                    mouse_temp=mouse_stats["mean"] if mouse_stats else None,
                    floor_temp=floor_stats["mean"] if floor_stats else None,
                    output_path=str(qc_img_path),
                )

            processed += 1
            if processed % 100 == 0:
                log.info(f"  Processed {processed}/{n_sampled} frames")

    if not rows:
        raise RuntimeError(f"No frames were processed from {seq_path.name}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info(f"  CSV saved: {csv_path}")

    _log_summary(df, seq_path.name, entry_info)
    return df, csv_path, entry_info


def _log_summary(df: pd.DataFrame, filename: str, entry_info: dict) -> None:
    n_pre = int((df["qc_flag"] == "pre_entry").sum())
    tracking_df = df[df["qc_flag"] != "pre_entry"]
    n = len(tracking_df)
    n_mouse = int(tracking_df["mouse_roi_valid"].sum()) if n else 0
    n_floor = int(tracking_df["floor_roi_valid"].sum()) if n else 0
    pct_mouse = 100 * n_mouse / max(n, 1)
    pct_floor = 100 * n_floor / max(n, 1)
    med_conf = tracking_df["tracking_confidence"].median() if n else float("nan")

    log.info(f"\n  --- Summary: {filename} ---")
    log.info(
        f"  Tracking start : frame {entry_info.get('tracking_start_frame', 0)} "
        f"(~{entry_info.get('elapsed_before_tracking_sec', 0):.1f}s)  "
        f"[{entry_info.get('method', '?')}]"
    )
    log.info(f"  Pre-entry frames (NA): {n_pre}")
    log.info(f"  Tracked frames : {n}")
    log.info(f"  Valid mouse ROI: {n_mouse} ({pct_mouse:.1f}%)")
    log.info(f"  Valid floor ROI: {n_floor} ({pct_floor:.1f}%)")
    log.info(f"  Median confidence: {med_conf:.2f}")

    flags = df["qc_flag"].value_counts()
    log.info(f"  QC flag counts : {flags.to_dict()}")
