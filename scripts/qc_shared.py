"""
Shared per-sample computation + rendering for the three "measurement location"
QC folders (tail, warm-spot, dorsal surface). Consolidated 2026-08-28 (Adam:
"save the bouts/qc_plots as well as the QC plots for warm spot, base of tail,
and the dorsal surface" -- now across 6 sessions, not 3, so the per-sample
computation that used to be copy-pasted into each render_*_qc.py script is
pulled into ONE place instead: each render script would otherwise redo the
same expensive full per-sample pass (segmentation, homography warp, thermal
sampling) three times per session for no reason.

SESSIONS and the registration-correction helpers are imported directly from
stage7_real_run.py (guarded behind __main__, safe to import) rather than
duplicated -- that dict is the single source of truth for session configs
going forward.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.seq_io import SeqReader, read_planck_constants, raw_to_celsius
from src.landmarks.bout_gating import iter_stationary_bouts, filter_bouts_after_entry
from src.landmarks.webcam_preprocessing import split_track_crops, detect_track_split_row, LANE_TOP
from src.landmarks.rgb_landmarks import RgbBackgroundModel, segment_mouse_rgb, extract_mouse_detection
from src.landmarks.thermal_measurement import (
    anterior_region_mask, dorsal_surface_mask, proximal_tail_points,
    warp_mask_to_thermal, warp_points_xy, warm_spot_temperature,
    dorsal_surface_temperature, tail_base_delta_t, gate_measurement,
)
from src.landmarks.outputs import thermal_time_to_rgb_time

from stage7_real_run import (
    SESSIONS, load_homography, build_thermal_native_lookup, registration_correction_homography,
    local_edge_refine,
    MIN_AREA, MAX_AREA, SAMPLE_RADIUS_PX, FLOOR_INNER_RADIUS_PX, FLOOR_OUTER_RADIUS_PX,
    FRAMES_PER_BOUT, BOUT_SAMPLE_FRACS, MAX_PRUNE_PX, DORSAL_STRICT_SIGMA, THERMAL_FPS, REPO,
)


def mask_contour_pts(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def render_example(out_path, title, crop_gray, mask, dorsal_mask, anterior_mask,
                    tail_centerline, prox_tail_pts, nose_pt, tail_base_pt,
                    thermal_celsius, warped_animal, warped_dorsal, warped_anterior,
                    warped_prox_tail_xy, tail_sample_center_xy, tail_temp_c, floor_temp_c, delta_t_c):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))

    # ---- Left: RGB crop ----
    axL.imshow(crop_gray, cmap="gray")
    for c in mask_contour_pts(mask):
        axL.plot(c[:, 0, 0], c[:, 0, 1], color="lime", linewidth=1.5)
    if dorsal_mask is not None:
        overlay = np.zeros((*dorsal_mask.shape, 4))
        overlay[dorsal_mask] = (0.2, 0.4, 1.0, 0.40)
        axL.imshow(overlay)
    if anterior_mask is not None:
        for c in mask_contour_pts(anterior_mask):
            axL.plot(c[:, 0, 0], c[:, 0, 1], color="white", linewidth=2.0, linestyle="--")
    if tail_centerline is not None:
        tc = np.array(tail_centerline)
        axL.plot(tc[:, 1], tc[:, 0], color="yellow", linewidth=1.5)
    if prox_tail_pts is not None:
        pt = np.array(prox_tail_pts)
        axL.plot(pt[:, 1], pt[:, 0], color="magenta", linewidth=2.5)
    if nose_pt is not None:
        axL.plot(nose_pt[1], nose_pt[0], marker="o", color="cyan", markersize=7, markeredgecolor="black")
    if tail_base_pt is not None:
        axL.plot(tail_base_pt[1], tail_base_pt[0], marker="*", color="magenta", markersize=13, markeredgecolor="black")
    axL.set_title("RGB (green=mask outline, blue fill=dorsal, white dashed=warm-spot region,\nmagenta=proximal tail sample, yellow=full tail centerline)", fontsize=8)
    axL.axis("off")
    ys, xs = np.where(mask)
    if len(ys):
        pad = 30
        axL.set_xlim(max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad))
        axL.set_ylim(min(mask.shape[0], ys.max() + pad), max(0, ys.min() - pad))

    # ---- Right: thermal ----
    im = axR.imshow(thermal_celsius, cmap="inferno")
    plt.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="°C")
    for c in mask_contour_pts(warped_animal):
        axR.plot(c[:, 0, 0], c[:, 0, 1], color="lime", linewidth=1.2)
    if warped_dorsal is not None:
        overlay = np.zeros((*warped_dorsal.shape, 4))
        overlay[warped_dorsal] = (0.2, 0.4, 1.0, 0.40)
        axR.imshow(overlay)
    if warped_anterior is not None:
        for c in mask_contour_pts(warped_anterior):
            axR.plot(c[:, 0, 0], c[:, 0, 1], color="white", linewidth=2.0, linestyle="--")
    if warped_prox_tail_xy is not None:
        for x, y in warped_prox_tail_xy:
            circ = plt.Circle((x, y), SAMPLE_RADIUS_PX, color="magenta", fill=False, linewidth=1.5)
            axR.add_patch(circ)
    if tail_sample_center_xy is not None:
        cx, cy = tail_sample_center_xy
        axR.plot(cx, cy, marker="*", color="gold", markersize=11, markeredgecolor="black")
        inner = plt.Circle((cx, cy), FLOOR_INNER_RADIUS_PX, color="cyan", fill=False, linewidth=1.2, linestyle="--")
        outer = plt.Circle((cx, cy), FLOOR_OUTER_RADIUS_PX, color="cyan", fill=False, linewidth=1.2, linestyle="--")
        axR.add_patch(inner)
        axR.add_patch(outer)
    axR.set_title("Thermal (magenta circles=tail sample pts, gold star=tail sample center,\ncyan dashed annulus=local floor ref, white dashed=warm-spot region)", fontsize=8)
    axR.axis("off")
    ys, xs = np.where(warped_animal)
    if len(ys):
        pad = 40
        axR.set_xlim(max(0, xs.min() - pad), min(thermal_celsius.shape[1], xs.max() + pad))
        axR.set_ylim(min(thermal_celsius.shape[0], ys.max() + pad), max(0, ys.min() - pad))

    meas_txt = []
    if tail_temp_c is not None:
        meas_txt.append(f"tail={tail_temp_c:.2f}C floor={floor_temp_c:.2f}C ΔT={delta_t_c:+.2f}C")
    fig.suptitle(title + ("\n" + " | ".join(meas_txt) if meas_txt else ""), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_notail_example(out_path, name, rec):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rec["crop"], cmap="gray")
    for c in mask_contour_pts(rec["mask"]):
        ax.plot(c[:, 0, 0], c[:, 0, 1], color="red", linewidth=1.5)
    ax.set_title(f"{name} bout {rec['bout_index']} t={rec['thermal_t']:.1f}s\nposture={rec['posture']}, NO TAIL FOUND (no measurement)", fontsize=9)
    ax.axis("off")
    ys, xs = np.where(rec["mask"])
    if len(ys):
        pad = 30
        ax.set_xlim(max(0, xs.min() - pad), min(rec["mask"].shape[1], xs.max() + pad))
        ax.set_ylim(min(rec["mask"].shape[0], ys.max() + pad), max(0, ys.min() - pad))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def compute_candidates(name, cfg):
    """Runs the full per-sample pass ONCE for a session; returns
    {"extended": [...], "fallback": [...], "no_tail": [...]} -- each render_*_qc.py
    script applies its own selection logic to this same shared result."""
    print(f"\n=== {name} (computing candidates) ===", flush=True)
    tracking_df = pd.read_csv(cfg["tracking_csv"])
    bouts_df_raw = pd.read_csv(cfg["bouts_csv"])
    bouts_df = filter_bouts_after_entry(bouts_df_raw, entry_time_sec=cfg["entry_time_thermal_sec"])
    # stage7_real_run.py's load_homography() returns H only (not a tuple) --
    # rmse_px is read separately, matching that module's own convention.
    H = load_homography(f"{REPO}/{cfg['homography_json']}")
    homography_rmse = json.load(open(f"{REPO}/{cfg['homography_json']}"))["rmse_px"]
    thermal_native_lookup = build_thermal_native_lookup(tracking_df)
    sync_result = cfg["sync_result"]

    bout_frame_plan = []
    for bout, frames in iter_stationary_bouts(tracking_df, bouts_df):
        b0, b1 = bout["bout_start_sec"], bout["bout_end_sec"]
        for frac in BOUT_SAMPLE_FRACS[:FRAMES_PER_BOUT]:
            bout_frame_plan.append((int(bout["bout_index"]), b0 + frac * (b1 - b0)))

    thermal_idx_wanted = sorted({int(round(t * THERMAL_FPS)) for _, t in bout_frame_plan})
    idx_set = set(thermal_idx_wanted)
    planck = read_planck_constants(cfg["thermal_seq"])
    thermal_frames = {}
    reader = SeqReader(cfg["thermal_seq"])
    for idx, raw in reader.frames():
        if idx in idx_set:
            thermal_frames[idx] = raw_to_celsius(raw, planck)
            if len(thermal_frames) == len(idx_set):
                break
    reader.close()

    cap = cv2.VideoCapture(cfg["rgb_video"])
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    total_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    split_row = detect_track_split_row(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
    bg_indices = np.linspace(0, total_rgb - 1, 30, dtype=int)
    bg_frames = []
    for idx in bg_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top, bottom = split_track_crops(gray, split_row=split_row)
            bg_frames.append(top if cfg["track"] == LANE_TOP else bottom)
    bg_model = RgbBackgroundModel.build(bg_frames)

    candidates = {"extended": [], "fallback": [], "no_tail": []}

    for bout_index, thermal_t in bout_frame_plan:
        thermal_idx = int(round(thermal_t * THERMAL_FPS))
        thermal_celsius = thermal_frames.get(thermal_idx)
        if thermal_celsius is None:
            continue
        rgb_t = thermal_time_to_rgb_time(thermal_t, sync_result)
        rgb_frame_idx = int(round(rgb_t * rgb_fps))
        if not (0 <= rgb_frame_idx < total_rgb):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top, bottom = split_track_crops(gray, split_row=split_row)
        crop = top if cfg["track"] == LANE_TOP else bottom
        mask = segment_mouse_rgb(crop, bg_model, min_area=MIN_AREA, max_area=MAX_AREA)
        if mask is None:
            continue
        dorsal_mask_source = segment_mouse_rgb(crop, bg_model, min_area=MIN_AREA, max_area=MAX_AREA,
                                                threshold_sigma=DORSAL_STRICT_SIGMA)
        if dorsal_mask_source is None:
            dorsal_mask_source = mask
        detection = extract_mouse_detection(mask, MIN_AREA, MAX_AREA, max_prune_px=MAX_PRUNE_PX)
        if detection is None:
            continue
        posture = detection.posture
        lm = detection.landmarks
        tail_lm = detection.tail_landmarks

        if posture != "extended" and tail_lm is None:
            candidates["no_tail"].append(dict(kind="no_tail", bout_index=bout_index, thermal_t=thermal_t, thermal_idx=thermal_idx,
                                               crop=crop, mask=mask, posture=posture))
            continue

        thermal_native_xy = thermal_native_lookup(thermal_t)
        H_sample, correction = registration_correction_homography(H, mask, thermal_celsius.shape, thermal_native_xy)

        warped_animal = warp_mask_to_thermal(mask, H_sample, thermal_celsius.shape)
        edge_dx, edge_dy, edge_improved, _, _ = local_edge_refine(thermal_celsius, warped_animal)
        if edge_improved:
            T_edge = np.array([[1, 0, edge_dx], [0, 1, edge_dy], [0, 0, 1]], dtype=np.float64)
            H_sample = T_edge @ H_sample
            warped_animal = warp_mask_to_thermal(mask, H_sample, thermal_celsius.shape)
        if posture == "extended":
            anterior = anterior_region_mask(dorsal_mask_source, lm.path_nose_to_tail[: lm.tail_base_index + 1])
            dorsal = dorsal_surface_mask(dorsal_mask_source, lm.path_nose_to_tail, lm.tail_base_index)
            prox_tail = proximal_tail_points(lm.tail_centerline)
            warped_anterior = warp_mask_to_thermal(anterior, H_sample, thermal_celsius.shape)
            warm_spot = warm_spot_temperature(thermal_celsius, warped_anterior)
            nose_pt = lm.nose_point
            tail_centerline = lm.tail_centerline
            tail_base_pt = lm.tail_base_point
        else:
            dorsal = dorsal_mask_source
            anterior = None
            warped_anterior = None
            warm_spot = None
            prox_tail = proximal_tail_points(tail_lm.tail_centerline)
            nose_pt = None
            tail_centerline = tail_lm.tail_centerline
            tail_base_pt = tail_lm.tail_base_point

        tail_meas = tail_base_delta_t(
            thermal_celsius, H_sample, prox_tail,
            sample_radius_px=SAMPLE_RADIUS_PX,
            floor_inner_radius_px=FLOOR_INNER_RADIUS_PX,
            floor_outer_radius_px=FLOOR_OUTER_RADIUS_PX,
            animal_mask_thermal=warped_animal,
        )
        warped_dorsal = warp_mask_to_thermal(dorsal, H_sample, thermal_celsius.shape)
        dorsal_meas = dorsal_surface_temperature(thermal_celsius, warped_dorsal)

        landmark_confidence = 1.0 if posture == "extended" else 0.7
        posture_ok = (posture == "extended") or (tail_lm is not None)
        qc_valid, _ = gate_measurement(
            delta_t_c=tail_meas.delta_t_c, landmark_confidence=landmark_confidence,
            sync_qc_pass=True, homography_qc_pass=homography_rmse < 2.0, posture_ok=posture_ok,
        )

        warped_prox_xy = warp_points_xy(prox_tail, H_sample)
        tail_center_xy = tuple(warped_prox_xy.mean(axis=0)) if len(warped_prox_xy) else None

        rec = dict(
            kind="measured",
            bout_index=bout_index, thermal_t=thermal_t, thermal_idx=thermal_idx,
            crop=crop, mask=mask, posture=posture, qc_valid=qc_valid,
            dorsal=dorsal, anterior=anterior, tail_centerline=tail_centerline,
            prox_tail=prox_tail, nose_pt=nose_pt, tail_base_pt=tail_base_pt,
            thermal_celsius=thermal_celsius, warped_animal=warped_animal,
            warped_dorsal=warped_dorsal, warped_anterior=warped_anterior,
            warped_prox_xy=warped_prox_xy, tail_center_xy=tail_center_xy,
            tail_temp_c=tail_meas.tail_temp_c, floor_temp_c=tail_meas.floor_temp_c,
            delta_t_c=tail_meas.delta_t_c, dorsal_mean_c=dorsal_meas.mean_c, warm_spot=warm_spot,
        )
        bucket = "extended" if posture == "extended" else "fallback"
        candidates[bucket].append(rec)

    cap.release()
    print(f"{name}: {len(candidates['extended'])} extended, {len(candidates['fallback'])} fallback, "
          f"{len(candidates['no_tail'])} no-tail candidates", flush=True)
    return candidates


def pick_diverse(items, n, prefer_valid=True, key="bout_index"):
    """Prefer qc_valid items (if prefer_valid), spread across distinct `key` values."""
    pool = [r for r in items if r.get("qc_valid")] if prefer_valid else list(items)
    if prefer_valid and len(pool) < n:
        pool = list(items)
    chosen, seen = [], set()
    for r in pool:
        if r[key] not in seen:
            chosen.append(r)
            seen.add(r[key])
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for r in pool:
            if r not in chosen:
                chosen.append(r)
            if len(chosen) >= n:
                break
    return chosen
