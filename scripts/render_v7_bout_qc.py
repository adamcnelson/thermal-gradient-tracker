import sys
sys.path.insert(0, "/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/thermal-gradient-tracker")

import numpy as np
import pandas as pd

from src.landmarks.bout_qc import save_v7_bout_diagnostic
from src.landmarks.bout_gating import filter_bouts_after_entry
from src.landmarks.sync import WindowedSyncResult
from src.landmarks.outputs import rgb_time_to_thermal_time
from src.velocity import compute_velocity
from src.analysis_config import AnalysisConfig

REPO = "/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/thermal-gradient-tracker"

# Same Stage 3 sync results used by stage7_real_run.py -- rgb_track_for_qc.py's
# CSVs carry raw RGB-video-native rgb_time_sec (frame_idx / fps against the
# webcam's own clock), NOT thermal time, so they must go through the same
# rgb_time_to_thermal_time() affine map as every other RGB-derived timestamp
# in this pipeline before they can be overlaid on tracking_df/bouts_df's
# thermal-clock elapsed_time_sec axis. Getting this wrong would silently
# reintroduce exactly the kind of cross-clock misalignment the new panels
# exist to help diagnose.
SYNC_RESULTS = {
    "Test_3": WindowedSyncResult(
        window_centers_sec=np.array([]), window_lags_sec=np.array([]),
        offset_sec=-2.8, drift_slope=0.0, r_squared=0.999, residual_max_sec=0.2,
        low_confidence=False,
        confidence_note="resolved 2026-08-25: 5 real anchors post camera_fps 8->10 fix, std=0.13s",
    ),
    "Test_4": WindowedSyncResult(
        window_centers_sec=np.array([]), window_lags_sec=np.array([]),
        offset_sec=5.5, drift_slope=0.0, r_squared=0.995, residual_max_sec=0.5,
        low_confidence=False,
        confidence_note="resolved 2026-08-25: 4 real anchors post camera_fps 8->10 fix, std=0.46s",
    ),
    "Test_7": WindowedSyncResult(
        window_centers_sec=np.array([]), window_lags_sec=np.array([]),
        offset_sec=-0.51, drift_slope=0.0, r_squared=0.9999, residual_max_sec=0.5,
        low_confidence=False,
        confidence_note="resolved 2026-08-26: 4 real anchors, fps already correct at 10.0, std<0.5s",
    ),
}
# Back lanes reuse their session's Front sync_result directly: Front/Back .seq files are
# separate exports of the SAME underlying recording (identical frame counts, confirmed in
# stage7_real_run.py), so camera_fps and the RGB<->thermal offset apply unchanged. Only the
# homography (different floor region) and entry_time (per-mouse) are independently derived.
SYNC_RESULTS["Test_3_Back"] = SYNC_RESULTS["Test_3"]
SYNC_RESULTS["Test_4_Back"] = SYNC_RESULTS["Test_4"]
SYNC_RESULTS["Test_7_Back"] = SYNC_RESULTS["Test_7"]

SESSIONS = {
    "Test_3": dict(
        tracking_csv=f"{REPO}/trackingOutputs/07-28-25_4540_B_4541_F_Test3-004_Front_tracking_every10frames.csv",
        bouts_csv=f"{REPO}/bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Front_bout_rows.csv",
        bout_output_csv=f"{REPO}/landmark_outputs/07-28-25_4540_B_4541_F_Test3-004_F_bout_output.csv",
        rgb_track_csv=f"{REPO}/landmark_outputs/07-28-25_4540_B_4541_F_Test3-004_F_rgb_track.csv",
        entry_time_thermal_sec=34.85,
        out=f"{REPO}/bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Front_v7_bouts_diagnostic",
        title="07-28-25_Test_3 (nestlet)",
    ),
    "Test_4": dict(
        tracking_csv=f"{REPO}/trackingOutputs/07-30-25_4540_F_4541_B_Test4-008_Front_tracking_every10frames.csv",
        bouts_csv=f"{REPO}/bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Front_bout_rows.csv",
        bout_output_csv=f"{REPO}/landmark_outputs/07-30-25_4540_F_4541_B_Test4-008_F_bout_output.csv",
        rgb_track_csv=f"{REPO}/landmark_outputs/07-30-25_4540_F_4541_B_Test4-008_F_rgb_track.csv",
        entry_time_thermal_sec=77.33,
        out=f"{REPO}/bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Front_v7_bouts_diagnostic",
        title="07-30-25_Test_4 (no nestlet)",
        config="analysis_config_fps10_scoped.json",
    ),
    "Test_7": dict(
        tracking_csv=f"{REPO}/trackingOutputs/08-07-25_4541_F_4540_B_Test7-020_Front_tracking_every10frames.csv",
        bouts_csv=f"{REPO}/bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Front_bout_rows.csv",
        bout_output_csv=f"{REPO}/landmark_outputs/08-07-25_4541_F_4540_B_Test7-020_F_bout_output.csv",
        rgb_track_csv=f"{REPO}/landmark_outputs/08-07-25_4541_F_4540_B_Test7-020_F_rgb_track.csv",
        entry_time_thermal_sec=28.0,
        out=f"{REPO}/bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Front_v7_bouts_diagnostic",
        title="08-07-25_Test_7 (Test7-020)",
        config="analysis_config.json",  # plain default -- Test_7's velocities were never on the wrong timescale
    ),
}
SESSIONS["Test_3"]["config"] = "analysis_config_fps10_scoped.json"

# Back-lane sessions, added 2026-08-28 alongside the warm-spot/tail/dorsal QC folders.
# entry_time_thermal_sec values here are auto-detected (Stage 1/2), NOT human-verified
# anchors like the Front sessions' -- same caveat noted in stage7_real_run.py's SESSIONS.
SESSIONS["Test_3_Back"] = dict(
    tracking_csv=f"{REPO}/trackingOutputs/07-28-25_4540_B_4541_F_Test3-004_Back_tracking_every10frames.csv",
    bouts_csv=f"{REPO}/bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Back_bout_rows.csv",
    bout_output_csv=f"{REPO}/landmark_outputs/07-28-25_4540_B_4541_F_Test3-004_B_bout_output.csv",
    rgb_track_csv=f"{REPO}/landmark_outputs/07-28-25_4540_B_4541_F_Test3-004_B_rgb_track.csv",
    entry_time_thermal_sec=71.0,
    out=f"{REPO}/bouts/qc_plots/07-28-25_4540_B_4541_F_Test3-004_Back_v7_bouts_diagnostic",
    title="07-28-25_Test_3 Back (nestlet)",
    config="analysis_config_fps10_scoped.json",
)
SESSIONS["Test_4_Back"] = dict(
    tracking_csv=f"{REPO}/trackingOutputs/07-30-25_4540_F_4541_B_Test4-008_Back_tracking_every10frames.csv",
    bouts_csv=f"{REPO}/bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Back_bout_rows.csv",
    bout_output_csv=f"{REPO}/landmark_outputs/07-30-25_4540_F_4541_B_Test4-008_B_bout_output.csv",
    rgb_track_csv=f"{REPO}/landmark_outputs/07-30-25_4540_F_4541_B_Test4-008_B_rgb_track.csv",
    entry_time_thermal_sec=81.0,
    out=f"{REPO}/bouts/qc_plots/07-30-25_4540_F_4541_B_Test4-008_Back_v7_bouts_diagnostic",
    title="07-30-25_Test_4 Back (no nestlet)",
    config="analysis_config_fps10_scoped.json",
)
SESSIONS["Test_7_Back"] = dict(
    tracking_csv=f"{REPO}/trackingOutputs/08-07-25_4541_F_4540_B_Test7-020_Back_tracking_every10frames.csv",
    bouts_csv=f"{REPO}/bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Back_bout_rows.csv",
    bout_output_csv=f"{REPO}/landmark_outputs/08-07-25_4541_F_4540_B_Test7-020_B_bout_output.csv",
    rgb_track_csv=f"{REPO}/landmark_outputs/08-07-25_4541_F_4540_B_Test7-020_B_rgb_track.csv",
    entry_time_thermal_sec=132.0,
    out=f"{REPO}/bouts/qc_plots/08-07-25_4541_F_4540_B_Test7-020_Back_v7_bouts_diagnostic",
    title="08-07-25_Test_7 Back (Test7-020)",
    config="analysis_config.json",
)

for name, cfg in SESSIONS.items():
    acfg = AnalysisConfig.load(f"{REPO}/{cfg['config']}")
    tracking_df = pd.read_csv(cfg["tracking_csv"])
    tracking_df = compute_velocity(tracking_df, acfg.bouts)
    bouts_df_raw = pd.read_csv(cfg["bouts_csv"])
    bouts_df = filter_bouts_after_entry(bouts_df_raw, entry_time_sec=cfg["entry_time_thermal_sec"])
    bout_output_df = pd.read_csv(cfg["bout_output_csv"])

    rgb_track_df = pd.read_csv(cfg["rgb_track_csv"])
    sync_result = SYNC_RESULTS[name]
    rgb_track_df["rgb_time_sec"] = rgb_track_df["rgb_time_sec"].apply(
        lambda rgb_t: rgb_time_to_thermal_time(rgb_t, sync_result)
    )

    save_v7_bout_diagnostic(
        tracking_df, bouts_df, bout_output_df, cfg["out"],
        title_extra=cfg["title"], rgb_track_df=rgb_track_df,
        rgb_sync_anchor_thermal_sec=cfg["entry_time_thermal_sec"],
    )
    print(f"{name}: saved {cfg['out']}.png / .pdf "
          f"({len(bouts_df)} bouts shaded, {len(bout_output_df)} bout measurements plotted, "
          f"{len(rgb_track_df)} RGB track samples, offset_sec={sync_result.offset_sec})")

print("DONE")
