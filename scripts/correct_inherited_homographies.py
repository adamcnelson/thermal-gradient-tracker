"""
Per-lane constant translation correction for the inherited (Hybrid) homographies.

Each inherited homography carries its base calibration's constant RGB->thermal
offset (found 2026-10-05 via sync QC: Regime B Front ~-22px in x, Regime A Front
~+28px in y in a 43px strip). With a lane's sync known
(thermalFeatures/trajectory_sync/session_sync.csv), the offset is measurable:
median(thermal-native centroid - RGB centroid warped through H) over samples
where both trackers are on the mouse (x within AGREE_PX of the median difference
-- excludes stretches where the thermal tracker latches onto an arena object).
The correction is folded into H as T @ H (registration.translate_homography).

Stability check: the offset is measured separately on each half of the session;
a real calibration offset is constant, so the halves should agree (|diff| <=
MAX_HALF_DIFF_PX, else the lane is marked unstable).

Writes corrected copies -- originals in homography_calibration_inherited/ are
never modified:
    homography_calibration_inherited_corrected/<same name>.json
        H                      corrected matrix (what downstream code reads)
        H_uncorrected          the inherited matrix
        translation_correction_px, translation_correction_halves_px, ...
    thermalFeatures/trajectory_sync/homography_corrections.csv   (summary)

Usage:
    python scripts/correct_inherited_homographies.py [--tracking-dir ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.landmarks.registration import apply_homography, translate_homography

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "homography_calibration_inherited"
OUT_DIR = REPO / "homography_calibration_inherited_corrected"
SYNC_DIR = REPO / "thermalFeatures" / "trajectory_sync"
DEFAULT_TRACKING_DIR = ("/Volumes/alcova/bedfordlab/ThermalGradient/SLURM_RESULTS/"
                        "results_fullrun_mgms2_2026-07-28/trackingOutputs")
LANE_NAMES = {"F": "Front", "B": "Back"}
AGREE_PX = 15.0
MAX_GAP_SEC = 3.0
MAX_HALF_DIFF_PX = 3.0


def paired_positions(tracking_csv, rgb_track_csv, H, fps, drift, offset):
    """Thermal-native (x, y) and RGB-warped (x, y) at the same moments, under the given sync."""
    d = pd.read_csv(tracking_csv)
    d = d[(d.tracking_confidence == 1.0) & d.mouse_centroid_x.notna() & d.mouse_centroid_y.notna()]
    r = pd.read_csv(rgb_track_csv).dropna(subset=["rgb_centroid_x"]).sort_values("rgb_time_sec")
    rt = r.rgb_time_sec.to_numpy(float)
    warped = apply_homography(H, r[["rgb_centroid_x", "rgb_centroid_y"]].to_numpy(float))
    q = d.frame_number.to_numpy(float) / fps * (1 + drift) + offset
    i1 = np.clip(np.searchsorted(rt, q), 1, len(rt) - 1)
    ok = (q >= rt[0]) & (q <= rt[-1]) & ((rt[i1] - rt[i1 - 1]) <= MAX_GAP_SEC)
    rgb = np.column_stack([np.interp(q[ok], rt, warped[:, 0]), np.interp(q[ok], rt, warped[:, 1])])
    thermal = d[["mouse_centroid_x", "mouse_centroid_y"]].to_numpy(float)[ok]
    return q[ok], thermal, rgb


def measure_offset(t, thermal, rgb):
    """(dx, dy) to add to RGB-warped positions, overall and per session half, + agreeing sample count."""
    diff = thermal - rgb
    agree = np.abs(diff[:, 0] - np.median(diff[:, 0])) < AGREE_PX
    mid = np.median(t[agree])
    halves = [np.median(diff[agree & sel], axis=0) for sel in (t < mid, t >= mid)]
    return np.median(diff[agree], axis=0), halves, int(agree.sum())


def main():
    parser = argparse.ArgumentParser(description="Per-lane translation correction of inherited homographies")
    parser.add_argument("--tracking-dir", default=DEFAULT_TRACKING_DIR)
    args = parser.parse_args()

    sync = pd.read_csv(SYNC_DIR / "session_sync.csv").set_index("session")
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for hj in sorted(SRC_DIR.glob("*_homography.json")):
        meta = json.load(open(hj))
        stem = hj.name.replace("_homography.json", "")
        session, lane = stem.rsplit("_", 1)[0], meta["lane"]
        s = sync.loc[session]
        H = np.array(meta["H"], dtype=np.float64)
        t, thermal, rgb = paired_positions(
            f"{args.tracking_dir}/{stem}_tracking_every10frames.csv",
            SYNC_DIR / "rgb_tracks" / f"{session}_{lane}_rgb_track.csv",
            H, s.camera_fps, s.drift_slope, s.offset_sec,
        )
        (dx, dy), halves, n = measure_offset(t, thermal, rgb)
        half_diff = float(np.max(np.abs(halves[0] - halves[1])))
        stable = half_diff <= MAX_HALF_DIFF_PX
        out = dict(meta)
        out.update(
            H=translate_homography(H, dx, dy).tolist(),
            H_uncorrected=meta["H"],
            translation_correction_px=[round(float(dx), 2), round(float(dy), 2)],
            translation_correction_halves_px=[[round(float(v), 2) for v in h] for h in halves],
            translation_correction_stable=bool(stable),
            translation_correction_n_samples=n,
            translation_correction_note=(
                "Constant offset of the inherited base calibration, measured at the session's "
                "trajectory sync (offset %+.2fs, drift %+.5f) as median(thermal-native - RGB-warped "
                "centroid) over %d samples where both trackers agree in x; folded in as T@H. "
                "Original matrix in H_uncorrected. scripts/correct_inherited_homographies.py"
                % (s.offset_sec, s.drift_slope, n)),
            source="inherited_base_regime+translation_correction",
        )
        with open(OUT_DIR / hj.name, "w") as f:
            json.dump(out, f, indent=2)
        rows.append(dict(session=session, lane=lane, regime=meta.get("regime", ""), dx_px=dx, dy_px=dy,
                         dx_half1=halves[0][0], dx_half2=halves[1][0], dy_half1=halves[0][1],
                         dy_half2=halves[1][1], max_half_diff_px=half_diff, stable=stable, n_samples=n))

    df = pd.DataFrame(rows)
    df.to_csv(SYNC_DIR / "homography_corrections.csv", index=False)
    print(df.groupby(["regime", "lane"]).agg(
        n=("dx_px", "size"), dx_mean=("dx_px", "mean"), dx_min=("dx_px", "min"), dx_max=("dx_px", "max"),
        dy_mean=("dy_px", "mean"), dy_min=("dy_px", "min"), dy_max=("dy_px", "max"),
        max_half_diff=("max_half_diff_px", "max"), n_unstable=("stable", lambda s: int((~s).sum())),
    ).round(1).to_string())
    unstable = df[~df.stable]
    if len(unstable):
        print("\nUnstable lanes (halves disagree by >%.0fpx):" % MAX_HALF_DIFF_PX)
        print(unstable[["session", "lane", "dx_half1", "dx_half2", "dy_half1", "dy_half2"]].round(1).to_string(index=False))
    print(f"\n{len(df)} corrected homographies -> {OUT_DIR}")


if __name__ == "__main__":
    main()
