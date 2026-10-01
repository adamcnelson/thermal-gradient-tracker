"""
QC review for the project_brief_v8.md §3.1 non-stationary sampling mode.

Renders a handful of real examples spread across time and posture type, using
the exact same per-sample computation and rendering code (qc_shared.py) as the
existing stationary-bout QC folders -- the only difference is the sample times
come from the non-stationary sampling plan (stage7_real_run.py's own logic,
reproduced here) instead of stationary bouts.

Usage: python scripts/render_nonstationary_qc.py [SESSION ...]  (default: all)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

import numpy as np
import pandas as pd

from qc_shared import SESSIONS, REPO, compute_candidates, render_example, render_notail_example
from src import paths
from src.analysis_config import AnalysisConfig
from src.bouts import classify_stationary
from src.velocity import compute_velocity
from src.logging_utils import setup_logger
from scripts.compute_bouts import _get_thresholds

OUT_DIR = f"{REPO}/bouts/qc_plots/nonstationary_qc"
os.makedirs(OUT_DIR, exist_ok=True)
N_PER_BUCKET = 3


def nonstationary_times(cfg):
    """Same recipe as stage7_real_run.py's non-stationary sampling plan."""
    log = setup_logger("render_nonstationary_qc")
    tracking_df = pd.read_csv(cfg["tracking_csv"])
    analysis_config = AnalysisConfig.load(str(paths.DEFAULT_ANALYSIS_CONFIG))
    disp_thresh, vel_thresh = _get_thresholds(analysis_config, log)
    vel_df = compute_velocity(tracking_df, analysis_config.bouts)
    stationary_mask = classify_stationary(vel_df, disp_thresh, vel_thresh)
    qc_ok = vel_df["qc_flag"] == "ok"
    roi_valid = vel_df["mouse_roi_valid"].astype(str) == "True"
    post_entry = vel_df["elapsed_time_sec"] >= cfg["entry_time_thermal_sec"]
    rows = vel_df[qc_ok & roi_valid & post_entry & ~stationary_mask]
    return [float(t) for t in rows["elapsed_time_sec"]]


def pick_spread(items, n):
    """Evenly spread across thermal_t, not bout_index (every non-stationary
    item shares bout_index=None, so pick_diverse's dedup key doesn't apply)."""
    items = sorted(items, key=lambda r: r["thermal_t"])
    if len(items) <= n:
        return items
    idxs = sorted(set(int(i) for i in np.linspace(0, len(items) - 1, n).round()))
    return [items[i] for i in idxs]


def bout_tag(bout_index):
    return f"bout{bout_index:02d}" if bout_index is not None else "nonstat"


def render_session(name, cfg, candidates):
    extended = [r for r in candidates["extended"] if r["bout_index"] is None]
    curled = [r for r in candidates["fallback"] if r["bout_index"] is None and r["posture"] == "curled"]
    ambiguous = [r for r in candidates["fallback"] if r["bout_index"] is None and r["posture"] == "ambiguous"]
    no_tail = [r for r in candidates["no_tail"] if r["bout_index"] is None]

    chosen = (pick_spread(extended, N_PER_BUCKET) + pick_spread(curled, N_PER_BUCKET)
              + pick_spread(ambiguous, N_PER_BUCKET) + pick_spread(no_tail, N_PER_BUCKET))

    for rec in chosen:
        if rec["kind"] == "no_tail":
            fname = f"{cfg['session_label']}_{cfg['track']}_{bout_tag(None)}_t{rec['thermal_t']:.1f}s_{rec['posture']}_NOTAIL.png"
            render_notail_example(f"{OUT_DIR}/{fname}", name, rec)
            print(f"  saved {fname}", flush=True)
            continue

        tag = "extended" if rec["posture"] == "extended" else "tailfallback"
        valid_tag = "valid" if rec["qc_valid"] else "notvalid"
        fname = (f"{cfg['session_label']}_{cfg['track']}_{bout_tag(None)}_t{rec['thermal_t']:.1f}s_"
                 f"{rec['posture']}_{tag}_{valid_tag}.png")
        title = (f"{name} NON-STATIONARY  t={rec['thermal_t']:.1f}s thermal  "
                 f"posture={rec['posture']} ({'full skeleton' if rec['posture']=='extended' else 'tail-only fallback'})  "
                 f"qc_valid={rec['qc_valid']}")
        if rec["dorsal_mean_c"] is not None:
            title += f"\ndorsal_mean={rec['dorsal_mean_c']:.2f}C"
        if rec["warm_spot"] is not None:
            title += f"  warm_spot(95th pct anterior)={rec['warm_spot']:.2f}C"
        render_example(
            f"{OUT_DIR}/{fname}", title,
            rec["crop"], rec["mask"], rec["dorsal"], rec["anterior"],
            rec["tail_centerline"], rec["prox_tail"], rec["nose_pt"], rec["tail_base_pt"],
            rec["thermal_celsius"], rec["warped_animal"], rec["warped_dorsal"], rec["warped_anterior"],
            rec["warped_prox_xy"], rec["tail_center_xy"],
            rec["tail_temp_c"], rec["floor_temp_c"], rec["delta_t_c"],
        )
        print(f"  saved {fname}", flush=True)
    print(f"{name}: {len(extended)} extended + {len(curled)} curled + {len(ambiguous)} ambiguous + "
          f"{len(no_tail)} no-tail non-stationary candidates, {len(chosen)} examples rendered", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(SESSIONS.keys())
    for name in names:
        cfg = SESSIONS[name]
        extra = nonstationary_times(cfg)
        candidates = compute_candidates(name, cfg, extra_samples=extra)
        render_session(name, cfg, candidates)
    print("\n=== DONE ===")
    print(f"output dir: {OUT_DIR}")
