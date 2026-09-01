import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os

from qc_shared import SESSIONS, REPO, compute_candidates, render_example, render_notail_example, pick_diverse

OUT_DIR = f"{REPO}/bouts/qc_plots/measurement_location_qc"
os.makedirs(OUT_DIR, exist_ok=True)


def render_session(name, cfg, candidates):
    chosen = (pick_diverse(candidates["extended"], 2)
              + pick_diverse(candidates["fallback"], 3)
              + candidates["no_tail"][:1])

    for rec in chosen:
        if rec["kind"] == "no_tail":
            fname = f"{cfg['session_label']}_{cfg['track']}_bout{rec['bout_index']:02d}_t{rec['thermal_t']:.1f}s_{rec['posture']}_NOTAIL.png"
            render_notail_example(f"{OUT_DIR}/{fname}", name, rec)
            print(f"  saved {fname}", flush=True)
            continue

        tag = "extended" if rec["posture"] == "extended" else "tailfallback"
        valid_tag = "valid" if rec["qc_valid"] else "notvalid"
        fname = f"{cfg['session_label']}_{cfg['track']}_bout{rec['bout_index']:02d}_t{rec['thermal_t']:.1f}s_{rec['posture']}_{tag}_{valid_tag}.png"
        title = (f"{name}  bout {rec['bout_index']}  t={rec['thermal_t']:.1f}s thermal  "
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
    print(f"{name}: {len(chosen)} tail-QC examples rendered", flush=True)


if __name__ == "__main__":
    for name, cfg in SESSIONS.items():
        candidates = compute_candidates(name, cfg)
        render_session(name, cfg, candidates)
    print("\n=== DONE ===")
    print(f"output dir: {OUT_DIR}")
