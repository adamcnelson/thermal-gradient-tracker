import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os

from qc_shared import SESSIONS, REPO, compute_candidates, render_example, pick_diverse

OUT_DIR = f"{REPO}/bouts/qc_plots/warm_spot_location_qc"
os.makedirs(OUT_DIR, exist_ok=True)
MAX_EXAMPLES_PER_SESSION = 6


def render_session(name, cfg, candidates):
    # warm_spot is only ever computed for "extended" posture (needs the full nose-to-tail
    # decomposition). qc_valid is NOT a useful filter here -- it gates the TAIL delta_t
    # measurement only, not warm-spot placement -- so don't prioritize by it; just prefer
    # samples that actually got a warm_spot value, spread across distinct bouts.
    pool = [r for r in candidates["extended"] if r.get("warm_spot") is not None]
    if not pool:
        pool = candidates["extended"]
    chosen = pick_diverse(pool, MAX_EXAMPLES_PER_SESSION, prefer_valid=False)

    for rec in chosen:
        valid_tag = "valid" if rec["qc_valid"] else "notvalid"
        fname = f"{cfg['session_label']}_{cfg['track']}_bout{rec['bout_index']:02d}_t{rec['thermal_t']:.1f}s_warmspot_{valid_tag}.png"
        title = (f"{name}  bout {rec['bout_index']}  t={rec['thermal_t']:.1f}s thermal  "
                 f"posture=extended (full skeleton)  tail_qc_valid={rec['qc_valid']}")
        if rec["dorsal_mean_c"] is not None:
            title += f"\ndorsal_mean={rec['dorsal_mean_c']:.2f}C"
        if rec["warm_spot"] is not None:
            title += f"  warm_spot(95th pct anterior)={rec['warm_spot']:.2f}C"
        else:
            title += "  warm_spot=None (warped anterior region had no valid thermal pixels)"
        render_example(
            f"{OUT_DIR}/{fname}", title,
            rec["crop"], rec["mask"], rec["dorsal"], rec["anterior"],
            rec["tail_centerline"], rec["prox_tail"], rec["nose_pt"], rec["tail_base_pt"],
            rec["thermal_celsius"], rec["warped_animal"], rec["warped_dorsal"], rec["warped_anterior"],
            rec["warped_prox_xy"], rec["tail_center_xy"],
            rec["tail_temp_c"], rec["floor_temp_c"], rec["delta_t_c"],
        )
        print(f"  saved {fname}", flush=True)
    print(f"{name}: {len(pool)} extended-with-warmspot candidates, {len(chosen)} examples rendered", flush=True)


if __name__ == "__main__":
    for name, cfg in SESSIONS.items():
        candidates = compute_candidates(name, cfg)
        render_session(name, cfg, candidates)
    print("\n=== DONE ===")
    print(f"output dir: {OUT_DIR}")
