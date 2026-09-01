import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os

from qc_shared import SESSIONS, REPO, compute_candidates, render_example, pick_diverse

OUT_DIR = f"{REPO}/bouts/qc_plots/dorsal_surface_qc"
os.makedirs(OUT_DIR, exist_ok=True)
# New folder, 2026-08-28 (Adam: "save... the QC plots for warm spot, base of tail, and the
# dorsal surface: the three key measurements"). Unlike warm-spot (extended-only), dorsal is
# computed for EVERY posture (extended via dorsal_surface_mask minus tail; curled/ambiguous
# via the whole dorsal_mask_source) -- so selection here deliberately spans both, not just
# extended, to show the full range of what the dorsal boundary actually looks like.
MAX_PER_POSTURE_GROUP = 3
# 2026-08-31 (Adam, after the zero-extended-posture investigation --
# see [[project-v7-extended-posture-skeleton-fragility]]): "ambiguous" frames are
# aspect-qualified as extended but get rejected because the skeleton has >2 endpoints
# (legs/ears resolved as separate branches) -- Adam wants to review a larger, dedicated
# sample of these directly, since some look just as extended to the eye as the frames
# that pass the strict skeleton check. Given its own (larger) budget, separate from curled.
MAX_AMBIGUOUS_PER_SESSION = 8


def render_session(name, cfg, candidates):
    extended_pool = [r for r in candidates["extended"] if r.get("dorsal_mean_c") is not None]
    curled_pool = [r for r in candidates["fallback"]
                   if r.get("dorsal_mean_c") is not None and r["posture"] == "curled"]
    ambiguous_pool = [r for r in candidates["fallback"]
                       if r.get("dorsal_mean_c") is not None and r["posture"] == "ambiguous"]
    chosen = (pick_diverse(extended_pool, MAX_PER_POSTURE_GROUP, prefer_valid=False)
              + pick_diverse(curled_pool, MAX_PER_POSTURE_GROUP, prefer_valid=False)
              + pick_diverse(ambiguous_pool, MAX_AMBIGUOUS_PER_SESSION, prefer_valid=False))

    for rec in chosen:
        valid_tag = "valid" if rec["qc_valid"] else "notvalid"
        fname = f"{cfg['session_label']}_{cfg['track']}_bout{rec['bout_index']:02d}_t{rec['thermal_t']:.1f}s_{rec['posture']}_dorsal_{valid_tag}.png"
        title = (f"{name}  bout {rec['bout_index']}  t={rec['thermal_t']:.1f}s thermal  "
                 f"posture={rec['posture']}  tail_qc_valid={rec['qc_valid']}")
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
    print(f"{name}: {len(extended_pool)} extended + {len(curled_pool)} curled + "
          f"{len(ambiguous_pool)} ambiguous with dorsal, {len(chosen)} examples rendered", flush=True)


if __name__ == "__main__":
    for name, cfg in SESSIONS.items():
        candidates = compute_candidates(name, cfg)
        render_session(name, cfg, candidates)
    print("\n=== DONE ===")
    print(f"output dir: {OUT_DIR}")
