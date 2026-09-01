"""
Orchestrator: computes per-sample candidates ONCE per session (expensive: full
segmentation + homography warp + thermal sampling pass) and feeds the result to
all three location-QC render scripts (tail, warm-spot, dorsal), instead of each
running its own independent (and 3x redundant) full pass. Run this instead of
the three render_*_qc.py scripts individually when regenerating all of them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qc_shared import SESSIONS, compute_candidates
import render_tail_qc
import render_warmspot_qc
import render_dorsal_qc

if __name__ == "__main__":
    for name, cfg in SESSIONS.items():
        candidates = compute_candidates(name, cfg)
        render_tail_qc.render_session(name, cfg, candidates)
        render_warmspot_qc.render_session(name, cfg, candidates)
        render_dorsal_qc.render_session(name, cfg, candidates)

    print("\n=== ALL DONE ===")
    print(f"tail QC:     {render_tail_qc.OUT_DIR}")
    print(f"warm-spot QC:{render_warmspot_qc.OUT_DIR}")
    print(f"dorsal QC:   {render_dorsal_qc.OUT_DIR}")
