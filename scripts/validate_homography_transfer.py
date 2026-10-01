"""
Quick check (project_v8, calibration scoping, 2026-09-29): does a session's
already-fitted homography transfer to a DIFFERENT same-regime session?

Reuses calibrate_homography.py's own rgb_background_frame()/
thermal_background_frame() so the warped image lands in exactly the pixel
coordinate frame the source homography was fit in -- not a re-derivation.

Warps the TARGET session's own RGB background frame through the SOURCE
session's fitted H into thermal pixel space, then overlays that warped
image's bright-rail silhouette (the rail is reflective/high-contrast
against the dark background, so a simple threshold gives a usable
outline) on top of the TARGET session's own real thermal background
frame. If the transform transfers, the warped target-RGB rail outline
should land on the target's real thermal rail structure. (Fixed
2026-09-30: an earlier version warped the SOURCE's own RGB instead, which
only redraws the source's own already-known fit regardless of target --
the target's independent data never entered the check.)

Usage:
    python scripts/validate_homography_transfer.py \\
        --source-homography homography_calibration/<...>_Front_homography.json \\
        --target-seq <raw or cropped target .seq> \\
        --out <output png>
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrate_homography import rgb_background_frame, thermal_background_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-homography", required=True)
    parser.add_argument("--target-seq", required=True)
    parser.add_argument("--target-video", required=True)
    parser.add_argument("--lane", required=True, choices=["F", "B"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = json.load(open(args.source_homography))
    H = np.array(src["H"], dtype=np.float64)

    # NOTE (2026-09-30 fix): this must warp the TARGET's own RGB background
    # through the SOURCE's H -- warping the source's own RGB (the original
    # version of this script) reproduces the source's own already-known-good
    # calibration shape regardless of target, since the output canvas size
    # is identical (43x412) for every session. That tests nothing about
    # whether H generalizes; it just redraws the source's own fit. The only
    # way the target session's independent data enters the check at all is
    # via its own RGB background.
    print("Building target RGB background...", flush=True)
    tgt_rgb = rgb_background_frame(args.target_video, args.lane)
    print("Building target thermal background...", flush=True)
    tgt_thermal = thermal_background_frame(args.target_seq)

    th, tw = tgt_thermal.shape[:2]
    warped_rgb = cv2.warpPerspective(tgt_rgb, H, (tw, th))

    # Bright-rail silhouette from the warped target RGB (rail is reflective,
    # high contrast against the dark background in every session observed).
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2GRAY) if warped_rgb.ndim == 3 else warped_rgb
    mask = gray > np.percentile(gray[gray > 0], 60) if (gray > 0).any() else np.zeros_like(gray, bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(tgt_thermal, cmap="jet" if tgt_thermal.ndim == 2 else None)
    if tgt_thermal.ndim == 2:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for c in contours:
        if cv2.contourArea(c) > 50:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color="lime", linewidth=1.5)
    ax.set_title(f"Target thermal + (target RGB warped through source H) rail outline (lime)\n"
                 f"source H: {Path(args.source_homography).name}\ntarget: {Path(args.target_seq).name}",
                 fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
