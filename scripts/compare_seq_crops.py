"""
Compare two cropped .seq files frame by frame (pixel data, raw ADU), to tell whether a
re-crop that is not byte-identical differs only by interpolation rounding (harmless) or
by geometry -- different crop corners would show large differences and a spatial shift.

Reports, over the first --max-frames frames: fraction of pixels that differ, the
distribution of |difference| in raw ADU, and the best integer shift between the two files'
mean images. For scale it also reports a typical within-frame
spatial std of the image itself.

Measured 2026-10-09 on the recover_crop gate (cluster re-crop vs the June 2026 crop of
07-08-25_4541_B_4547_F Front): 18.8% of pixels differ, p99 2 ADU, max 5 ADU, mean 0.2 ADU, no
shift -- same corners, OpenCV interpolation rounding only. --gate encodes that tolerance.

Usage:
    python scripts/compare_seq_crops.py A.seq B.seq [--max-frames 300] [--gate]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seq_io import SeqReader


def read_frames(path, n):
    out = []
    with SeqReader(str(path)) as r:
        for i, raw in r.frames():
            out.append(raw.astype(np.int32))
            if len(out) >= n:
                break
    return np.stack(out)


def best_integer_shift(a, b, max_dx=5, max_dy=3):
    """Integer (dx, dy) minimising mean |a - shift(b)| over the overlap. (cv2.phaseCorrelate is
    unreliable on these thin 43x412 strips: +0.5 px for identical images.)"""
    h, w = a.shape
    def mad(dx, dy):
        ys, xs = slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx))
        yb, xb = slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx))
        return float(np.abs(a[ys, xs] - b[yb, xb]).mean())
    scores = {(dx, dy): mad(dx, dy) for dx in range(-max_dx, max_dx + 1) for dy in range(-max_dy, max_dy + 1)}
    (bx, by), best = min(scores.items(), key=lambda kv: kv[1])
    return bx, by, best, scores[(0, 0)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--gate", action="store_true",
                   help="Exit 1 unless the files match within rounding: max |diff| <= --max-abs-adu, "
                        "99th percentile <= --max-p99-adu, and zero spatial shift")
    p.add_argument("--max-abs-adu", type=float, default=10.0)
    p.add_argument("--max-p99-adu", type=float, default=3.0)
    args = p.parse_args()

    a, b = read_frames(args.a, args.max_frames), read_frames(args.b, args.max_frames)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH: {a.shape} vs {b.shape}")
        sys.exit(1)
    d = np.abs(a - b)
    print(f"frames compared: {n}, frame shape: {a.shape[1:]}")
    print(f"pixels differing: {100 * (d > 0).mean():.2f}%")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  |diff| percentile {q:>5}: {np.percentile(d, q):.1f} ADU")
    print(f"  mean |diff|: {d.mean():.3f} ADU   (typical within-frame spatial std: {a.std(axis=(1, 2)).mean():.1f} ADU)")
    dx, dy, best, zero = best_integer_shift(a.mean(0), b.mean(0))
    print(f"shift that realigns B onto A (x, y): ({dx:+d}, {dy:+d}) px  "
          f"(mean |diff| at best {best:.1f} vs at zero shift {zero:.1f} ADU)")
    print("verdict hint: rounding-only => |diff| <= ~2 ADU and shift ~0; different corners => large diffs and/or shift")
    if args.gate:
        ok = d.max() <= args.max_abs_adu and np.percentile(d, 99) <= args.max_p99_adu and (dx, dy) == (0, 0)
        print("GATE PASS: equivalent within rounding" if ok else "GATE FAIL: differences beyond rounding")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
