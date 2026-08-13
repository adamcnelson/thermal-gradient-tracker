"""Synthetic mouse-silhouette mask generator for Stage 5 landmark tests."""

import numpy as np


def make_synthetic_mouse_mask(
    h: int = 80,
    w: int = 220,
    nose_x: int = 15,
    tail_base_x: int = 150,
    tail_tip_x: int = 200,
    cy: int = 40,
    trunk_peak: float = 26.0,
    shoulder_width: float = 16.0,
    tail_width: float = 3.0,
    head_len: int = 25,
    taper_len: int = 6,
) -> np.ndarray:
    """
    Body: head taper-in -> wide trunk -> short *sharp* taper down to
    shoulder_width right at tail_base_x (brief: "a sharp drop at the tail
    base"), then a long thin tail. An earlier version of this fixture used
    a single smooth sin() taper across the whole body, which (due to a
    saturating min(t*1.3, 1.0) term) collapsed to tail_width well before
    the intended tail_base_x — a fixture bug, not an algorithm bug, caught
    by the tail-base test failing with a landmark ~35px off from where the
    mask was actually thin. This piecewise version keeps the trunk clearly
    wide (shoulder_width, still well above the width-threshold used for
    tail-base detection) right up until the intentional sharp transition.
    """
    mask = np.zeros((h, w), dtype=bool)
    trunk_end_x = tail_base_x - taper_len
    for x in range(nose_x, tail_base_x):
        if x < nose_x + head_len:
            t = (x - nose_x) / head_len
            width = 6 + (trunk_peak - 6) * np.sin(np.pi / 2 * t)
        elif x < trunk_end_x:
            t = (x - (nose_x + head_len)) / max(1, trunk_end_x - (nose_x + head_len))
            width = trunk_peak - (trunk_peak - shoulder_width) * t
        else:
            t = (x - trunk_end_x) / taper_len
            width = shoulder_width - (shoulder_width - tail_width) * t
        half = width / 2
        y0, y1 = int(cy - half), int(cy + half)
        mask[y0:y1, x] = True
    for x in range(tail_base_x, tail_tip_x):
        t = (x - tail_base_x) / (tail_tip_x - tail_base_x)
        half = (tail_width / 2) * (1 - 0.3 * t)
        y0, y1 = int(cy - half), int(cy + half + 1)
        mask[y0:y1, x] = True
    return mask
