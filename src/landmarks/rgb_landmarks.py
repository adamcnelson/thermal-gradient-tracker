"""
Stage 5 — RGB landmark extraction, classical-CV primary method
(project_brief_v7.md §6 Stage 5).

A black mouse on a white plate is near-solved with a temporal background
model + threshold, per the brief. This module is deliberately decoupled
from src/mouse_segmentation.py (the thermal-side equivalent) rather than
sharing code with it — project_brief_v7.md §3's core principle is that RGB
geometry and thermal radiometry never mix roles, and that separation is
easiest to keep true if the two segmentation implementations don't share a
code path either, even though the underlying algorithm (temporal median
background + adaptive threshold + morphology) is conceptually the same.

Pipeline: segment_mouse_rgb() -> binary mask -> skeletonize -> order into a
nose<->tail path -> width profile via distance transform -> tail-base
landmark (width threshold crossing) -> tail centerline (path from tail
base to tail tip).

The supervised-pose fallback (SLEAP/DeepLabCut/Lightning Pose) named in the
brief for frames where classical CV fails QC is not implemented here —
it's GPU/ARCC-trained per the brief and out of scope until the classical
method's yield is measured against the (not-yet-labeled) validation set.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

Point = Tuple[int, int]  # (row, col) = (y, x), matching numpy array indexing


# ── segmentation (temporal background model, decoupled from the thermal side) ──


class RgbBackgroundModel:
    """Temporal median background over a sample of track-matched-crop RGB frames."""

    def __init__(self, background: np.ndarray):
        self._bg = background.astype(np.float32)

    @property
    def background(self) -> np.ndarray:
        return self._bg

    @classmethod
    def build(cls, frames: Sequence[np.ndarray]) -> "RgbBackgroundModel":
        if len(frames) == 0:
            raise ValueError("Need at least 1 frame to build a background model")
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        return cls(np.median(stack, axis=0))

    def foreground_score(self, frame: np.ndarray) -> np.ndarray:
        return np.abs(frame.astype(np.float32) - self._bg)


def segment_mouse_rgb(
    frame: np.ndarray,
    background_model: RgbBackgroundModel,
    min_area: int,
    max_area: int,
    threshold_sigma: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Segment the mouse from one grayscale RGB (track-matched-crop) frame.
    Returns a bool mask of the single largest qualifying blob, or None.
    """
    fg_score = background_model.foreground_score(frame)
    threshold = float(np.mean(fg_score) + threshold_sigma * np.std(fg_score))
    raw_mask = (fg_score > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    labeled, n_labels = ndimage.label(cleaned)
    if n_labels == 0:
        return None

    best_mask, best_area = None, 0
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        area = int(np.sum(component))
        if min_area <= area <= max_area and area > best_area:
            best_mask, best_area = component, area

    return best_mask


def is_plausible_mouse_blob(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_aspect_ratio: float = 1.8,
) -> bool:
    """
    False-positive filtering (brief: "mandatory, not optional" — fecal boli
    are small, roughly round/compact blobs). Rejects by area and by
    bounding-box aspect ratio (an extended mouse is elongated; a bolus is
    close to isotropic). Track-continuity filtering is a temporal check
    that belongs at the calling/orchestration layer, not here.

    NOTE (2026-08-17): this aspect-ratio-only check also rejects every
    curled/resting posture, not just debris — see classify_mouse_blob()
    below, which was added after a real-data yield diagnostic found this
    was the dominant cause of low segmentation yield (curled mice have
    essentially the same area as extended ones, just a low aspect ratio).
    Kept as-is for callers that specifically want "is this elongated,"
    e.g. tail/skeleton extraction, which does require an extended posture.
    """
    area = int(np.sum(mask))
    if not (min_area <= area <= max_area):
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_idx, col_idx = np.where(rows)[0], np.where(cols)[0]
    height = row_idx[-1] - row_idx[0] + 1
    width = col_idx[-1] - col_idx[0] + 1
    aspect = max(height, width) / max(min(height, width), 1)
    return aspect >= min_aspect_ratio


def classify_mouse_blob(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_aspect_ratio: float = 1.8,
    curled_min_area_frac: float = 5.0,
) -> Optional[str]:
    """
    Classify a segmented blob as "extended" or "curled" posture, or reject
    it as implausible (a fecal bolus) — returns None in that case.

    A real-data yield diagnostic (2026-08-17, both real sessions) found
    that curled/resting mice have essentially the SAME area as extended
    ones (~9,600-9,800px median in both sessions) — same animal, same
    mass, just compact — while is_plausible_mouse_blob()'s aspect-ratio
    filter alone can't tell that apart from a small round fecal bolus, so
    it silently rejected every curled frame too. That turned out to be the
    dominant segmentation failure mode in practice (confirmed by eye
    against saved examples): 62% of sampled Test_3 frames, 16% of Test_4.

    Fix: use area AND aspect jointly instead of aspect alone. An elongated
    blob (aspect >= min_aspect_ratio) is "extended" regardless of size
    (within the area bounds already enforced). A compact blob is "curled"
    only if its area is well above min_area (curled_min_area_frac x,
    default 5x) — comfortably separating a real curled mouse from a
    bolus-sized object near the area floor. Anything else is rejected.

    The returned posture label is meant to flow into
    thermal_measurement.gate_measurement()'s posture_ok parameter — curled
    frames are real, valid detections that should be gated at the
    measurement-validity stage (brief §6: "postural state is not curled or
    rearing"), not silently discarded at segmentation.
    """
    area = int(np.sum(mask))
    if not (min_area <= area <= max_area):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_idx, col_idx = np.where(rows)[0], np.where(cols)[0]
    height = row_idx[-1] - row_idx[0] + 1
    width = col_idx[-1] - col_idx[0] + 1
    aspect = max(height, width) / max(min(height, width), 1)
    if aspect >= min_aspect_ratio:
        return "extended"
    if area >= curled_min_area_frac * min_area:
        return "curled"
    return None


# ── skeleton path ordering ──────────────────────────────────────────────────


def prune_short_skeleton_branches(skeleton: np.ndarray, max_prune_px: int) -> np.ndarray:
    """
    Remove short spurious branches (endpoint -> nearest branch-point runs
    of length <= max_prune_px) from a binary skeleton, returning a cleaned
    copy. A no-op on an already-simple (branchless) skeleton, so calling
    this unconditionally never changes behavior for frames that were
    already valid.

    Real-data motivation (2026-08-26, Adam: "the extended posture
    definition feels too conservative"): a diagnostic on real Test_3/4/7
    frames found the aspect-ratio gate in classify_mouse_blob() is NOT the
    yield bottleneck — plenty of frames pass it (e.g. Test_7: 23/62). The
    actual bottleneck is order_skeleton_path()'s exact-2-endpoints check:
    91% of Test_7's aspect-qualified frames fail it (100% for Test_3).
    Measuring real branch lengths on those failures showed two distinct
    populations: most branches are substantial (20-230px — genuine body
    curvature from a hunched/curled posture, where forcing the frame
    through as "extended" would feed a badly-shaped body into
    anterior_region_mask(), which already has a documented failure mode
    for exactly this case) and a handful are clearly noise-scale (1-10px —
    a stray pixel or tiny mask irregularity, not real anatomy). This
    function only removes the second population. Chosen deliberately
    NOT to just loosen the aspect-ratio threshold or the endpoint check
    wholesale, since that would rescue yield by accepting genuinely
    hunched bodies too, trading measurement correctness for sample count.

    Iterative: after removing one layer of short branches, some
    branch-points can become degree-2 (no longer a junction) or new short
    stubs can be exposed, so this repeats until a pass removes nothing.
    """
    skel = skeleton.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    while True:
        skel_u8 = skel.astype(np.uint8)
        neighbor_count = cv2.filter2D(skel_u8, -1, kernel) - skel_u8
        endpoints = [tuple(p) for p in np.argwhere(skel & (neighbor_count == 1))]
        branchpoints = {tuple(p) for p in np.argwhere(skel & (neighbor_count >= 3))}
        if len(endpoints) <= 2:
            break  # already simple (or a closed loop -- pruning can't fix that)
        pixels = {tuple(p) for p in np.argwhere(skel)}
        to_remove = set()
        for ep in endpoints:
            run = [ep]
            visited = {ep}
            current = ep
            hit_branchpoint = False
            while True:
                y, x = current
                nbrs = [
                    (y + dy, x + dx)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy, dx) != (0, 0)
                    and (y + dy, x + dx) in pixels
                    and (y + dy, x + dx) not in visited
                ]
                if not nbrs:
                    break
                nxt = nbrs[0]
                visited.add(nxt)
                if nxt in branchpoints:
                    hit_branchpoint = True
                    break
                run.append(nxt)
                current = nxt
            if hit_branchpoint and len(run) <= max_prune_px:
                to_remove.update(run)
        if not to_remove:
            break
        for p in to_remove:
            skel[p] = False
    return skel


def order_skeleton_path(skeleton: np.ndarray, max_prune_px: int = 0) -> List[Point]:
    """
    Walk a single-pixel-wide skeleton from one endpoint to the other and
    return the ordered list of (row, col) pixels.

    max_prune_px : passed straight to prune_short_skeleton_branches()
        before the endpoint check (default 0 -- a genuine no-op, existing
        callers see no behavior change unless they opt in). See that
        function's docstring for why this exists and what it deliberately
        does NOT try to rescue.

    Raises ValueError if the (possibly pruned) skeleton isn't a simple
    open path (0 endpoints -> a closed loop; >2 endpoints -> branched,
    e.g. a curled/self-occluded animal) — both are legitimate QC-reject
    cases for the caller, not bugs.
    """
    if max_prune_px > 0:
        skeleton = prune_short_skeleton_branches(skeleton, max_prune_px)

    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, np.ones((3, 3), dtype=np.uint8)) - skel_u8
    endpoints = [tuple(p) for p in np.argwhere(skeleton & (neighbor_count == 1))]

    if len(endpoints) != 2:
        raise ValueError(
            f"Expected a simple open skeleton path (2 endpoints), found {len(endpoints)} "
            "— likely a branched or closed skeleton (curled posture, self-occlusion, or "
            "segmentation artifact)."
        )

    pixels = {tuple(p) for p in np.argwhere(skeleton)}
    start = endpoints[0]
    visited = {start}
    path = [start]
    current = start
    while True:
        y, x = current
        candidates = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
            and (y + dy, x + dx) in pixels
            and (y + dy, x + dx) not in visited
        ]
        if not candidates:
            break
        nxt = candidates[0]
        visited.add(nxt)
        path.append(nxt)
        current = nxt

    if len(path) != len(pixels):
        raise ValueError(
            f"Skeleton walk covered {len(path)}/{len(pixels)} pixels — the skeleton has a "
            "branch the endpoint-count check didn't catch (e.g. a short spur)."
        )
    return path


def width_profile(mask: np.ndarray, path: Sequence[Point]) -> np.ndarray:
    """Local width (px) at each path point, via 2x the distance-to-background transform."""
    dist = ndimage.distance_transform_edt(mask)
    return np.array([2.0 * dist[y, x] for (y, x) in path])


# ── nose/tail orientation + tail-base landmark ──────────────────────────────


@dataclass
class MouseSkeletonLandmarks:
    path_nose_to_tail: List[Point]
    widths_nose_to_tail: np.ndarray
    nose_point: Point
    tail_tip_point: Point
    tail_base_point: Point
    tail_base_index: int  # index into path_nose_to_tail / widths_nose_to_tail
    tail_centerline: List[Point]  # path_nose_to_tail[tail_base_index:]


def find_tail_base(
    path: Sequence[Point], widths: np.ndarray, trunk_width_frac: float = 0.35
) -> MouseSkeletonLandmarks:
    """
    Orient the path nose-first and locate the tail-base landmark (brief §6
    Stage 5): the point where width drops below trunk_width_frac * trunk
    width and stays there. Orientation is determined by which end has the
    longer sustained run of low width — the tail is thin along its whole
    length, whereas the nose end can dip briefly right at the tip but does
    not sustain it.
    """
    if len(path) < 3:
        raise ValueError("Path too short to determine nose/tail orientation")

    trunk_width = float(np.max(widths))
    threshold = trunk_width_frac * trunk_width
    below = widths < threshold

    def leading_run(arr: np.ndarray) -> int:
        n = 0
        for v in arr:
            if v:
                n += 1
            else:
                break
        return n

    lead = leading_run(below)
    trail = leading_run(below[::-1])

    if lead == trail:
        raise ValueError(
            "Cannot determine nose/tail orientation — both ends have equally sustained "
            "low width; not a plausible mouse silhouette."
        )

    if lead > trail:
        # path[0] end is the tail -> reverse so we return nose-first
        path_oriented = list(reversed(path))
        widths_oriented = widths[::-1]
        tail_base_index = len(path) - lead
    else:
        path_oriented = list(path)
        widths_oriented = widths
        tail_base_index = len(path) - trail

    return MouseSkeletonLandmarks(
        path_nose_to_tail=path_oriented,
        widths_nose_to_tail=widths_oriented,
        nose_point=path_oriented[0],
        tail_tip_point=path_oriented[-1],
        tail_base_point=path_oriented[tail_base_index],
        tail_base_index=tail_base_index,
        tail_centerline=path_oriented[tail_base_index:],
    )


# ── tail-only extraction (no nose/whole-body decomposition needed) ──────────


@dataclass
class TailOnlyLandmarks:
    tail_base_point: Point
    tail_tip_point: Point
    tail_centerline: List[Point]  # base -> tip order


def _walk_leaf_segment(pixels: set, branchpoints: set, endpoint: Point) -> List[Point]:
    """Walk a skeleton from `endpoint` until hitting a branch point (excluded
    from the returned path -- it belongs to the body mass, not the
    appendage) or running out of unvisited neighbors. Ordered tip -> base."""
    path = [endpoint]
    visited = {endpoint}
    current = endpoint
    while True:
        y, x = current
        nbrs = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
            and (y + dy, x + dx) in pixels
            and (y + dy, x + dx) not in visited
        ]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt in branchpoints:
            break
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path


def find_tail_appendage(
    mask: np.ndarray,
    max_tail_width_px: float = 20.0,
    min_tail_length_px: int = 15,
    min_thin_fraction: float = 0.8,
    max_prune_px: int = 0,
) -> Optional[TailOnlyLandmarks]:
    """
    Find just the tail (base + centerline) without requiring a full, clean
    nose-to-tail skeleton path — usable on hunched/curled-with-tail-out
    bodies where extract_landmarks_from_mask() raises (branched skeleton)
    even though the tail itself is a real, clearly resolvable thin
    appendage.

    Real motivation (2026-08-26, Adam: "tail-ΔT is one of the primary
    reasons for this whole endeavor, find a solution"): a diagnostic
    found most of the "ambiguous" (aspect-qualified but skeleton-
    rejected) frames in Test_3/Test_7 have EXACTLY this shape — a
    compact, hunched body mass with a clearly visible thin tail sticking
    out. tail_base_delta_t() (thermal_measurement.py) only ever consumes
    `proximal_tail_points_rgb` — it has no dependency on a nose point or
    a whole-body straight-line decomposition at all. Forcing the WHOLE
    body through extract_landmarks_from_mask() was never actually
    necessary for tail-ΔT specifically — that's a real, separate
    requirement from warm-spot/dorsal, which DO need the whole-body
    partition (anterior_region_mask() assumes a genuinely straight body;
    see [[project-v7-stage5-bakeoff]] Finding #2 for why forcing a
    hunched body through THAT pipeline is actively wrong, not just
    unnecessary — this function deliberately keeps that pipeline
    untouched and does not claim "extended" posture for these frames).

    max_tail_width_px is an ABSOLUTE pixel threshold, not trunk-relative
    (an earlier version used trunk_width_frac * this-frame's own max
    body width — reverted after a real failure case: one real Test_7
    frame had a genuinely large trunk width (74px, well within the real,
    confirmed range — see below), which inflated the relative threshold
    enough to admit a wide body-fold as "thin", producing a wrong tail
    pick). The default 20.0 comes directly from real, confirmed data:
    measured actual tail-segment widths from 3,816 already-successful
    "extended" detections across all 3 sessions gave a median of 8px and
    a 95th percentile of 17px, while real trunk (max body) widths ranged
    50-97px (median 78px) — an absolute cutoff around 20px sits with
    wide margin above real tails and equally wide margin below real
    trunks, and is robust to any single frame's own trunk-width estimate
    being unusually large or small.

    Candidate tail = the leaf skeleton segment (an endpoint walked inward
    to the nearest branch point, or to another endpoint if the skeleton
    is unbranched), TRUNCATED at the point it first becomes body-thick
    (width >= max_tail_width_px, walking tip -> base -- see
    tail_base_point below for why the raw walk-to-branch-point isn't
    anatomically meaningful), then gated on that truncated run: (a) at
    least min_tail_length_px long — real tail scale, not a stub,
    whisker, or single noisy pixel — and (b) thin along at least
    min_thin_fraction of that truncated length (in practice this is
    ~1.0 by construction post-truncation; kept as a defensive check, not
    the primary discriminator it was before truncation moved earlier in
    the pipeline — see the real bug this fixed, below). Among qualifying
    candidates, the LONGEST (truncated) one wins — a mouse has exactly
    one tail, and length is what separates it from shorter thin
    protrusions (a real failure mode seen without this: a short ear/
    head-bump edge case winning simply because the true tail didn't
    pass the thin check).

    Real bug found 2026-08-27 (Adam: a tail "clearly visible to the
    naked eye" produced no measurement at all): gating on thin_fraction
    BEFORE truncating (the original order) let the same deep-branch-
    point problem described below corrupt candidate SELECTION, not just
    the returned base point. Confirmed on a real Test_4 frame: a
    211-point leaf stayed 4-18px wide (real tail) for its first 142
    points, then jumped to 36-81px (body) over its last 69 before
    reaching the branch point -- raw thin_fraction over the whole leaf
    was 0.67, below the 0.8 gate, even though the true tail run was
    100% thin. Moving truncation before the gates fixes this without
    loosening min_thin_fraction itself.

    Returns None if no segment qualifies — e.g. a genuinely tucked-tail
    curl, where the tail isn't separably resolvable in the mask at all.
    That's a real, correct absence, not a bug: this function does not
    fabricate a tail that isn't visibly there.

    tail_base_point is NOT the winning leaf's skeleton branch point.
    Real bug found 2026-08-26 (Adam: "the tail is cleanly delineated in
    RGB, but the [sampled point] is way off... in the middle of the back
    of the mouse"): for a curled/rounded body, the skeleton branch point
    where a thin tail leaf joins the rest of the skeleton graph can sit
    deep INSIDE the body mass, well past the true tail-body surface
    transition — the body's own rounded shape produces extra internal
    skeleton branches, so "the first branch point encountered walking in
    from the tip" is not anatomically meaningful. Confirmed on a real
    Test_3 frame: a 146-point leaf stayed 7-12px wide (genuine tail) for
    its first ~120 points, then jumped to 32px, then 55px, then 66px
    (clearly body) over its last ~25 points before reaching the actual
    branch point. proximal_tail_points() (thermal_measurement.py) takes
    the first 20% of the centerline by arc length FROM THE BASE, so an
    untruncated base sampled body tissue, not tail. Fixed by truncating
    every candidate leaf at the point its OWN width first reaches
    max_tail_width_px (walking tip -> base) BEFORE gating/selection,
    instead of returning the winning leaf all the way to the graph
    branch point — reuses the same real, data-grounded threshold as the
    thin-fraction candidate check above, no new tuning.

    max_prune_px: real bug found 2026-08-27 (Adam, reviewing a "cool"-zone
    Test_7 bout stuck at 0% yield: a visibly-tailed frame still returned
    no tail). skeletonize() can fray right at a thin tail's own tip into
    two spurious short branches (a pixel-level artifact, not real
    anatomy) -- confirmed on the real frame: two endpoints at raw leaf
    lengths 9px and 2px sat exactly where the visible tail tip was,
    both below min_tail_length_px, while the genuine tail continued
    past the branch point they created. This is the SAME noise-scale
    problem prune_short_skeleton_branches() was built for (see its
    docstring) -- order_skeleton_path() already uses it for the
    whole-body path; find_tail_appendage() never did. Opt-in (default 0
    = old behavior unchanged) per this project's convention.
    """
    skeleton = skeletonize(mask)
    if max_prune_px > 0:
        skeleton = prune_short_skeleton_branches(skeleton, max_prune_px)
    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, np.ones((3, 3), dtype=np.uint8)) - skel_u8
    endpoints = [tuple(p) for p in np.argwhere(skeleton & (neighbor_count == 1))]
    if not endpoints:
        return None
    branchpoints = {tuple(p) for p in np.argwhere(skeleton & (neighbor_count >= 3))}
    pixels = {tuple(p) for p in np.argwhere(skeleton)}

    dist = ndimage.distance_transform_edt(mask)
    threshold = max_tail_width_px

    candidates = []
    for ep in endpoints:
        leaf = _walk_leaf_segment(pixels, branchpoints, ep)
        widths = np.array([2.0 * dist[y, x] for (y, x) in leaf])
        # Truncate tip -> base at the point the leaf first becomes
        # body-thick, BEFORE the length/thin-fraction gates below --
        # real bug found 2026-08-27 (Adam: a tail "clearly visible to the
        # naked eye" was returning no measurement at all). The graph
        # branch point (where an untruncated `leaf` ends) can sit deep
        # inside a curled/rounded body (see tail_base_point below), so a
        # long real tail's raw, untruncated thin_fraction gets dragged
        # down by irrelevant deep-body pixels between the true tail-body
        # transition and the branch point. Confirmed on a real Test_4
        # frame: a 211-point leaf stayed 4-18px wide (real tail) for its
        # first 142 points, then jumped to 36-81px (body) over the last
        # 69 -- raw thin_fraction 0.67, below the 0.8 gate, even though
        # the true tail run was 100% thin. Gating on the truncated run
        # instead fixes this without loosening the gate itself.
        cutoff = len(leaf)
        for i, w in enumerate(widths):
            if w >= threshold:
                cutoff = i
                break
        leaf = leaf[:cutoff]
        widths = widths[:cutoff]
        if len(leaf) < min_tail_length_px:
            continue
        thin_fraction = float(np.mean(widths < threshold)) if len(widths) else 0.0
        if thin_fraction < min_thin_fraction:
            continue
        candidates.append(leaf)
    if not candidates:
        return None

    best = max(candidates, key=len)  # already truncated + length-gated above (>= min_tail_length_px)
    centerline = list(reversed(best))  # walked tip -> base; return base -> tip
    return TailOnlyLandmarks(
        tail_base_point=centerline[0],
        tail_tip_point=centerline[-1],
        tail_centerline=centerline,
    )


# ── orchestration ────────────────────────────────────────────────────────────


def extract_landmarks_from_mask(
    mask: np.ndarray, trunk_width_frac: float = 0.35, max_prune_px: int = 0
) -> MouseSkeletonLandmarks:
    """
    Full classical-CV pipeline from a binary mouse mask to landmarks.
    Requires an open, unbranched nose-to-tail skeleton (raises ValueError
    otherwise, e.g. a curled/self-occluded posture) — see
    extract_mouse_detection() below for the posture-aware entry point that
    doesn't attempt this on masks already known to be curled.

    max_prune_px : forwarded to order_skeleton_path() -- see
        prune_short_skeleton_branches() for why this exists (default 0,
        no behavior change).
    """
    skeleton = skeletonize(mask)
    path = order_skeleton_path(skeleton, max_prune_px=max_prune_px)
    widths = width_profile(mask, path)
    return find_tail_base(path, widths, trunk_width_frac=trunk_width_frac)


@dataclass
class MouseDetectionResult:
    posture: str  # "extended" | "curled" | "ambiguous"
    mask: np.ndarray
    landmarks: Optional[MouseSkeletonLandmarks]  # only set when posture == "extended"
    tail_landmarks: Optional[TailOnlyLandmarks] = None  # see extract_mouse_detection()'s docstring


def extract_mouse_detection(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_aspect_ratio: float = 1.8,
    curled_min_area_frac: float = 5.0,
    trunk_width_frac: float = 0.35,
    max_prune_px: int = 0,
) -> Optional[MouseDetectionResult]:
    """
    Posture-aware entry point combining classify_mouse_blob() with landmark
    extraction — the intended real-batch-use replacement for the older
    "segment -> is_plausible_mouse_blob -> extract_landmarks_from_mask,
    catch ValueError" pattern.

    A real-data yield diagnostic (2026-08-17, see
    [[project-v7-rgb-yield-fix]] in memory) found that curled/resting mice
    — likely the majority state during a stationary bout, which is exactly
    when Stage 6 measurement happens — were being silently dropped
    entirely: is_plausible_mouse_blob()'s aspect-ratio filter rejected
    them outright, and even after that was fixed, a curled body has no
    single open nose-to-tail curve, so skeleton extraction would only
    raise ValueError anyway. Rather than attempt (and catch the failure
    of) an operation known in advance to be inapplicable, classify the
    posture FIRST and only run skeleton extraction for "extended" blobs.

    Returns None if the blob is rejected outright (debris-scale, brief's
    "reject by area/aspect" false-positive filtering). Returns posture
    "curled" with landmarks=None for a real but compact/resting mouse —
    callers can still compute a whole-mask dorsal-surface measurement
    (brief: not applicable, but Adam's added whole-animal mean/median is)
    from `result.mask` directly, since there's no tail to exclude from a
    curled mask the way dorsal_surface_mask() excludes it from an
    extended one. Warm-spot and tail-ΔT remain inapplicable for curled
    frames — they need the nose/tail landmarks this posture doesn't have.
    Returns posture "ambiguous" with landmarks=None for the rare case
    where the blob passes the elongation check but the skeleton still
    isn't a simple open path (e.g. a genuinely branched/noisy mask) —
    kept distinct from "curled" since it's a real classification failure,
    not an expected physiological state.

    max_prune_px : forwarded to extract_landmarks_from_mask() -- tolerates
        short spurious skeleton branches (segmentation noise, e.g. a
        stray pixel or tiny mask irregularity) before the simple-path
        check, without accepting genuinely hunched/curled bodies whose
        branches are long (real anatomy, not noise). See
        prune_short_skeleton_branches()'s docstring for the real-data
        analysis behind this (2026-08-26). Default 0 -- no behavior
        change unless a caller opts in.

    tail_landmarks (2026-08-26, see find_tail_appendage()'s docstring for
    the full rationale): for "curled" and "ambiguous" postures -- where
    `landmarks` is None because there's no valid whole-body decomposition
    -- this field is populated whenever find_tail_appendage() can still
    resolve a real, separately-visible tail from the mask alone (common
    for a hunched/resting body with its tail out, which is a large
    fraction of real "ambiguous" frames). Callers that only need tail-ΔT
    (not warm-spot/dorsal, which genuinely require the "extended" whole-
    body case) should check this field for postures other than
    "extended" rather than treating them as a total loss. None here means
    no separately-resolvable tail was found (e.g. a genuinely tucked
    curl) -- a real absence, not a failure to look.
    """
    posture = classify_mouse_blob(mask, min_area, max_area, min_aspect_ratio, curled_min_area_frac)
    if posture is None:
        return None
    if posture == "curled":
        tail_landmarks = find_tail_appendage(mask, max_prune_px=max_prune_px)
        return MouseDetectionResult(posture="curled", mask=mask, landmarks=None, tail_landmarks=tail_landmarks)
    try:
        landmarks = extract_landmarks_from_mask(mask, trunk_width_frac=trunk_width_frac, max_prune_px=max_prune_px)
    except ValueError:
        tail_landmarks = find_tail_appendage(mask, max_prune_px=max_prune_px)
        return MouseDetectionResult(posture="ambiguous", mask=mask, landmarks=None, tail_landmarks=tail_landmarks)
    return MouseDetectionResult(posture="extended", mask=mask, landmarks=landmarks)
