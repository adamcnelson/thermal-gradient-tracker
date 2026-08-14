"""
Stage 8 — validation set design (project_brief_v7.md §8).

Selects a stratified sample of frames for hand-labeling (75 frames, brief's
target), honoring — where the data actually supports it — the brief's
stratification axes: gradient position (over-sample hot/mid), session
spread (~n_total/n_sessions_target per session, not concentrated), and
nestlet (both nestlet-present and no-nestlet SESSIONS represented, with
on/adjacent-to-nestlet frames prioritized within nestlet sessions when
identifiable).

Two of the brief's axes are NOT automatable yet and this module says so
rather than faking it:
- Posture (curled/rearing/grooming/extended) has no classifier built
  anywhere in this codebase — pass posture_col only once one exists.
- "On or adjacent to nestlet" needs the Stage 2b nestlet detector, which
  was never built (only referenced as an optional mask parameter in
  src/landmarks/thermal_measurement.py). on_nestlet_col is accepted as an
  optional column for when that data exists (e.g. from manual review), not
  computed here.

This function selects WHICH (session, track, frame) to label — it does not
label anything. See src/landmarks/labeling_tool.py-equivalent script
(scripts/label_validation_frame.py) for the actual labeling interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_ZONE_FRACTIONS = {"hot": 0.4, "mid": 0.4, "cool": 0.2}


@dataclass
class ValidationFrameSelection:
    rows: pd.DataFrame
    warnings: List[str] = field(default_factory=list)


def _sample_session_frames(
    pool: pd.DataFrame,
    quota: int,
    zone_col: str,
    zone_fractions: Dict[str, float],
    on_nestlet_col: Optional[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fill one session's frame quota: on-nestlet frames first (if identifiable), then
    zone-weighted sampling of the rest, then fill any shortfall from whatever remains."""
    if quota <= 0 or pool.empty:
        return pool.iloc[0:0]

    picks = []
    remaining_quota = quota

    if on_nestlet_col and on_nestlet_col in pool.columns:
        on_nestlet_pool = pool[pool[on_nestlet_col] == True]  # noqa: E712
        n_on_nestlet = min(len(on_nestlet_pool), max(1, quota // 3)) if not on_nestlet_pool.empty else 0
        if n_on_nestlet:
            chosen = on_nestlet_pool.sample(n=n_on_nestlet, random_state=int(rng.integers(0, 2**32 - 1)))
            picks.append(chosen)
            remaining_quota -= len(chosen)
            pool = pool.drop(chosen.index)

    if remaining_quota <= 0:
        return pd.concat(picks, ignore_index=True)

    zone_targets = {z: round(remaining_quota * f) for z, f in zone_fractions.items()}
    diff = remaining_quota - sum(zone_targets.values())
    if diff != 0:
        biggest = max(zone_fractions, key=zone_fractions.get)
        zone_targets[biggest] += diff

    for zone, target in zone_targets.items():
        if target <= 0 or zone_col not in pool.columns:
            continue
        zone_pool = pool[pool[zone_col] == zone]
        n = min(len(zone_pool), target)
        if n:
            chosen = zone_pool.sample(n=n, random_state=int(rng.integers(0, 2**32 - 1)))
            picks.append(chosen)
            pool = pool.drop(chosen.index)

    selected = pd.concat(picks, ignore_index=True) if picks else pool.iloc[0:0]

    shortfall = quota - len(selected)
    if shortfall > 0 and not pool.empty:
        n = min(len(pool), shortfall)
        chosen = pool.sample(n=n, random_state=int(rng.integers(0, 2**32 - 1)))
        selected = pd.concat([selected, chosen], ignore_index=True)

    return selected


def select_validation_frames(
    candidates: pd.DataFrame,
    n_total: int = 75,
    n_sessions_target: int = 25,
    zone_fractions: Optional[Dict[str, float]] = None,
    session_col: str = "session",
    track_col: str = "track",
    zone_col: str = "gradient_zone",
    nestlet_present_col: str = "nestlet_present",
    on_nestlet_col: Optional[str] = None,
    posture_col: Optional[str] = None,
    random_seed: int = 123,
) -> ValidationFrameSelection:
    """Stratified frame selection for hand-labeling — see module docstring for scope/limits."""
    zone_fractions = zone_fractions or DEFAULT_ZONE_FRACTIONS
    rng = np.random.default_rng(random_seed)
    warnings: List[str] = []

    required = [session_col, track_col, zone_col, nestlet_present_col]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidates missing required columns: {missing}")

    if on_nestlet_col is None:
        warnings.append(
            "No on_nestlet_col provided — cannot prioritize on/adjacent-to-nestlet frames "
            "(brief §8); Stage 2b's nestlet detector was never built."
        )
    elif on_nestlet_col not in candidates.columns:
        raise ValueError(f"on_nestlet_col {on_nestlet_col!r} not in candidates")

    if posture_col is None:
        warnings.append(
            "No posture_col provided — cannot stratify by posture (brief §8); no automated "
            "posture classifier exists in this codebase yet."
        )
    elif posture_col not in candidates.columns:
        raise ValueError(f"posture_col {posture_col!r} not in candidates")

    sessions_df = candidates[[session_col, nestlet_present_col]].drop_duplicates()
    nestlet_sessions = sessions_df[sessions_df[nestlet_present_col] == True][session_col].tolist()  # noqa: E712
    no_nestlet_sessions = sessions_df[sessions_df[nestlet_present_col] == False][session_col].tolist()  # noqa: E712
    n_avail = len(nestlet_sessions) + len(no_nestlet_sessions)
    if n_avail == 0:
        raise ValueError("No candidate sessions available")

    n_pick = min(n_sessions_target, n_avail)
    if n_pick < n_sessions_target:
        warnings.append(f"Only {n_avail} session(s) available in candidate pool; target was {n_sessions_target}")

    frac_nestlet = len(nestlet_sessions) / n_avail
    n_nestlet_pick = min(len(nestlet_sessions), max(1, round(n_pick * frac_nestlet))) if nestlet_sessions else 0
    n_no_nestlet_pick = min(len(no_nestlet_sessions), n_pick - n_nestlet_pick)
    n_nestlet_pick = min(len(nestlet_sessions), n_pick - n_no_nestlet_pick)
    if nestlet_sessions and no_nestlet_sessions and (n_nestlet_pick == 0 or n_no_nestlet_pick == 0):
        warnings.append("Could not include both nestlet and no-nestlet sessions given n_sessions_target")

    picked_sessions: List = []
    if n_nestlet_pick:
        picked_sessions += list(rng.choice(nestlet_sessions, size=n_nestlet_pick, replace=False))
    if n_no_nestlet_pick:
        picked_sessions += list(rng.choice(no_nestlet_sessions, size=n_no_nestlet_pick, replace=False))

    base = n_total // len(picked_sessions)
    remainder = n_total % len(picked_sessions)
    per_session_quota = {s: base for s in picked_sessions}
    if remainder:
        bonus = rng.choice(picked_sessions, size=remainder, replace=False)
        for s in bonus:
            per_session_quota[s] += 1

    selected_chunks = []
    for session in picked_sessions:
        pool = candidates[candidates[session_col] == session]
        chunk = _sample_session_frames(
            pool, per_session_quota[session], zone_col, zone_fractions, on_nestlet_col, rng
        )
        selected_chunks.append(chunk)

    result_df = (
        pd.concat(selected_chunks, ignore_index=True) if selected_chunks else candidates.iloc[0:0]
    )
    if len(result_df) < n_total:
        warnings.append(
            f"Only {len(result_df)} frame(s) selected — candidate pool exhausted for some "
            f"session/zone combination(s); target was {n_total}"
        )

    if track_col in result_df.columns and not result_df.empty:
        track_counts = result_df[track_col].value_counts(normalize=True)
        underrepresented = track_counts[track_counts < 0.2]
        if not underrepresented.empty:
            warnings.append(f"Track balance skewed: {track_counts.to_dict()}")

    return ValidationFrameSelection(rows=result_df, warnings=warnings)
