"""Load, clean, and join the experimental metadata LUT to tracking files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ── LUT loading ────────────────────────────────────────────────────────────────

_VIRUS_MAP = {"gi": "Gi", "gq": "Gq", "none": "none", "": "none"}
_INJECTION_MAP = {"saline": "Saline", "vehicle": "Vehicle", "dcz": "DCZ"}
_CRANIOTOMY_MAP = {
    "pre-craniotomy": "habituation",
    "pre craniotomy": "habituation",
    "pre": "habituation",
    "post": "experimental",
    "post-craniotomy": "experimental",
    "postcraniotomy": "experimental",
}


def _normalize(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def load_lut(path: str) -> pd.DataFrame:
    """
    Load and clean the metadata LUT.

    - Reads with utf-8-sig to strip BOM.
    - Normalizes Virus, Injection, and Craniotomy columns.
    - Adds a `phase` column from Craniotomy.
    - Strips whitespace from all string columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Metadata LUT not found: {path}\n"
            "Place LUT_CLEAN_July6.csv in the metadata/ directory."
        )
    df = pd.read_csv(str(p), encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Strip all string columns
    for col in df.columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # Virus
    if "Virus" in df.columns:
        df["Virus"] = df["Virus"].apply(
            lambda v: _VIRUS_MAP.get(_normalize(v).lower(), _normalize(v) or "none")
        )

    # Injection
    if "Injection" in df.columns:
        df["Injection"] = df["Injection"].apply(
            lambda v: _INJECTION_MAP.get(_normalize(v).lower(), _normalize(v))
        )

    # Craniotomy → phase
    if "Craniotomy" in df.columns:
        df["phase"] = df["Craniotomy"].apply(
            lambda v: _CRANIOTOMY_MAP.get(_normalize(v).lower(), _normalize(v))
        )

    return df


def filter_excluded(lut_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop single-recording rows and rows without a Video_name_SEQ.

    Exclusion criteria (any one is sufficient):
      1. NOTES contains both 'single' and 'recording' (case-insensitive)
      2. Video_name_NOTES == 'Exclude'
      3. Video_name_SEQ is empty

    Returns (kept_df, dropped_df).
    """
    def _is_excluded(row):
        notes = _normalize(row.get("NOTES", "")).lower()
        vid_notes = _normalize(row.get("Video_name_NOTES", ""))
        vsn = _normalize(row.get("Video_name_SEQ", ""))
        if "single" in notes and "recording" in notes:
            return True, "single recording in NOTES"
        if vid_notes == "Exclude":
            return True, "Video_name_NOTES == Exclude"
        if not vsn:
            return True, "empty Video_name_SEQ"
        return False, ""

    mask_drop = []
    reasons = []
    for _, row in lut_df.iterrows():
        ex, reason = _is_excluded(row)
        mask_drop.append(ex)
        reasons.append(reason)

    mask_drop_s = pd.Series(mask_drop, index=lut_df.index)
    dropped = lut_df[mask_drop_s].copy()
    dropped["exclusion_reason"] = [r for r, m in zip(reasons, mask_drop) if m]
    kept = lut_df[~mask_drop_s].copy()
    return kept, dropped


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_tracking_filename(video_file: str) -> Tuple[str, str]:
    """
    Extract (stem, lane) from a cropped tracking filename.

    '07-28-25_4540_B_4541_F_Test3-004_Front.seq' → ('07-28-25_4540_B_4541_F_Test3-004', 'F')
    '...Back.seq' → (..., 'B')
    Also handles the .csv tracker output with _Front/_Back embedded.
    """
    name = Path(video_file).name
    for suffix, lane in [("_Front.seq", "F"), ("_Back.seq", "B")]:
        if name.endswith(suffix):
            return name[: -len(suffix)], lane
    # Fallback: look for _Front or _Back anywhere
    m = re.search(r"_(Front|Back)", name, re.IGNORECASE)
    if m:
        lane = "F" if m.group(1).lower() == "front" else "B"
        stem = name[: m.start()]
        return stem, lane
    return Path(video_file).stem, ""


def _parse_animal_lane_tokens(stem: str) -> Dict[str, str]:
    """
    Parse all {NNNN}_{B|F} pairs from a stem, return {lane: mouse_id}.
    Works regardless of token order.
    """
    tokens: Dict[str, str] = {}
    for m in re.finditer(r"(\d{4})_(B|F)\b", stem):
        tokens[m.group(2)] = m.group(1)
    return tokens


def _extract_date(stem: str) -> str:
    """Extract leading date token (MM-DD-YY or similar)."""
    m = re.match(r"^(\d{2}-\d{2}-\d{2})", stem)
    return m.group(1) if m else ""


def resolve_lut_row(
    stem: str,
    lane: str,
    lut_df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], str]:
    """
    Find the unique LUT row for (stem, lane).

    Returns (row_or_None, status_string).
    Status is one of: 'matched_exact', 'matched_token', 'unmatched', 'ambiguous'.
    """
    vsn_col = "Video_name_SEQ"
    lane_col = "Lane"

    if vsn_col not in lut_df.columns or lane_col not in lut_df.columns:
        return None, "unmatched_missing_columns"

    # 1. Exact match
    exact = lut_df[
        (lut_df[vsn_col].str.strip() == stem) &
        (lut_df[lane_col].str.strip() == lane)
    ]
    if len(exact) == 1:
        return exact.iloc[0], "matched_exact"
    if len(exact) > 1:
        return None, "ambiguous_exact"

    # 2. Token-based fallback: match date + animal-lane pairs
    date = _extract_date(stem)
    tokens = _parse_animal_lane_tokens(stem)

    if not tokens or not date or lane not in tokens:
        return None, "unmatched"

    target_mouse = tokens[lane]
    candidates = lut_df[lut_df[lane_col].str.strip() == lane].copy()

    matched = []
    for _, row in candidates.iterrows():
        lut_stem = _normalize(row[vsn_col])
        if not lut_stem:
            continue
        lut_date = _extract_date(lut_stem)
        lut_tokens = _parse_animal_lane_tokens(lut_stem)
        if lut_date == date and lut_tokens.get(lane) == target_mouse:
            matched.append(row)

    if len(matched) == 1:
        return matched[0], "matched_token"
    if len(matched) > 1:
        return None, "ambiguous_token"
    return None, "unmatched"


# ── join ───────────────────────────────────────────────────────────────────────

def join_metadata(
    tracking_files: List[str],
    lut_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join tracking CSVs to metadata LUT.

    Returns (master_df, join_report_df).

    master_df has one row per tracking sample with metadata columns appended.
    join_report_df has one row per tracking file with join outcome.
    """
    meta_cols = [
        "Mouse_ID", "Tail_ID", "Virus", "Injection", "phase",
        "Lane", "Video_name_SEQ", "Video_name_NOTES",
        "Trial_duration", "Arena_time", "Arena_Entry_time", "Arena_Exit_time",
        "NOTES",
    ]
    existing_meta_cols = [c for c in meta_cols if c in lut_df.columns]

    master_parts: List[pd.DataFrame] = []
    report_rows: List[dict] = []

    flagged_stems = set()

    for fpath in tracking_files:
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            report_rows.append({
                "tracking_file": fpath,
                "status": "read_error",
                "reason": str(exc),
                "mouse_id": None,
                "virus": None,
                "injection": None,
                "phase": None,
            })
            continue

        # Derive stem and lane from the video_file column (first row)
        if "video_file" in df.columns and len(df) > 0:
            vf = str(df["video_file"].iloc[0])
        else:
            vf = Path(fpath).name

        stem, lane = parse_tracking_filename(vf)
        row_match, status = resolve_lut_row(stem, lane, lut_df)

        report_row: dict = {
            "tracking_file": Path(fpath).name,
            "stem": stem,
            "lane": lane,
            "status": status,
        }

        if row_match is not None:
            for col in existing_meta_cols:
                report_row[col.lower()] = row_match.get(col, "")

            # Flag seq name warning
            vnn = _normalize(row_match.get("Video_name_NOTES", ""))
            if "incorrect" in vnn.lower() or ("seq" in vnn.lower() and "name" in vnn.lower()):
                report_row["seq_name_warning"] = vnn
                flagged_stems.add(stem)

            # Attach metadata to every row of the tracking df
            df = df.copy()
            df["mouse_id"] = _normalize(row_match.get("Mouse_ID", ""))
            df["tail_id"] = _normalize(row_match.get("Tail_ID", ""))
            df["virus"] = _normalize(row_match.get("Virus", "none"))
            df["injection"] = _normalize(row_match.get("Injection", ""))
            df["phase"] = _normalize(row_match.get("phase", ""))
            df["lane"] = lane
            df["track"] = "Front" if lane == "F" else "Back"
            df["seq_name_warning"] = vnn if vnn else ""

            master_parts.append(df)
        else:
            report_row["reason"] = f"Could not resolve to unique LUT row ({status})"

        report_rows.append(report_row)

    master_df = pd.concat(master_parts, ignore_index=True) if master_parts else pd.DataFrame()
    report_df = pd.DataFrame(report_rows)

    return master_df, report_df
