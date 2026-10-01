"""
Final RGB<->thermal orientation classification for all 49 sessions
(project_v8, "automate parts of the per-session calibration", 2026-09-29).

Combines:
  1. Individually-read sessions (24 definitive reads + 3 inconclusive), from
     the manual contact-sheet + corner-exit-read method (see
     render_orientation_contact_sheet.py / render_orientation_grid.py and
     project memory project_v8_orientation_detection.md for the full
     validation history -- 3 known sessions + chunk 1 (10 sessions) + the
     2-regime hypothesis test (11 sessions) + 6 extra spot-checks).
  2. A date-based bulk rule for every session NOT individually read: the
     2-regime hypothesis was confirmed with zero exceptions across every
     date group tested (11 "vertical" confirmations pre-Test4, 13
     "horizontal+vertical" confirmations Test4-onward) -- sessions before
     2025-07-30 get "vertical", sessions from 2025-07-30 onward get
     "horizontal+vertical".

Also flags the known corrupted/oversized .seq files (real data-integrity
issues found in the Alcova corpus 2026-09-29, unrelated to orientation
itself -- see project memory) so they aren't silently treated as usable.

Output: thermalFeatures/orientation_check_grid/orientation_classification_final.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metadata import load_lut, filter_excluded

OUT_DIR = Path(__file__).resolve().parent.parent / "thermalFeatures" / "orientation_check_grid"
REGIME_CUTOFF = pd.Timestamp("2025-07-30")  # first Test_4 date; confirmed camera-repositioning boundary

# (flip, confidence, source) for every session actually read this project (2026-09-29).
# "known" = pre-existing, fully calibrated (homography_calibration/ already has these).
VERIFIED = {
    "07-28-25_4540_B_4541_F_Test3-004": ("vertical", "known", "verified"),
    "07-30-25_4540_F_4541_B_Test4-008": ("horizontal+vertical", "known", "verified"),
    "08-07-25_4541_F_4540_B_Test7-020": ("horizontal+vertical", "known", "verified"),

    "07-08-25_4539_B_4540_F": ("vertical", "high", "verified"),
    "07-08-25_4541_B_4547_F": ("inconclusive", None, "attempted"),
    "07-08-25_4549_B_4548_F": ("vertical", "high", "verified"),
    "07-09-25_4551_B_4550_F": ("vertical", "high", "verified"),
    "07-09-25_4552_F": ("inconclusive", None, "attempted"),
    "07-10-25_4540_F_4547_B": ("vertical", "high", "verified"),
    "07-10-25_4549_F_4551_B": ("vertical", "medium", "verified"),
    "07-11-25_4539_F_4541_B": ("inconclusive", None, "attempted"),
    "07-11-25_4548_F_4550_B": ("vertical", "high", "verified"),
    "07-11-25_4552_B": ("vertical", "high", "verified"),

    "07-28-25_4548_B_4551_F_Test3-003": ("vertical", "presumed", "same-day-as-known"),
    "07-30-25_4548_F_4551_B_Test4-007": ("horizontal+vertical", "presumed", "same-day-as-known"),
    "07-22-25_4548_B_4539_F_rehabituation-002": ("vertical", "medium", "verified"),
    "07-23-25_4541_B_4551_F_rehabituation-001": ("vertical", "medium", "verified"),
    "08-04-25_4552_F_4547_B_Test5-005": ("horizontal+vertical", "medium", "verified"),
    "08-05-25_4548_F_4551_B_Test6-010": ("horizontal+vertical", "medium", "verified"),
    "08-11-25_4548_F_4551_B_Test8-026": ("horizontal+vertical", "low-medium", "verified"),
    "08-19-25_4551_F_4548_B_Test9-002": ("horizontal+vertical", "high", "verified"),
    "08-21-25_4548_F_4551_B_Test10-002": ("horizontal+vertical", "medium-high", "verified"),
    "08-25-25_4551_F_4548_B_Test11-008": ("horizontal+vertical", "high", "verified"),
    "08-25-25_4550_F_4539_B_Test11-001": ("horizontal+vertical", "medium", "verified"),
    "08-08-25_4550_F_4539_B_Test7-024": ("horizontal+vertical", "high", "verified"),
    "08-12-25_4539_F_4550_B_Test8-005": ("horizontal+vertical", "medium", "verified"),
    "08-22-25_4547_F_4552_B_Test10-005": ("horizontal+vertical", "medium", "verified"),
}

# Confirmed real, pre-existing (not caused by this work) data-integrity problems.
BAD_DATA = {
    "07-31-25_4550_F_4539_B_Test4-001": "seq is a 337,227-byte stub (~6GB expected)",
    "08-01-25_4551_F_4548_B_Test5-002": "seq is a 337,227-byte stub (~6GB expected)",
    "08-01-25_4541_F_4540_B_Test5-003": "seq is a 337,227-byte stub (~6GB expected)",
    "08-21-25_4540_F_4541_B_Test10-003": "seq is ~259GB, ~40x the normal size -- likely corrupted/duplicated",
}


def classify(row):
    stem = row["Video_name_SEQ"]
    if stem in VERIFIED:
        flip, confidence, source = VERIFIED[stem]
    else:
        flip = "vertical" if row["DateParsed"] < REGIME_CUTOFF else "horizontal+vertical"
        confidence = "date-inferred"
        source = "date-rule"
    return pd.Series(
        {"flip": flip, "confidence": confidence, "source": source,
         "data_status": "CORRUPTED" if stem in BAD_DATA else "ok",
         "data_note": BAD_DATA.get(stem, "")}
    )


if __name__ == "__main__":
    lut = load_lut(str(Path(__file__).resolve().parent.parent / "metadata" / "LUT_CLEAN_July6.csv"))
    kept, _ = filter_excluded(lut)
    sessions = (
        kept[["Video_name_SEQ", "Video_name", "Date"]]
        .drop_duplicates(subset="Video_name_SEQ")
        .dropna(subset=["Video_name_SEQ"])
        .reset_index(drop=True)
    )
    sessions["DateParsed"] = pd.to_datetime(sessions["Date"], format="%m/%d/%y")
    sessions = sessions.join(sessions.apply(classify, axis=1))
    sessions = sessions.sort_values("DateParsed").drop(columns=["DateParsed"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "orientation_classification_final.csv"
    sessions.to_csv(out_path, index=False)

    print(f"{len(sessions)} sessions classified -> {out_path}\n")
    print(sessions["flip"].value_counts(dropna=False))
    print()
    print(sessions["source"].value_counts())
    print()
    print(f"Corrupted-data sessions: {(sessions['data_status'] == 'CORRUPTED').sum()}")
    print(sessions[sessions["data_status"] == "CORRUPTED"][["Video_name_SEQ", "data_note"]].to_string(index=False))
