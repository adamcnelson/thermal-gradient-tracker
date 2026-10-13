"""
Experiment-history cheat sheet: per mouse, how many usable recordings exist in each condition,
plus a per-recording table saying why each recording is or isn't usable.

"Usable" = the recording's lane is in the current Stage 7 analysis corpus
(landmark_outputs/master_landmarks_with_metadata_bout.csv). Conditions come from the LUT:
    pre_cranio                Craniotomy == Pre-craniotomy (all habituation, Saline)
    post_saline               Craniotomy == Post, Injection == Saline (rehabituation)
    post_{dcz,vehicle}_{nestlet,no_nestlet}   Post test sessions, split by the LUT Nestlet column

Writes (--out-dir, default ../metadata next to the repo):
    experiment_overview_by_mouse.csv     one row per mouse: usable counts, then all-recorded counts
    experiment_overview_recordings.csv   one row per LUT recording, with status + reason

Usage:
    python scripts/build_experiment_overview.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LUT = REPO / "metadata" / "LUT_CLEAN_July6.csv"
MASTER_BOUT = REPO / "landmark_outputs" / "master_landmarks_with_metadata_bout.csv"
ORIENTATION = REPO / "thermalFeatures" / "orientation_check_grid" / "orientation_classification_final.csv"
HYBRID_RUN_LOG = REPO / "thermalFeatures" / "hybrid_calibration_qc" / "run_log.csv"

CONDITIONS = ["pre_cranio", "post_saline", "post_dcz_nestlet", "post_vehicle_nestlet",
              "post_dcz_no_nestlet", "post_vehicle_no_nestlet"]
# Recordings that are unusable for a reason not derivable from the files below (2026-10 recovery).
KNOWN_REASONS = {
    "07-09-25_4552_F": "raw recording is only ~60 s (aborted); dropped",
    "07-23-25_4540_B_4547_F_rehabituation-002": "raw .seq is a 337 KB stub; original recording needed",
}


def condition(r) -> str:
    if r.Craniotomy == "Pre-craniotomy":
        return "pre_cranio"
    if r.Injection == "Saline":
        return "post_saline"
    nest = "nestlet" if r.Nestlet == "Y" else "no_nestlet"
    return f"post_{r.Injection.lower()}_{nest}"


def unusable_reason(r, corrupted, uncropped) -> str:
    seq = r.Video_name_SEQ
    if pd.isna(seq):
        return "webcam only (no thermal recording)" if r.Camera == "webcam" else "no thermal .seq listed in LUT"
    if seq in KNOWN_REASONS:
        return KNOWN_REASONS[seq]
    if seq in corrupted:
        return "corrupted .seq on Alcova; original recording needed"
    if (seq, r.Lane) in uncropped:
        return "this lane was never cropped (raw .seq exists) -- recoverable"
    return "not in Stage 7 corpus (reason unknown)"


def main():
    parser = argparse.ArgumentParser(description="Per-mouse experiment-history overview")
    parser.add_argument("--out-dir", default=str(REPO.parent / "metadata"))
    args = parser.parse_args()

    lut = pd.read_csv(LUT)
    bouts = pd.read_csv(MASTER_BOUT, usecols=["session", "track", "mouse_id"])
    n_bouts = bouts.groupby(["session", "track"]).size()
    orient = pd.read_csv(ORIENTATION)
    corrupted = set(orient.loc[orient.data_status == "CORRUPTED", "Video_name_SEQ"])
    log = pd.read_csv(HYBRID_RUN_LOG)
    uncropped = set(zip(log.loc[log.status == "skip_no_cropped_seq", "session"],
                        log.loc[log.status == "skip_no_cropped_seq", "lane"]))

    rec = lut.assign(
        condition=[condition(r) for r in lut.itertuples()],
        n_bouts=[n_bouts.get((s, l), 0) for s, l in zip(lut.Video_name_SEQ, lut.Lane)],
    )
    rec["usable"] = rec.n_bouts > 0
    rec["reason_if_unusable"] = [
        "" if u else unusable_reason(r, corrupted, uncropped) for u, r in zip(rec.usable, lut.itertuples())]
    rec["virus"] = rec.Virus.fillna("none")
    recordings = rec[["Mouse_ID", "virus", "Tail_ID", "Date", "Test", "condition", "Injection", "Craniotomy",
                      "Nestlet", "Lane", "Video_name_SEQ", "usable", "n_bouts", "reason_if_unusable",
                      "NOTES"]].rename(columns=str.lower)
    recordings["date"] = pd.to_datetime(recordings.date, format="%m/%d/%y")
    recordings = recordings.sort_values(["mouse_id", "date", "lane"])

    def counts(df):
        return df.pivot_table(index="mouse_id", columns="condition", values="usable", aggfunc="size",
                              fill_value=0).reindex(columns=CONDITIONS, fill_value=0)

    usable = counts(recordings[recordings.usable]).add_prefix("usable_")
    recorded = counts(recordings).add_prefix("recorded_")
    ids = recordings.groupby("mouse_id").agg(virus=("virus", "first"), tail_id=("tail_id", "first"))
    by_mouse = ids.join(usable).join(recorded).fillna(0)
    by_mouse[[c for c in by_mouse if c.startswith(("usable_", "recorded_"))]] = \
        by_mouse[[c for c in by_mouse if c.startswith(("usable_", "recorded_"))]].astype(int)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_mouse.to_csv(out / "experiment_overview_by_mouse.csv")
    recordings.to_csv(out / "experiment_overview_recordings.csv", index=False, date_format="%Y-%m-%d")
    print(by_mouse.to_string())
    print(f"\n{recordings.usable.sum()}/{len(recordings)} recordings usable -> {out}")
    print(recordings[~recordings.usable].reason_if_unusable.value_counts().to_string())


if __name__ == "__main__":
    main()
