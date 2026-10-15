"""
Inputs for an all-lanes re-track (slurm/run_retrack_all.sbatch), one run folder per run tag
(thermalFeatures/<run tag>/, e.g. retrack_v3_2026-10).

The 85 lanes were produced by three tracking runs and four Stage 7 configs. A re-track changes
ONLY the tracking (the run's tracking config), so every lane keeps the calibration its existing
Stage 7 config already carries -- homography, sync, entry time, mouse_id, seq/video paths -- and
only tracking_csv / bouts_csv are repointed. (retrack_v2_2026-10 was built by this script's
earlier version, scripts/build_v2_retrack_config.py at 8babc9e.)

  lanes    (run locally, outputs committed) one row per lane (stage7 key, seq relpath under
           croppedSeqFiles/, stem, start frame; start frames from the three runs'
           retrack_start_frames.csv, each lane in exactly one), plus a snapshot of the four
           Stage 7 configs in <run-dir>/stage7_calibration_sources/ -- two of them were
           generated on the cluster and are not in git, so the job reads only the snapshot --
           and a copy of --tracking-config as <run-dir>/tracking_config.json, the config the
           job tracks with, so each run folder records what it was made with
  config   one combined Stage 7 config, <run-dir>/stage7/stage7_sessions_config_<run tag>.json:
           each existing entry with tracking_csv ->
           <run-dir>/trackingOutputs/<stem>_tracking_every10frames.csv and bouts_csv ->
           per-lane rows split from <run-dir>/retrack/bout_table.csv. mouse_id is kept; a
           lane whose new metadata-join mouse_id differs is reported and makes this exit 1.

Usage:
    python scripts/build_retrack_all_config.py lanes  --run-dir thermalFeatures/retrack_v3_2026-10 \
        --tracking-config tracking_config_retrack_fps10_v3.json        # locally, commit the outputs
    python scripts/build_retrack_all_config.py config --run-dir thermalFeatures/retrack_v3_2026-10  # in the job
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
STAGE7_CONFIGS = [
    "stage7_sessions_config.json",
    "thermalFeatures/stage7_inherited/stage7_sessions_config_inherited.json",
    "thermalFeatures/recover_2026-10/stage7/stage7_sessions_config_recover.json",
    "thermalFeatures/recover_lanes_2026-10/stage7/stage7_sessions_config_recover_lanes.json",
]
START_FRAME_CSVS = [
    "thermalFeatures/retrack_2026-10/retrack_start_frames.csv",
    "thermalFeatures/recover_2026-10/entry_detection/retrack_start_frames.csv",
    "thermalFeatures/recover_lanes_2026-10/entry_detection/retrack_start_frames.csv",
]


def stem_of(entry: dict) -> str:
    return Path(entry["thermal_seq_relpath"]).stem


def load_entries(repo: Path, configs) -> dict:
    """All lanes from the Stage 7 configs, keyed as in those configs (keys must not collide)."""
    entries = {}
    for cfg in configs:
        for key, entry in json.load(open(repo / cfg)).items():
            if key.startswith("_"):
                continue
            if key in entries:
                raise SystemExit(f"duplicate Stage 7 key {key} in {cfg}")
            entries[key] = {**entry, "_source_config": Path(cfg).name}
    return entries


def snapshot_paths(run_dir: str):
    return [f"{run_dir}/stage7_calibration_sources/{Path(c).name}" for c in STAGE7_CONFIGS]


def write_lanes(repo: Path, run_dir: str, tracking_config: str) -> None:
    starts = pd.concat([pd.read_csv(repo / f) for f in START_FRAME_CSVS], ignore_index=True)
    if starts.seq_stem.duplicated().any():
        raise SystemExit(f"stem in several start-frame CSVs: {sorted(starts.seq_stem[starts.seq_stem.duplicated()])}")
    starts = starts.set_index("seq_stem")
    snap = repo / run_dir / "stage7_calibration_sources"
    snap.mkdir(parents=True, exist_ok=True)
    for cfg in STAGE7_CONFIGS:
        shutil.copyfile(repo / cfg, snap / Path(cfg).name)
    rows = []
    for key, e in load_entries(repo, STAGE7_CONFIGS).items():
        stem = stem_of(e)
        if stem not in starts.index:
            raise SystemExit(f"{key}: no start frame for {stem}")
        rows.append(dict(key=key, seq_relpath=e["thermal_seq_relpath"], stem=stem,
                         start_frame=int(starts.loc[stem, "start_frame"])))
    if len(rows) != len(starts):
        raise SystemExit(f"{len(starts)} start frames but {len(rows)} Stage 7 lanes")
    out = repo / run_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "lanes.csv", index=False)
    starts.reset_index().to_csv(out / "retrack_start_frames.csv", index=False)
    shutil.copyfile(repo / tracking_config, out / "tracking_config.json")
    print(f"{len(rows)} lanes -> {run_dir}/lanes.csv")


def write_config(repo: Path, run_dir: str) -> None:
    bout_table = pd.read_csv(repo / run_dir / "retrack" / "bout_table.csv")
    joined = pd.read_csv(repo / run_dir / "retrack" / "metadata_join_report.csv")
    lut_mouse = {(r.stem, r.lane): int(r.mouse_id) for r in joined.itertuples() if pd.notna(r.mouse_id)}
    bouts_dir = repo / run_dir / "stage7" / "bouts"
    bouts_dir.mkdir(parents=True, exist_ok=True)
    config = {"_comment": (
        "GENERATED by scripts/build_retrack_all_config.py -- do not edit by hand. The four existing Stage 7 "
        f"configs combined, each lane's calibration unchanged, tracking_csv/bouts_csv repointed to the "
        f"{Path(run_dir).name} re-track ({run_dir}/tracking_config.json).")}
    problems = []
    for key, e in load_entries(repo, snapshot_paths(run_dir)).items():
        stem = stem_of(e)
        tracking = f"{run_dir}/trackingOutputs/{stem}_tracking_every10frames.csv"
        if not (repo / tracking).exists():
            problems.append(f"{key}: missing {tracking}")
            continue
        new_mouse = lut_mouse.get((stem.rsplit("_", 1)[0], e["track"]))
        if new_mouse != e["mouse_id"]:
            problems.append(f"{key}: Stage 7 mouse_id {e['mouse_id']} != new metadata join {new_mouse}")
        bouts = bout_table[bout_table.video_file == f"{stem}.seq"]
        bouts_rel = f"{run_dir}/stage7/bouts/{stem}_bout_rows.csv"
        bouts.to_csv(repo / bouts_rel, index=False)
        source = e.pop("_source_config")
        config[key] = {**e, "tracking_csv": tracking, "bouts_csv": bouts_rel, "n_bouts": len(bouts),
                       "calibration_from": source}
    out = repo / run_dir / "stage7" / f"stage7_sessions_config_{Path(run_dir).name}.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"{len(config) - 1} lanes -> {out.relative_to(repo)}")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("step", choices=["lanes", "config"])
    p.add_argument("--run-dir", required=True, help="Repo-relative, thermalFeatures/<run tag>")
    p.add_argument("--tracking-config", help="lanes step: the tracking config this run uses (copied into the run dir)")
    p.add_argument("--repo-dir", default=str(REPO))
    args = p.parse_args()
    if args.step == "lanes":
        if not args.tracking_config:
            p.error("lanes needs --tracking-config")
        write_lanes(Path(args.repo_dir), args.run_dir, args.tracking_config)
    else:
        write_config(Path(args.repo_dir), args.run_dir)


if __name__ == "__main__":
    main()
