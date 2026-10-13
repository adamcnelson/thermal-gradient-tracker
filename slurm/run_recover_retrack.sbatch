#!/bin/bash
#
# run_recover_retrack.sbatch
# ---------------------------------------------------------------------------
# Re-track the 10 recovered lanes (5 sessions, 2026-10) from their RGB entry frames,
# like run_retrack.sbatch did for the other 71 lanes. Start frames:
# thermalFeatures/recover_2026-10/entry_detection/retrack_start_frames.csv (committed;
# built locally by detect_entry_rgb.py --sync-dir thermalFeatures/recover_2026-10/
# trajectory_sync + build_retrack_start_frames.py, contact sheets checked, 07-11 Back
# overridden to thermal 0 -- see entry_time_overrides.csv).
#
# Only the tracking (the heavy part: ~12 GB of cropped .seq read from Alcova) runs here.
# Bouts / metadata join / Stage 7 config work on small CSVs and are run afterwards on
# the combined 81 lanes.
#
# Read from Alcova, write to MedicineBow only; Globus results back afterwards.
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_retrack
#SBATCH --time=06:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --mail-user=anelso74@uwyo.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
REPO_DIR="/project/huddlevidmicro/anelso74/thermal-gradient-tracker"
SEQ_INPUT_DIR="/cluster/alcova/bedfordlab/ThermalGradient/croppedSeqFiles"
START_FRAMES="${REPO_DIR}/thermalFeatures/recover_2026-10/entry_detection/retrack_start_frames.csv"
CONFIG_TRACKING="${REPO_DIR}/tracking_config_retrack_fps10.json"
OUT_DIR="${REPO_DIR}/thermalFeatures/recover_2026-10/retrack/trackingOutputs"

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_retrack :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
for f in "${START_FRAMES}" "${CONFIG_TRACKING}"; do
    [ -f "${f}" ] || { echo "FATAL: missing ${f}" >&2; exit 1; }
done
[ -d "${SEQ_INPUT_DIR}" ] || { echo "FATAL: missing ${SEQ_INPUT_DIR}" >&2; exit 1; }

python scripts/batch_track_temperatures.py \
    --input-dir "${SEQ_INPUT_DIR}" --recursive \
    --config "${CONFIG_TRACKING}" \
    --output-dir "${OUT_DIR}" \
    --start-frames-csv "${START_FRAMES}" --only-listed \
    --overwrite

n=$(ls "${OUT_DIR}"/*_tracking_every10frames.csv 2>/dev/null | wc -l)
echo "Done: $(date). ${n} tracking CSVs (expected 10) in ${OUT_DIR} -- Globus back to Alcova."
[ "${n}" -eq 10 ] || { echo "FATAL: expected 10 tracking CSVs, got ${n}" >&2; exit 1; }
