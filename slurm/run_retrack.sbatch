#!/bin/bash
#
# run_retrack.sbatch
# ---------------------------------------------------------------------------
# Re-run Stage 1 tracking for the inherited-homography session lanes, starting
# each lane at its RGB-confirmed mouse entry instead of the tracker's own
# auto-detected start -- which (2026-10-05) was >10s late in 48/63 lanes
# (median 145s, up to 611s never tracked) and fell back to frame 0 in 7 lanes
# (pre-entry frames tracked).
#
# Only the start frame changes vs the original 2026-07-28 full run: same
# cropped .seq inputs, same tracking code, and tracking_config_retrack_fps10.json
# (= tracking_config.json with the camera_fps=10 that run actually used).
#
# Steps:
#   1. retrack_start_frames.csv from entry_times.csv (run_entry_detection.sbatch), plus the
#      Test_3/4/7 lanes from stage7_sessions_config.json's verified entry times
#   2. copy the July run's tracking CSVs into a NEW dir, re-track the listed
#      lanes into it (unlisted files keep their July CSVs)
#   3. bouts -> metadata join -> treatment analysis with TODAY's code, twice:
#      on the July tracking (baseline) and on the re-tracked dir -- downstream
#      code/config has changed since July, so only baseline-vs-retrack isolates
#      the effect of the start-frame fix.
#
# Output (MedicineBow only): ${RUN_DIR}/{trackingOutputs,trackingOutputs_july,baseline,retrack}/
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=retrack
#SBATCH --time=08:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
ALCOVA_TG="/cluster/alcova/bedfordlab/ThermalGradient"
SEQ_INPUT_DIR="${ALCOVA_TG}/croppedSeqFiles"
JULY_TRACKING="${ALCOVA_TG}/SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/trackingOutputs"
ENTRY_TIMES="${REPO_DIR}/thermalFeatures/entry_detection/entry_times.csv"
RUN_DIR="${REPO_DIR}/thermalFeatures/retrack_2026-10"
CONFIG_TRACKING="${REPO_DIR}/tracking_config_retrack_fps10.json"
CONFIG_ANALYSIS="${REPO_DIR}/analysis_config.json"
LUT="${REPO_DIR}/metadata/LUT_CLEAN_July6.csv"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "retrack :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
for f in "${ENTRY_TIMES}" "${CONFIG_TRACKING}" "${CONFIG_ANALYSIS}" "${LUT}"; do
    [ -f "${f}" ] || { echo "FATAL: missing ${f}" >&2; exit 1; }
done
for d in "${SEQ_INPUT_DIR}" "${JULY_TRACKING}"; do
    [ -d "${d}" ] || { echo "FATAL: missing ${d}" >&2; exit 1; }
done
mkdir -p "${RUN_DIR}"

echo "=== [1/4] start frames :: $(date) ==="
python scripts/build_retrack_start_frames.py --entry-times "${ENTRY_TIMES}" \
    --include-stage7-config "${REPO_DIR}/stage7_sessions_config.json" \
    --output "${RUN_DIR}/retrack_start_frames.csv"

echo "=== [2/4] re-track listed lanes :: $(date) ==="
if [ ! -d "${RUN_DIR}/trackingOutputs" ]; then
    cp -r "${JULY_TRACKING}" "${RUN_DIR}/trackingOutputs"
fi
# cp from Alcova carries its ACL-derived mode bits (no POSIX access, even for the owner) -- the
# first run died on a PermissionError stat'ing a copied CSV. Always fix, also for a pre-existing copy.
chmod -R u+rwX "${RUN_DIR}/trackingOutputs"
python scripts/batch_track_temperatures.py \
    --input-dir "${SEQ_INPUT_DIR}" --recursive \
    --config "${CONFIG_TRACKING}" \
    --output-dir "${RUN_DIR}/trackingOutputs" \
    --start-frames-csv "${RUN_DIR}/retrack_start_frames.csv" --only-listed \
    --overwrite

downstream() {  # $1 = tracking dir, $2 = output dir
    mkdir -p "$2"
    python scripts/batch_compute_bouts.py --input-dir "$1" --config "${CONFIG_ANALYSIS}" \
        --output-dir "$2" --recursive --overwrite
    python scripts/join_metadata.py --tracking-dir "$1" --bouts-dir "$2" --metadata "${LUT}" \
        --config "${CONFIG_ANALYSIS}" --output-dir "$2" --overwrite
    python scripts/analyze_treatment_effects.py --master "$2/master_tracking_with_metadata.csv" \
        --bout-table "$2/bout_table.csv" --config "${CONFIG_ANALYSIS}" --output-dir "$2" --overwrite
}

echo "=== [3/4] downstream on July tracking (baseline) :: $(date) ==="
# A MedicineBow copy, so no downstream step can ever write next to Alcova's July outputs.
if [ ! -d "${RUN_DIR}/trackingOutputs_july" ]; then
    cp -r "${JULY_TRACKING}" "${RUN_DIR}/trackingOutputs_july"
fi
chmod -R u+rwX "${RUN_DIR}/trackingOutputs_july"
downstream "${RUN_DIR}/trackingOutputs_july" "${RUN_DIR}/baseline"

echo "=== [4/4] downstream on re-tracked data :: $(date) ==="
downstream "${RUN_DIR}/trackingOutputs" "${RUN_DIR}/retrack"

echo "Done: $(date). Results in ${RUN_DIR}/ -- Globus back to Alcova for review."
