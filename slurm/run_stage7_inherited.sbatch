#!/bin/bash
#
# run_stage7_inherited.sbatch
# ---------------------------------------------------------------------------
# Stage 7 (stage7_real_run.py: RGB+thermal registration / landmark measurement)
# for every inherited-homography lane (35 sessions, 65 lanes), several lanes at a
# time -- each lane is one sequential RGB video decode (~15 min), so serial would
# take ~16 h.
#
# Steps:
#   1. session_sync_results.json from sync_estimates.csv (deterministic)
#   2. scripts/build_stage7_inherited_config.py: Stage 7 config generated from the
#      corrected homographies, trajectory syncs, RGB entry times and the re-tracked
#      tracking/bouts (run_retrack.sbatch output) -- see that script's docstring
#   3. one stage7_real_run.py process per lane, ${PARALLEL} at a time, each logged
#      to ${RUN_DIR}/logs/<lane>.log; failed lanes are listed at the end
#
# The human-verified Test_3/4/7 config (stage7_sessions_config.json) is not used.
# Inputs read from Alcova; all output on MedicineBow: ${RUN_DIR}/
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=stage7_inherited
#SBATCH --time=10:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
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
RETRACK_DIR="thermalFeatures/retrack_2026-10"          # repo-relative, from run_retrack.sbatch
RUN_DIR_REL="thermalFeatures/stage7_inherited"
RUN_DIR="${REPO_DIR}/${RUN_DIR_REL}"
CONFIG="${RUN_DIR}/stage7_sessions_config_inherited.json"
PARALLEL="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}/logs" "${RUN_DIR}/landmark_outputs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "stage7_inherited :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)  parallel=${PARALLEL}"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
for f in "${RETRACK_DIR}/retrack/bout_table.csv" "${RETRACK_DIR}/retrack/metadata_join_report.csv" \
         "${RETRACK_DIR}/retrack_start_frames.csv" "thermalFeatures/trajectory_sync/sync_estimates.csv"; do
    [ -f "${f}" ] || { echo "FATAL: missing ${REPO_DIR}/${f}" >&2; exit 1; }
done

echo "=== [1/3] session syncs :: $(date) ==="
python scripts/consolidate_sync_estimates.py > /dev/null

echo "=== [2/3] Stage 7 config :: $(date) ==="
python scripts/build_stage7_inherited_config.py --retrack-dir "${RETRACK_DIR}" \
    --output "${RUN_DIR_REL}/stage7_sessions_config_inherited.json" --bouts-out-dir "${RUN_DIR_REL}/bouts"

echo "=== [3/3] Stage 7, ${PARALLEL} lanes at a time :: $(date) ==="
python -c "import json; print('\n'.join(k for k in json.load(open('${CONFIG}')) if not k.startswith('_')))" \
    > "${RUN_DIR}/lanes.txt"
echo "$(wc -l < "${RUN_DIR}/lanes.txt") lanes"
# Keep going past individual lane failures; they are reported below.
xargs -P "${PARALLEL}" -I{} sh -c \
    'python scripts/stage7_real_run.py --sessions "$1" --sessions-config "$2" --repo-dir "$3" \
         --thermal-seq-root "$4/croppedSeqFiles" --rgb-video-root "$4/Process_Jason" \
         --output-dir "$5/landmark_outputs" > "$5/logs/$1.log" 2>&1 && echo "OK   $1" || echo "FAIL $1"' \
    _ {} "${CONFIG}" "${REPO_DIR}" "${ALCOVA_TG}" "${RUN_DIR}" < "${RUN_DIR}/lanes.txt" \
    | tee "${RUN_DIR}/lane_status.txt" || true

n_fail=$(grep -c "^FAIL" "${RUN_DIR}/lane_status.txt" || true)
echo "Done: $(date). $(grep -c '^OK' "${RUN_DIR}/lane_status.txt" || true) OK, ${n_fail} failed."
if [ "${n_fail}" -gt 0 ]; then
    grep "^FAIL" "${RUN_DIR}/lane_status.txt"
    echo "See ${RUN_DIR}/logs/<lane>.log for each failure." >&2
    exit 1
fi
