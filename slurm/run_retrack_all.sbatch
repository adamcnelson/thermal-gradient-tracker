#!/bin/bash
#
# run_retrack_all.sbatch
# ---------------------------------------------------------------------------
# Re-track ALL 85 lanes into one run folder, thermalFeatures/${RUN_TAG}/, with the tracking
# config snapshotted there (tracking_config.json), then rebuild everything downstream of
# the tracking in one place. Submit with the run tag:
#
#   sbatch --export=ALL,RUN_TAG=retrack_v3_2026-10 slurm/run_retrack_all.sbatch
#
# Runs: retrack_v2_2026-10 (job 24611205, this script as run_retrack_v2.sbatch at 8babc9e;
# tracker v2), retrack_v3_2026-10 (v2 + mouse_roi_clip_fallback).
#
#   1. tracking, ${TRACK_PARALLEL} lanes at a time, each from the same start frame as
#      before (lanes.csv + tracking_config.json, built locally by
#      build_retrack_all_config.py lanes and committed)
#   2. bouts + LUT join + treatment analysis on the 85 lanes -> ONE master/bout table
#      (replaces the three per-run masters)
#   3. Stage 7 config: the four existing configs' calibration (homography, sync, entry,
#      mouse_id) unchanged, tracking/bouts repointed (build_retrack_all_config.py config)
#   4. Stage 7, ${STAGE7_PARALLEL} lanes at a time
#
# Only the tracker changes vs the current outputs. Afterwards (locally, from the repo root):
# Globus ${RUN_DIR} back to Alcova SLURM_RESULTS/${RUN_TAG}/ and thermalFeatures/${RUN_TAG}/,
# then join into the run's OWN folder (never landmark_outputs/, which the retrack_2026-10 R
# analysis reads):
#   python scripts/join_landmark_metadata.py \
#       --landmark-dir thermalFeatures/${RUN_TAG}/stage7/landmark_outputs \
#       --velocity-from thermalFeatures/${RUN_TAG}/retrack/master_tracking_with_metadata.csv \
#       --output-dir thermalFeatures/${RUN_TAG}/landmark_join
#   cd r_analysis_RGB_Thermal && DATA_VERSION=${RUN_TAG} scripts/run_all.sh
#     -> r_analysis_RGB_Thermal/output/${RUN_TAG}/ (needs a ${RUN_TAG} entry in R/data_versions.R)
#
# Inputs read from Alcova; all output on MedicineBow: ${RUN_DIR}/
# ---------------------------------------------------------------------------
# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=retrack_all
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
: "${RUN_TAG:?set it: sbatch --export=ALL,RUN_TAG=<run tag> slurm/run_retrack_all.sbatch}"
RUN_DIR_REL="thermalFeatures/${RUN_TAG}"
RUN_DIR="${REPO_DIR}/${RUN_DIR_REL}"
CONFIG_TRACKING="${RUN_DIR_REL}/tracking_config.json"
STAGE7_CONFIG="${RUN_DIR}/stage7/stage7_sessions_config_${RUN_TAG}.json"
N_LANES=85
TRACK_PARALLEL=8
STAGE7_PARALLEL="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}/trackingOutputs" "${RUN_DIR}/retrack" "${RUN_DIR}/track_logs" \
         "${RUN_DIR}/stage7/logs" "${RUN_DIR}/stage7/landmark_outputs"
cd "${REPO_DIR}"

module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "retrack_all ${RUN_TAG} :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date) :: $(git rev-parse --short HEAD)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
for f in "${CONFIG_TRACKING}" analysis_config.json metadata/LUT_CLEAN_July6.csv "${RUN_DIR_REL}/lanes.csv" \
         "${RUN_DIR_REL}"/stage7_calibration_sources/stage7_sessions_config{,_inherited,_recover,_recover_lanes}.json; do
    [ -f "${f}" ] || { echo "FATAL: missing ${REPO_DIR}/${f}" >&2; exit 1; }
done
grep -q '"blob_selection": "warmest"' "${CONFIG_TRACKING}" || { echo "FATAL: ${CONFIG_TRACKING} is not a v2+ tracker config" >&2; exit 1; }
[ -z "$(ls "${RUN_DIR}"/trackingOutputs/*.csv 2>/dev/null)" ] || { echo "FATAL: ${RUN_DIR} already has tracking output; use a new RUN_TAG" >&2; exit 1; }

# Keep going past individual lane failures; each step's count check below stops the job.
# xargs appends each lane's "<seq relpath> <start frame>" (lanes.csv columns 2 and 4) after the
# fixed args, so inside sh: $1 Alcova root, $2 tracking config, $3 run dir, $4 relpath, $5 start.
echo "=== [1/4] tracking, ${TRACK_PARALLEL} lanes at a time :: $(date) ==="
tail -n +2 "${RUN_DIR_REL}/lanes.csv" | cut -d, -f2,4 | tr ',' ' ' | \
xargs -P "${TRACK_PARALLEL}" -n 2 sh -c \
    'stem=$(basename "$4" .seq); ${TRACK_CMD:-python} scripts/track_temperatures.py --input "$1/croppedSeqFiles/$4" \
         --config "$2" --output-dir "$3/trackingOutputs" --tracking-start-frame "$5" --overwrite \
         > "$3/track_logs/${stem}.log" 2>&1 && echo "OK   ${stem}" || echo "FAIL ${stem}"' \
    _ "${ALCOVA_TG}" "${CONFIG_TRACKING}" "${RUN_DIR}" | tee "${RUN_DIR}/track_status.txt" || true
n=$( (ls "${RUN_DIR}"/trackingOutputs/*_tracking_every10frames.csv 2>/dev/null || true) | wc -l)
[ "${n}" -eq "${N_LANES}" ] || { echo "FATAL: expected ${N_LANES} tracking CSVs, got ${n}" >&2; grep FAIL "${RUN_DIR}/track_status.txt" >&2; exit 1; }

echo "=== [2/4] bouts + LUT join + treatment analysis :: $(date) ==="
python scripts/batch_compute_bouts.py --input-dir "${RUN_DIR_REL}/trackingOutputs" --config analysis_config.json \
    --output-dir "${RUN_DIR_REL}/retrack" --recursive --overwrite
python scripts/join_metadata.py --tracking-dir "${RUN_DIR_REL}/trackingOutputs" --bouts-dir "${RUN_DIR_REL}/retrack" \
    --metadata metadata/LUT_CLEAN_July6.csv --config analysis_config.json --output-dir "${RUN_DIR_REL}/retrack" --overwrite
python scripts/analyze_treatment_effects.py --master "${RUN_DIR_REL}/retrack/master_tracking_with_metadata.csv" \
    --bout-table "${RUN_DIR_REL}/retrack/bout_table.csv" --config analysis_config.json \
    --output-dir "${RUN_DIR_REL}/retrack" --overwrite

echo "=== [3/4] Stage 7 config :: $(date) ==="
python scripts/build_retrack_all_config.py config --run-dir "${RUN_DIR_REL}"
python -c "import json; print('\n'.join(k for k in json.load(open('${STAGE7_CONFIG}')) if not k.startswith('_')))" \
    > "${RUN_DIR}/stage7/lanes.txt"
n=$(wc -l < "${RUN_DIR}/stage7/lanes.txt")
[ "${n}" -eq "${N_LANES}" ] || { echo "FATAL: expected ${N_LANES} lanes in the Stage 7 config, got ${n}" >&2; exit 1; }

echo "=== [4/4] Stage 7, ${STAGE7_PARALLEL} lanes at a time :: $(date) ==="
xargs -P "${STAGE7_PARALLEL}" -I{} sh -c \
    'python scripts/stage7_real_run.py --sessions "$1" --sessions-config "$2" --repo-dir "$3" \
         --thermal-seq-root "$4/croppedSeqFiles" --rgb-video-root "$4/Process_Jason" \
         --output-dir "$5/landmark_outputs" > "$5/logs/$1.log" 2>&1 && echo "OK   $1" || echo "FAIL $1"' \
    _ {} "${STAGE7_CONFIG}" "${REPO_DIR}" "${ALCOVA_TG}" "${RUN_DIR}/stage7" < "${RUN_DIR}/stage7/lanes.txt" \
    | tee "${RUN_DIR}/stage7/lane_status.txt" || true

n_fail=$(grep -c "^FAIL" "${RUN_DIR}/stage7/lane_status.txt" || true)
echo "Done: $(date). $(grep -c '^OK' "${RUN_DIR}/stage7/lane_status.txt" || true) OK, ${n_fail} failed."
if [ "${n_fail}" -gt 0 ]; then
    grep "^FAIL" "${RUN_DIR}/stage7/lane_status.txt"
    echo "See ${RUN_DIR}/stage7/logs/<lane>.log for each failure." >&2
    exit 1
fi
echo "Globus ${RUN_DIR} back to Alcova (SLURM_RESULTS/${RUN_TAG}/)."
