#!/bin/bash
#
# run_recover_lanes_stage7.sbatch
# ---------------------------------------------------------------------------
# Second (last) cluster step for the 4 recovered never-cropped test lanes (see
# run_recover_lanes_crop.sbatch): re-track from the RGB entry frame, bouts + LUT join,
# Stage 7 config, Stage 7 -- the same chain as run_recover_retrack.sbatch +
# run_recover_stage7.sbatch, in one job.
#
# Committed inputs (built locally, force-added from git-ignored thermalFeatures/):
#   ${RUN_DIR_REL}/entry_detection/retrack_start_frames.csv   new lanes only (contact sheets checked)
#   ${RUN_DIR_REL}/trajectory_sync/session_sync_results.json  the sessions' EXISTING syncs, unchanged
#     (each new lane's own estimate agreed within 0.15 s, so the in-corpus lanes stay untouched)
# Corrected homographies: homography_calibration_inherited_corrected/ (committed).
# The sessions' existing lanes have no start frame here, so the config generator skips them.
#
# Inputs read from Alcova; all output on MedicineBow: ${RUN_DIR}/{retrack,stage7}/
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_lanes_stage7
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
ALCOVA_TG="/cluster/alcova/bedfordlab/ThermalGradient"
RUN_DIR_REL="thermalFeatures/recover_lanes_2026-10"
RUN_DIR="${REPO_DIR}/${RUN_DIR_REL}"
RETRACK_REL="${RUN_DIR_REL}/retrack"
STAGE7_REL="${RUN_DIR_REL}/stage7"
CONFIG="${REPO_DIR}/${STAGE7_REL}/stage7_sessions_config_recover_lanes.json"
SESSIONS=(07-30-25_4548_F_4551_B_Test4-007 08-08-25_4552_F_4547_B_Test7-022
          08-11-25_4548_F_4551_B_Test8-026 08-21-25_4548_F_4551_B_Test10-002)
N_LANES=4

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/${RETRACK_REL}/trackingOutputs" "${REPO_DIR}/${RETRACK_REL}/retrack" \
         "${REPO_DIR}/${STAGE7_REL}/logs" "${REPO_DIR}/${STAGE7_REL}/landmark_outputs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_lanes_stage7 :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
for f in "${RUN_DIR_REL}/entry_detection/retrack_start_frames.csv" "${RUN_DIR_REL}/trajectory_sync/session_sync_results.json" \
         tracking_config_retrack_fps10.json analysis_config.json metadata/LUT_CLEAN_July6.csv; do
    [ -f "${f}" ] || { echo "FATAL: missing ${REPO_DIR}/${f}" >&2; exit 1; }
done

echo "=== [1/4] re-track from entry :: $(date) ==="
cp "${RUN_DIR_REL}/entry_detection/retrack_start_frames.csv" "${RETRACK_REL}/retrack_start_frames.csv"
python scripts/batch_track_temperatures.py --input-dir "${ALCOVA_TG}/croppedSeqFiles" --recursive \
    --config tracking_config_retrack_fps10.json --output-dir "${RETRACK_REL}/trackingOutputs" \
    --start-frames-csv "${RETRACK_REL}/retrack_start_frames.csv" --only-listed --overwrite
n=$(ls "${RETRACK_REL}"/trackingOutputs/*_tracking_every10frames.csv | wc -l)
[ "${n}" -eq "${N_LANES}" ] || { echo "FATAL: expected ${N_LANES} tracking CSVs, got ${n}" >&2; exit 1; }

echo "=== [2/4] bouts + LUT join :: $(date) ==="
python scripts/batch_compute_bouts.py --input-dir "${RETRACK_REL}/trackingOutputs" --config analysis_config.json \
    --output-dir "${RETRACK_REL}/retrack" --recursive --overwrite
python scripts/join_metadata.py --tracking-dir "${RETRACK_REL}/trackingOutputs" --bouts-dir "${RETRACK_REL}/retrack" \
    --metadata metadata/LUT_CLEAN_July6.csv --config analysis_config.json --output-dir "${RETRACK_REL}/retrack" --overwrite

echo "=== [3/4] Stage 7 config :: $(date) ==="
python scripts/build_stage7_inherited_config.py --retrack-dir "${RETRACK_REL}" \
    --sync-json "${RUN_DIR_REL}/trajectory_sync/session_sync_results.json" --sessions "${SESSIONS[@]}" \
    --output "${STAGE7_REL}/stage7_sessions_config_recover_lanes.json" --bouts-out-dir "${STAGE7_REL}/bouts"
python -c "import json; print('\n'.join(k for k in json.load(open('${CONFIG}')) if not k.startswith('_')))" \
    > "${REPO_DIR}/${STAGE7_REL}/lanes.txt"
n=$(wc -l < "${REPO_DIR}/${STAGE7_REL}/lanes.txt")
[ "${n}" -eq "${N_LANES}" ] || { echo "FATAL: expected ${N_LANES} lanes in the Stage 7 config, got ${n}" >&2; exit 1; }

echo "=== [4/4] Stage 7, ${N_LANES} lanes in parallel :: $(date) ==="
xargs -P "${N_LANES}" -I{} sh -c \
    'python scripts/stage7_real_run.py --sessions "$1" --sessions-config "$2" --repo-dir "$3" \
         --thermal-seq-root "$4/croppedSeqFiles" --rgb-video-root "$4/Process_Jason" \
         --output-dir "$5/landmark_outputs" > "$5/logs/$1.log" 2>&1 && echo "OK   $1" || echo "FAIL $1"' \
    _ {} "${CONFIG}" "${REPO_DIR}" "${ALCOVA_TG}" "${REPO_DIR}/${STAGE7_REL}" < "${REPO_DIR}/${STAGE7_REL}/lanes.txt" \
    | tee "${REPO_DIR}/${STAGE7_REL}/lane_status.txt" || true

n_fail=$(grep -c "^FAIL" "${REPO_DIR}/${STAGE7_REL}/lane_status.txt" || true)
echo "Done: $(date). $(grep -c '^OK' "${REPO_DIR}/${STAGE7_REL}/lane_status.txt" || true) OK, ${n_fail} failed."
if [ "${n_fail}" -gt 0 ]; then
    grep "^FAIL" "${REPO_DIR}/${STAGE7_REL}/lane_status.txt"; exit 1
fi
echo "Globus ${RUN_DIR}/retrack and ${RUN_DIR}/stage7 back to Alcova (SLURM_RESULTS/recover_lanes_2026-10/)."
