#!/bin/bash
#
# run_recover_lanes_crop.sbatch
# ---------------------------------------------------------------------------
# Recover 4 post-craniotomy test lanes whose cropped .seq was never made in June -- the
# other lane of each session was cropped and is in the corpus (found 2026-10-13 by
# scripts/build_experiment_overview.py; raw .seq present at full size):
#   07-30-25_4548_F_4551_B_Test4-007   Front  (4548 DCZ)
#   08-08-25_4552_F_4547_B_Test7-022   Back   (4547 Vehicle)
#   08-11-25_4548_F_4551_B_Test8-026   Back   (4551 Vehicle)
#   08-21-25_4548_F_4551_B_Test10-002  Back   (4551 DCZ)
#
#   1. Crop each raw session (the cropper writes both lanes). Gate: the re-cropped EXISTING
#      lane must match its June crop in croppedSeqFiles/ within rounding
#      (compare_seq_crops.py --gate); then it is deleted, so only the missing lane is kept
#      and a Globus copy can never overwrite a canonical crop.
#   2. Initial tracking of the 4 new lanes (auto start; re-tracked from entry later).
#   3. Full-session RGB track of each new lane (scripts/compute_rgb_track.py) -- the input
#      for its sync check and per-lane homography correction.
#
# The cropper runs from its Alcova location, read-only (PYTHONDONTWRITEBYTECODE).
# All output on MedicineBow: ${RUN_DIR}/ -- Globus back afterwards.
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_lanes_crop
#SBATCH --time=06:00:00
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
export PYTHONDONTWRITEBYTECODE=1

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
REPO_DIR="/project/huddlevidmicro/anelso74/thermal-gradient-tracker"
ALCOVA_TG="/cluster/alcova/bedfordlab/ThermalGradient"
CROPPER="${ALCOVA_TG}/SEQcropper/thermal-gradient-seq-cropper"
CROP_CONFIG="${CROPPER}/crop_config.json"
RUN_DIR="${REPO_DIR}/thermalFeatures/recover_lanes_2026-10"
# session | crop subfolder (= Process_Jason subfolder) | existing lane word | missing lane (F/B) | RGB video
SESSIONS=(
    "07-30-25_4548_F_4551_B_Test4-007|07-30-25_Test_4|Back|F|2025-07-30_09-55-39.mp4"
    "08-08-25_4552_F_4547_B_Test7-022|08-08-25_Test_7|Front|B|2025-08-08_09-43-47.mp4"
    "08-11-25_4548_F_4551_B_Test8-026|08-11-25_Test_8|Front|B|2025-08-11_11-29-26.mp4"
    "08-21-25_4548_F_4551_B_Test10-002|08-21-25_Test_10|Front|B|2025-08-21_10-45-56.mp4"
)

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}/croppedSeqFiles" "${RUN_DIR}/trackingOutputs" \
         "${RUN_DIR}/trajectory_sync/rgb_tracks" "${RUN_DIR}/logs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_lanes_crop :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH" >&2; exit 1; }
for s in "${SESSIONS[@]}"; do
    IFS='|' read -r sess sub have _ video <<< "${s}"
    for f in "${ALCOVA_TG}/Process_Jason/${sub}/${sess}.seq" "${ALCOVA_TG}/croppedSeqFiles/${sub}/${sess}_${have}.seq" \
             "${ALCOVA_TG}/Process_Jason/${sub}/${video}"; do
        [ -f "${f}" ] || { echo "FATAL: missing ${f}" >&2; exit 1; }
    done
done

echo "=== [1/3] crop + gate on the existing lane :: $(date) ==="
printf '%s\n' "${SESSIONS[@]}" | xargs -P 4 -I{} sh -c '
    IFS="|" read -r sess sub have miss video <<EOF
$1
EOF
    out="$4/croppedSeqFiles/$sub"; mkdir -p "$out"
    python "$2/scripts/crop_seq.py" --input "$5/Process_Jason/$sub/$sess.seq" --config "$3" --output-dir "$out" \
        --overwrite > "$4/logs/crop_$sess.log" 2>&1 || { echo "CROP FAIL $sess"; exit 0; }
    if python scripts/compare_seq_crops.py "$out/${sess}_$have.seq" "$5/croppedSeqFiles/$sub/${sess}_$have.seq" --gate \
        > "$4/logs/gate_$sess.log" 2>&1; then
        rm -f "$out/${sess}_$have.seq"; echo "CROP OK   $sess (gate PASS on $have; kept the missing lane only)"
    else
        rm -f "$out/${sess}"_*.seq; echo "GATE FAIL $sess (see logs/gate_$sess.log; outputs removed)"
    fi' _ {} "${CROPPER}" "${CROP_CONFIG}" "${RUN_DIR}" "${ALCOVA_TG}" | tee "${RUN_DIR}/crop_status.txt"
if grep -q "FAIL" "${RUN_DIR}/crop_status.txt"; then
    echo "FATAL: a crop or gate failed -- see ${RUN_DIR}/logs/" >&2; exit 1
fi
ls -la "${RUN_DIR}"/croppedSeqFiles/*/

echo "=== [2/3] initial tracking (auto start) :: $(date) ==="
python scripts/batch_track_temperatures.py --input-dir "${RUN_DIR}/croppedSeqFiles" --recursive \
    --config "${REPO_DIR}/tracking_config_retrack_fps10.json" --output-dir "${RUN_DIR}/trackingOutputs"

echo "=== [3/3] RGB tracks of the new lanes :: $(date) ==="
printf '%s\n' "${SESSIONS[@]}" | xargs -P 4 -I{} sh -c '
    IFS="|" read -r sess sub have miss video <<EOF
$1
EOF
    python scripts/compute_rgb_track.py --video "$3/Process_Jason/$sub/$video" --lane "$miss" --session-label "$sess" \
        --output-dir "$2/trajectory_sync/rgb_tracks" > "$2/logs/rgb_track_${sess}_$miss.log" 2>&1 \
        && echo "RGB OK   $sess $miss" || echo "RGB FAIL $sess $miss"' _ {} "${RUN_DIR}" "${ALCOVA_TG}" \
    | tee "${RUN_DIR}/rgb_status.txt"
if grep -q "FAIL" "${RUN_DIR}/rgb_status.txt"; then
    echo "FATAL: an RGB track failed -- see ${RUN_DIR}/logs/" >&2; exit 1
fi
ls -la "${RUN_DIR}/trackingOutputs"/*.csv "${RUN_DIR}/trajectory_sync/rgb_tracks/"
echo "Done: $(date). Globus ${RUN_DIR} back to Alcova (SLURM_RESULTS/)."
