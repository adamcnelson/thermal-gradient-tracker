#!/bin/bash
#
# run_recover_crop.sbatch
# ---------------------------------------------------------------------------
# First step of recovering sessions that never got a homography because they
# were never cropped (2026-10-08): 07-09-25_4552_F and three rehabituation
# sessions. (07-23-25 rehabituation-002's raw .seq is a 337 KB stub -- not
# recoverable from Alcova.)
#
#   1. Reproduction gate: re-crop 07-08-25_4541_B_4547_F.seq with the cropper's
#      crop_config.json and byte-compare both lanes with the existing
#      croppedSeqFiles/Test_1 outputs. If they differ, the config/tool is not
#      what produced the existing corpus -> stop.
#   2. Crop the 4 raw sessions (in parallel) with the same config.
#   3. Initial tracking of the new lanes (tracking_config_retrack_fps10.json,
#      auto-detected start) -- only needed to estimate their RGB<->thermal sync;
#      they are re-tracked from their RGB entry frames later, like the others.
#
# The cropper (thermal-gradient-seq-cropper) is run from its Alcova location,
# read-only: PYTHONDONTWRITEBYTECODE stops Python writing __pycache__ there.
# All output on MedicineBow: ${RUN_DIR}/
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_crop
#SBATCH --time=06:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
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
RUN_DIR="${REPO_DIR}/thermalFeatures/recover_2026-10"

# Reproduction gate input: raw file + its existing (June batch) cropped outputs.
REPRO_RAW="${ALCOVA_TG}/Process_Jason/Test_1/07-08-25_4541_B_4547_F.seq"
REPRO_EXISTING="${ALCOVA_TG}/croppedSeqFiles/Test_1"
# Sessions to crop: raw file -> output subfolder (named like croppedSeqFiles/).
RAW_FILES=(
    "${ALCOVA_TG}/Process_Jason/Test_1/07-09-25_4552_F.seq|Test_1"
    "${ALCOVA_TG}/07-22-25_Rehabituation/07-22-25_4548_B_4539_F_rehabituation-002.seq|07-22-25_Rehabituation"
    "${ALCOVA_TG}/07-22-25_Rehabituation/07-22-25_4540_B_4550_F_rehabituation-003.seq|07-22-25_Rehabituation"
    "${ALCOVA_TG}/07-23-25_Rehabituation/07-23-25_4541_B_4551_F_rehabituation-001.seq|07-23-25_Rehabituation"
)

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}/repro" "${RUN_DIR}/croppedSeqFiles" "${RUN_DIR}/trackingOutputs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_crop :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH" >&2; exit 1; }
for f in "${CROP_CONFIG}" "${REPRO_RAW}" "${REPRO_EXISTING}/07-08-25_4541_B_4547_F_Front.seq"; do
    [ -e "${f}" ] || { echo "FATAL: missing ${f}" >&2; exit 1; }
done

echo "=== [1/3] reproduction gate :: $(date) ==="
python "${CROPPER}/scripts/crop_seq.py" --input "${REPRO_RAW}" --config "${CROP_CONFIG}" \
    --output-dir "${RUN_DIR}/repro" --overwrite
for lane in Front Back; do
    if cmp -s "${RUN_DIR}/repro/07-08-25_4541_B_4547_F_${lane}.seq" "${REPRO_EXISTING}/07-08-25_4541_B_4547_F_${lane}.seq"; then
        echo "REPRO OK: ${lane} byte-identical to the existing crop"
    else
        echo "FATAL: re-cropped ${lane} differs from the existing crop -- crop_config.json/tool is not what produced the corpus; stopping." >&2
        exit 1
    fi
done
rm -f "${RUN_DIR}"/repro/*.seq   # proven identical; keep only the log

echo "=== [2/3] crop 4 sessions :: $(date) ==="
printf '%s\n' "${RAW_FILES[@]}" | xargs -P 4 -I{} sh -c '
    raw="${1%%|*}"; sub="${1##*|}"
    python "$2/scripts/crop_seq.py" --input "$raw" --config "$3" --output-dir "$4/croppedSeqFiles/$sub" --overwrite \
        > "$4/crop_$(basename "$raw" .seq).log" 2>&1 && echo "CROP OK   $(basename "$raw")" || echo "CROP FAIL $(basename "$raw")"' \
    _ {} "${CROPPER}" "${CROP_CONFIG}" "${RUN_DIR}" | tee "${RUN_DIR}/crop_status.txt"
if grep -q "CROP FAIL" "${RUN_DIR}/crop_status.txt"; then
    echo "FATAL: a crop failed -- see ${RUN_DIR}/crop_*.log" >&2; exit 1
fi
ls -la "${RUN_DIR}"/croppedSeqFiles/*/

echo "=== [3/3] initial tracking (auto start) :: $(date) ==="
python scripts/batch_track_temperatures.py --input-dir "${RUN_DIR}/croppedSeqFiles" --recursive \
    --config "${REPO_DIR}/tracking_config_retrack_fps10.json" --output-dir "${RUN_DIR}/trackingOutputs"

echo "Done: $(date). Cropped .seq + tracking in ${RUN_DIR}/ -- Globus back to Alcova for the next steps."
