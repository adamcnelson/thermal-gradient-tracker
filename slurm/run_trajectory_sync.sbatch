#!/bin/bash
#
# run_trajectory_sync.sbatch
# ---------------------------------------------------------------------------
# Estimates the RGB<->thermal sync (offset, drift, true thermal fps) for every
# session lane with an inherited homography (homography_calibration_inherited/),
# via scripts/estimate_sync_trajectory.py. These estimates are the missing
# input that keeps those sessions out of stage7_sessions_config.json (see the
# Stage 5 note in run_thermal_gradient.sbatch).
#
# Why a separate job, on the cluster: each lane needs a full sequential decode
# of its ~2000s RGB video (~65 lanes, ~30 GB read from Alcova). Run locally over
# the VPN-mounted Alcova share, the mount dropped/hung mid-run twice
# (2026-10-01/02). On MedicineBow, Alcova reads stay on the cluster network.
#
# What it does:
#   1. --validate : recompute Test_3/4/7's RGB tracks here and compare the
#                   estimated offsets to their human-anchored ones. Exits
#                   non-zero (stopping the job) if any lane is >2s off -- so a
#                   cluster-environment problem fails loudly before step 2.
#   2. batch      : RGB track + sync estimate for every inherited lane ->
#                   ${OUTPUT_DIR}/sync_estimates.csv (with per-lane flags; see
#                   the script docstring for what each flag means).
#
# Same storage model as run_thermal_gradient.sbatch: read from Alcova, write
# to MedicineBow only; copying results back is a separate human step.
# Expected runtime: well under 2 hours with 16 CPUs.
# ---------------------------------------------------------------------------

#SBATCH --account=<PROJECT_NAME>
#SBATCH --job-name=trajectory_sync
#SBATCH --time=04:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The script decodes one RGB video per worker process (--jobs below), so unlike
# the main pipeline this job DOES use file-level parallelism within one task.
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<USER_EMAIL>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# =============================================================================
# USER-CONFIG BLOCK
# =============================================================================
REPO_DIR="<PATH_TO_REPO_ON_MEDICINEBOW>/thermal-gradient-tracker"
VENV_PATH="${REPO_DIR}/.venv"

# Inputs on Alcova (read-only). TRACKING_DIR must be the tracking run the
# estimator was validated against (thermal-native centroids, Alcova crop).
ALCOVA_TG="/cluster/alcova/bedfordlab/ThermalGradient"
TRACKING_DIR="${ALCOVA_TG}/SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/trackingOutputs"
RGB_VIDEO_ROOT="${ALCOVA_TG}/Process_Jason"

# Output on MedicineBow only.
OUTPUT_DIR="${REPO_DIR}/thermalFeatures/trajectory_sync"
JOBS="${SLURM_CPUS_PER_TASK:-16}"

# =============================================================================
# ENVIRONMENT (same as run_thermal_gradient.sbatch -- see its comments)
# =============================================================================
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "FATAL: virtualenv not found at ${VENV_PATH} (see DEPLOY_MEDICINEBOW.md)" >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

echo "trajectory_sync :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
echo "Python: $(which python) ($(python --version 2>&1))  jobs=${JOBS}"
echo "Tracking (Alcova, read-only): ${TRACKING_DIR}"
echo "RGB video (Alcova, read-only): ${RGB_VIDEO_ROOT}"
echo "Output (MedicineBow):          ${OUTPUT_DIR}"

# =============================================================================
# PREFLIGHT
# =============================================================================
python -c "import numpy, cv2, pandas, scipy" \
    || { echo "FATAL: required Python packages failed to import." >&2; exit 1; }
for d in "${TRACKING_DIR}" "${RGB_VIDEO_ROOT}" "${REPO_DIR}/homography_calibration_inherited"; do
    [ -d "${d}" ] || { echo "FATAL: input directory not found: ${d}" >&2; exit 1; }
done
case "${OUTPUT_DIR}" in
    /cluster/alcova/*) echo "FATAL: OUTPUT_DIR points at Alcova (${OUTPUT_DIR}); this job must never write there." >&2; exit 1 ;;
esac
mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(--tracking-dir "${TRACKING_DIR}" --rgb-video-root "${RGB_VIDEO_ROOT}"
             --output-dir "${OUTPUT_DIR}" --jobs "${JOBS}")

# =============================================================================
# 1. Validation gate (Test_3/4/7 vs human-anchored offsets)
# =============================================================================
echo "=== [1/2] validate :: START $(date) ==="
python scripts/estimate_sync_trajectory.py --validate "${COMMON_ARGS[@]}"
echo "=== [1/2] validate :: END   $(date) ==="

# =============================================================================
# 2. All inherited-homography lanes
# =============================================================================
echo "=== [2/2] batch :: START $(date) ==="
python scripts/estimate_sync_trajectory.py "${COMMON_ARGS[@]}"
echo "=== [2/2] batch :: END   $(date) ==="

echo "Done: ${OUTPUT_DIR}/sync_estimates.csv (+ sync_estimates_validation.csv, rgb_tracks/)."
echo "Copy back to Alcova via Globus if wanted -- see DEPLOY_MEDICINEBOW.md step 12."
