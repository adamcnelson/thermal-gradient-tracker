#!/bin/bash
#
# run_entry_detection.sbatch
# ---------------------------------------------------------------------------
# Per-lane mouse-entry times (scripts/detect_entry_rgb.py) for every session
# in thermalFeatures/trajectory_sync/session_sync.csv: the hand drop-in and
# release, detected in the first 10 min of each RGB video and converted to the
# thermal clock with the session's trajectory sync. Runs the Test_3/4/7
# validation first (Front lanes vs their confirmed entry times).
#
# Needs run_trajectory_sync.sbatch's sync_estimates.csv in ${SYNC_DIR};
# session_sync.csv is regenerated here from it (deterministic).
#
# Output (MedicineBow): ${REPO_DIR}/thermalFeatures/entry_detection/
#     entry_times.csv, entry_times_validation.csv, contact_sheets/*.png
# Copy back via Globus and review the contact sheets.
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=entry_detection
#SBATCH --time=02:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One worker process per session video (--jobs below).
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
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
RGB_VIDEO_ROOT="/cluster/alcova/bedfordlab/ThermalGradient/Process_Jason"
SYNC_DIR="${REPO_DIR}/thermalFeatures/trajectory_sync"
JOBS="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
source "${REPO_DIR}/.venv/bin/activate"

echo "entry_detection :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
[ -f "${SYNC_DIR}/sync_estimates.csv" ] \
    || { echo "FATAL: ${SYNC_DIR}/sync_estimates.csv missing -- run run_trajectory_sync.sbatch first" >&2; exit 1; }
[ -d "${RGB_VIDEO_ROOT}" ] || { echo "FATAL: ${RGB_VIDEO_ROOT} not found" >&2; exit 1; }
python scripts/consolidate_sync_estimates.py > /dev/null

echo "=== [1/2] validate (Test_3/4/7) :: $(date) ==="
python scripts/detect_entry_rgb.py --validate --rgb-video-root "${RGB_VIDEO_ROOT}" --jobs 3

echo "=== [2/2] all sessions :: $(date) ==="
python scripts/detect_entry_rgb.py --rgb-video-root "${RGB_VIDEO_ROOT}" --jobs "${JOBS}"

echo "Done: $(date). Results in ${REPO_DIR}/thermalFeatures/entry_detection/"
