#!/bin/bash
#
# run_recover_sync.sbatch
# ---------------------------------------------------------------------------
# RGB<->thermal sync for the 5 recovered sessions (10 lanes) that got an inherited
# homography in 2026-10: 07-08-25_4541_B_4547_F and 07-11-25_4539_F_4541_B (already
# cropped + tracked in the July full run; orientation was inconclusive, assigned
# "vertical" by date) and the 3 rehabituation sessions cropped + tracked by
# run_recover_crop.sbatch.
#
# Same estimator as run_trajectory_sync.sbatch (scripts/estimate_sync_trajectory.py),
# restricted with --sessions. Differences:
#   - tracking comes from two runs: the July full run (Alcova) and the recover crop
#     job's trackingOutputs (MedicineBow); each lane uses whichever holds its CSV.
#   - --alt-regime-check also fits every lane under the other orientation's base
#     homography. Tested on known lanes (2026-10-12): right orientation loss 0.01-0.12,
#     wrong one 0.75-0.84 -> alt_loss should be far above loss for all 10 lanes.
#   - no --validate step: run_trajectory_sync.sbatch already validated this
#     environment on Test_3/4/7.
#
# Read from Alcova, write to MedicineBow only; Globus results back afterwards.
# Expected runtime: < 1 hour (10 RGB decodes in parallel).
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_sync
#SBATCH --time=02:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
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
TRACKING_JULY="${ALCOVA_TG}/SLURM_RESULTS/results_fullrun_mgms2_2026-07-28/trackingOutputs"
TRACKING_RECOVER="${REPO_DIR}/thermalFeatures/recover_2026-10/trackingOutputs"
RGB_VIDEO_ROOT="${ALCOVA_TG}/Process_Jason"   # its parent also roots *_Rehabituation/
OUTPUT_DIR="${REPO_DIR}/thermalFeatures/recover_2026-10/trajectory_sync"
SESSIONS=(
    07-08-25_4541_B_4547_F
    07-11-25_4539_F_4541_B
    07-22-25_4548_B_4539_F_rehabituation-002
    07-22-25_4540_B_4550_F_rehabituation-003
    07-23-25_4541_B_4551_F_rehabituation-001
)

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_sync :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
for d in "${TRACKING_JULY}" "${TRACKING_RECOVER}" "${RGB_VIDEO_ROOT}"; do
    [ -d "${d}" ] || { echo "FATAL: input directory not found: ${d}" >&2; exit 1; }
done

python scripts/estimate_sync_trajectory.py --sessions "${SESSIONS[@]}" \
    --tracking-dir "${TRACKING_JULY}" "${TRACKING_RECOVER}" \
    --rgb-video-root "${RGB_VIDEO_ROOT}" --output-dir "${OUTPUT_DIR}" \
    --jobs "${SLURM_CPUS_PER_TASK:-10}" --alt-regime-check

echo "Done: $(date). ${OUTPUT_DIR}/sync_estimates.csv (+ rgb_tracks/) -- Globus back to Alcova."
