#!/bin/bash
#
# run_sync_qc.sbatch
# ---------------------------------------------------------------------------
# Renders visual QC figures (scripts/render_sync_qc.py) for selected sessions,
# on MedicineBow -- each figure needs a full sequential decode of the session's
# RGB video, which kept stalling locally over the VPN-mounted Alcova share.
#
# Default mode: corrected-vs-uncorrected homography check -- six real thermal
# frames per session with the RGB mouse outline warped through the per-lane
# translation-corrected homography (green, homography_calibration_inherited_
# corrected/) and the original inherited one (yellow dashed), at the session's
# trajectory sync. Right correction => green sits on the thermal mouse.
#
# Needs the outputs of run_trajectory_sync.sbatch in ${SYNC_DIR}
# (sync_estimates.csv + rgb_tracks/); session_sync.csv is regenerated here
# from sync_estimates.csv (deterministic) so the cluster clone has it.
#
# Usage:  sbatch slurm/run_sync_qc.sbatch                      (default sessions)
#         sbatch slurm/run_sync_qc.sbatch <session> [<session> ...]
# Output: ${SYNC_DIR}/qc/<session>_homography_corrected_qc.png (MedicineBow).
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=sync_qc
#SBATCH --time=01:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
SYNC_DIR="${REPO_DIR}/thermalFeatures/trajectory_sync"
HOMOGRAPHY_DIR="homography_calibration_inherited_corrected"

# Representative sessions: Regime A primary, Regime B stable, and one
# split-half-"unstable" lane from each regime (see homography_corrections.csv).
DEFAULT_SESSIONS=(
    07-29-25_4547_F_4552_B_Test3-005
    08-05-25_4540_F_4541_B_Test6-013
    07-09-25_4551_B_4550_F
    08-12-25_4539_F_4550_B_Test8-005
)
if [ "$#" -gt 0 ]; then SESSIONS=("$@"); else SESSIONS=("${DEFAULT_SESSIONS[@]}"); fi

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
source "${REPO_DIR}/.venv/bin/activate"

echo "sync_qc :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)"
echo "Sessions: ${SESSIONS[*]}"
for f in "${SYNC_DIR}/sync_estimates.csv" "${SYNC_DIR}/rgb_tracks"; do
    [ -e "${f}" ] || { echo "FATAL: ${f} not found -- run run_trajectory_sync.sbatch first" >&2; exit 1; }
done
[ -d "${REPO_DIR}/${HOMOGRAPHY_DIR}" ] || { echo "FATAL: ${HOMOGRAPHY_DIR}/ missing -- git pull?" >&2; exit 1; }

python scripts/consolidate_sync_estimates.py > /dev/null

# One process per session, in parallel (each is one video decode).
printf '%s\n' "${SESSIONS[@]}" | xargs -P "${SLURM_CPUS_PER_TASK:-4}" -I{} \
    python scripts/render_sync_qc.py --sessions {} \
        --alcova-root "${ALCOVA_TG}" \
        --homography-dir "${HOMOGRAPHY_DIR}" --compare-uncorrected

echo "Done: $(date). Figures in ${SYNC_DIR}/qc/"
