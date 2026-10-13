#!/bin/bash
#
# run_recover_stage7.sbatch
# ---------------------------------------------------------------------------
# Stage 7 (stage7_real_run.py) for the 10 recovered lanes (5 sessions, 2026-10), all
# lanes in parallel -- same per-lane invocation as run_stage7_inherited.sbatch.
#
# Config + per-lane bouts are generated locally and committed (force-added from the
# git-ignored thermalFeatures/): thermalFeatures/recover_2026-10/stage7/
#   stage7_sessions_config_recover.json  (build_stage7_inherited_config.py --sessions ...,
#                                         recovered sync/entry/re-track, LUT mouse IDs)
#   bouts/<stem>_bout_rows.csv
# Tracking comes from run_recover_retrack.sbatch's output, already at the config's path
# here (thermalFeatures/recover_2026-10/retrack/trackingOutputs/).
#
# Rehab videos are outside Process_Jason/; their rgb_video_relpath starts "../".
# Inputs read from Alcova; all output on MedicineBow: ${RUN_DIR}/
# ---------------------------------------------------------------------------

# SUBJECT TO CHANGE: filled in for Adam's MedicineBow setup (2026-10-05); per-user values
# will move to a git-ignored slurm/local.env so each clone (Adam, Lucas) sets its own.
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=recover_stage7
#SBATCH --time=06:00:00
#SBATCH --qos=fast
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
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
RUN_DIR="${REPO_DIR}/thermalFeatures/recover_2026-10/stage7"
CONFIG="${RUN_DIR}/stage7_sessions_config_recover.json"
PARALLEL="${SLURM_CPUS_PER_TASK:-10}"

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}/logs" "${RUN_DIR}/landmark_outputs"
cd "${REPO_DIR}"
module load gcc/14.2.0
module load python/3.12.0
export PATH="${REPO_DIR}/vendor/Image-ExifTool-13.59:${PATH}"
source "${REPO_DIR}/.venv/bin/activate"

echo "recover_stage7 :: job ${SLURM_JOB_ID:-N/A} on $(hostname) :: $(date)  parallel=${PARALLEL}"
command -v exiftool >/dev/null || { echo "FATAL: exiftool not on PATH (temperatures would be wrong)" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "FATAL: missing ${CONFIG}" >&2; exit 1; }
# Every lane's tracking CSV and bout rows must be where the config says.
python - "${CONFIG}" <<'EOF'
import json, sys, pathlib
cfg = json.load(open(sys.argv[1]))
missing = [p for k, e in cfg.items() if not k.startswith("_")
           for p in (e["tracking_csv"], e["bouts_csv"], e["homography_json"]) if not pathlib.Path(p).exists()]
if missing:
    sys.exit("FATAL: missing inputs:\n  " + "\n  ".join(missing))
EOF

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
echo "Globus ${RUN_DIR} back to Alcova (SLURM_RESULTS/recover_2026-10/)."
