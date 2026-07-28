#!/bin/bash
#
# run_thermal_gradient.sbatch
# ---------------------------------------------------------------------------
# SLURM batch job for the thermal-gradient-tracker pipeline on the University
# of Wyoming ARCC "MedicineBow" cluster.
#
# This script is a *teaching artifact* as well as a working job — it is
# heavily commented so a teammate who has never used Slurm or MedicineBow
# before can understand what every line does. See slurm/DEPLOY_MEDICINEBOW.md
# for the full step-by-step deployment guide.
#
# What this job does (see README.md for full pipeline detail):
#   1. batch_track_temperatures.py  — track the mouse + extract temperatures
#                                      for every .seq file (compute-heavy,
#                                      this is almost all of the ~1 hour).
#   2. batch_compute_bouts.py       — detect stationary rest bouts from the
#                                      tracking output.
#   3. join_metadata.py             — join tracking/bout data to the
#                                      experimental metadata LUT.
#   4. analyze_treatment_effects.py — fit treatment-effect models and write
#                                      plots/tables.
#
# All four stages are already fully non-interactive in this codebase. The
# interactive/manual QC scripts (create_arena_mask.py, select_training_frames.py,
# train_mouse_detector.py) are retired and are NOT part of this job —
# tracking_config.json (including its arena_polygon) is a finalized, fixed
# input and is only ever *read* here, never regenerated.
#
# ---------------------------------------------------------------------------
# STORAGE MODEL — READ THIS BEFORE EDITING PATHS BELOW
# ---------------------------------------------------------------------------
# The data and code live on the bedfordlab "Alcova" project
# (/cluster/alcova/bedfordlab), a SEPARATE service from MedicineBow with a
# different permission model:
#   - Alcova uses Windows-style (ACL) permissions.
#   - MedicineBow uses POSIX permissions.
#   - READING from Alcova on MedicineBow is reliable.
#   - WRITING from MedicineBow to Alcova is NOT reliable (permission
#     translation across the ACL/POSIX boundary can fail).
#
# Therefore, in this job:
#   - INPUT  (.seq files)   is read from Alcova (or a staged MedicineBow copy).
#   - OUTPUT (everything this job writes) goes to MedicineBow storage
#     (/project or /gscratch) — NEVER to an Alcova path.
#   - Copying the finished results back to Alcova is a SEPARATE, human-run
#     step performed AFTER this job succeeds — see slurm/DEPLOY_MEDICINEBOW.md
#     step 12. It is intentionally NOT done inside this script.
# ---------------------------------------------------------------------------

# =============================================================================
# SLURM DIRECTIVES
# =============================================================================
# Every #SBATCH line below is a request to the scheduler. These are parsed by
# `sbatch`, not bash — they must appear before any executable command in the
# file (comments are fine).

# --account is REQUIRED by ARCC policy. Replace with your project account —
# this is the ARCC project you were added to; it's named in the welcome email
# from arcc-admin@uwyo.edu, or run `my_accounts` on a login node. The job will
# be rejected at submission time if this is left as the placeholder.
#SBATCH --account=<PROJECT_NAME>

# A human-readable name shown in `squeue`.
#SBATCH --job-name=thermal_gradient

# Wall-clock limit (HH:MM:SS). The pipeline takes ~1 hour serial on a desktop
# PC; 4 hours gives generous headroom for a slower/shared compute node and
# cluster variability without losing scheduling priority. If your dataset
# grows much larger than the one this was tuned on and 4h is not enough,
# raise --time (up to 12:00:00 on "Fast" QoS) or switch to "Normal" QoS for
# longer jobs — see DEPLOY_MEDICINEBOW.md "Troubleshooting".
#SBATCH --time=04:00:00

# QoS/partition selection. The desktop run is ~1 hour, well inside the
# "Fast" QoS's 12-hour cap, and "Fast" carries higher scheduling priority
# than the 3-day-cap default "Normal" QoS — so this job should target Fast.
# The exact partition/QoS name was not independently re-verified from inside
# this coding session against current ARCC docs; confirm with `sinfo` /
# `sacctmgr show qos` or ARCC support before your first submission and edit
# below if it differs from "fast".
#SBATCH --qos=fast
##SBATCH --partition=<CONFIRM_WITH_ARCC_IF_REQUIRED_ALONGSIDE_QOS>

# Single node — this job is not MPI/multi-node; it is one Python process
# running the four pipeline stages in sequence.
#SBATCH --nodes=1
#SBATCH --ntasks=1

# CPUs for this task. batch_track_temperatures.py (via src/batch.py) processes
# .seq files SERIALLY, one at a time — there is no internal multiprocessing or
# threading in this codebase (confirmed by review: no multiprocessing,
# ThreadPoolExecutor, or joblib usage in scripts/ or src/). A modest core
# count still helps because numpy/opencv/scikit-image can use a handful of
# threads internally for individual operations. 8 is a reasonable default;
# do not request a large core count expecting file-level parallelism — that
# would require the job-array approach described in DEPLOY_MEDICINEBOW.md,
# not more cores on this single task.
#SBATCH --cpus-per-task=8

# Memory must be requested explicitly on MedicineBow — `--mem=0` is
# disallowed by ARCC policy. 32G is a conservative starting point for
# frame-by-frame thermal video processing with numpy/opencv. If the job is
# killed with an out-of-memory (oom-kill) error (check the .err log and
# `sacct -j <jobid> --format=MaxRSS,State`), raise this value and resubmit.
#SBATCH --mem=32G

# Email notifications. ALL = job begin, end, fail, requeue, etc. Replace the
# placeholder with your actual email before submitting.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<USER_EMAIL>

# Slurm stdout/stderr logs. %x = job name, %j = job ID. The `logs/` directory
# is created below (mkdir -p) before Slurm needs it.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# FAIL FAST
# =============================================================================
# -e : exit immediately if any command exits non-zero
# -u : treat unset variables as an error
# -o pipefail : a pipeline's exit status is the last non-zero exit, not 0
# This ensures a broken stage stops the whole job (and reports failure via
# --mail-type=ALL) instead of silently continuing to the next stage on bad
# data.
set -euo pipefail

# =============================================================================
# USER-CONFIG BLOCK — every path a deployer must set lives here and ONLY here.
# =============================================================================
# Mirrors the README's philosophy: paths are passed as CLI arguments, source
# code is never edited to point at a particular machine's data.
#
# ---- CODE (MedicineBow, /project — durable, POSIX) -------------------------
# REPO_DIR   — where this repo lives on MedicineBow. Must be under /project
#              (or /gscratch), never $HOME (50 GB quota). The repo is
#              git-cloned/pulled here directly; it is not itself synced from
#              Alcova.
# VENV_PATH  — the one-time project virtualenv (see DEPLOY_MEDICINEBOW.md
#              "One-time environment setup"). This job only activates it, it
#              never runs `pip install` itself.
REPO_DIR="<PATH_TO_REPO_ON_MEDICINEBOW>/thermal-gradient-tracker"
VENV_PATH="${REPO_DIR}/.venv"

# ---- INPUT (Alcova, /cluster/alcova/bedfordlab — read-only from this job) --
# ALCOVA_SEQ_INPUT_DIR — directory containing the cropped .seq files, on
#     Alcova. Reading from Alcova on MedicineBow is reliable; this job never
#     writes here. For heavy/iterative runs you may instead stage a one-time
#     copy of the .seq files onto MedicineBow storage (via Globus/rsync, done
#     by a human beforehand — see DEPLOY_MEDICINEBOW.md step 4) and point this
#     at that staged copy instead; either way this variable must resolve to a
#     READ-ONLY source as far as this job is concerned.
ALCOVA_SEQ_INPUT_DIR="/cluster/alcova/bedfordlab/<PATH_TO_CROPPED_SEQ_FILES>"

# Fixed config/metadata inputs — also only ever read, never written, and
# normally live inside the repo (MedicineBow /project) rather than on Alcova.
CONFIG_TRACKING="${REPO_DIR}/tracking_config.json"
CONFIG_ANALYSIS="${REPO_DIR}/analysis_config.json"
LUT="${REPO_DIR}/metadata/LUT_CLEAN_July6.csv"

# ---- OUTPUT (MedicineBow ONLY — /project or /gscratch) ----------------------
# OUTPUT_DIR / BOUTS_DIR — where this job writes everything it produces.
#
#   *** MUST resolve to a MedicineBow path (/project/... or /gscratch/...). ***
#   *** NEVER point these at /cluster/alcova/... — writing from MedicineBow ***
#   *** to Alcova is unreliable (POSIX -> ACL permission translation can    ***
#   *** silently fail) and is NOT done inside this job under any           ***
#   *** circumstance. Copying finished results back to Alcova is a         ***
#   *** separate, human-run step — see slurm/DEPLOY_MEDICINEBOW.md step 12.***
#
# Defaults below place outputs under REPO_DIR (i.e. MedicineBow /project),
# which satisfies this rule as long as REPO_DIR itself is never changed to
# point at Alcova.
OUTPUT_DIR="${REPO_DIR}/trackingOutputs"
BOUTS_DIR="${REPO_DIR}/bouts"

# Set to "--recursive" if ALCOVA_SEQ_INPUT_DIR has session subfolders (matches
# the README's "all subfolders at once" usage), or "" for a single flat
# folder.
RECURSIVE_FLAG="--recursive"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

echo "============================================================"
echo "thermal-gradient-tracker :: SLURM job"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Job name:    ${SLURM_JOB_NAME:-N/A}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Input  (Alcova, read-only): ${ALCOVA_SEQ_INPUT_DIR}"
echo "Output (MedicineBow only):  ${OUTPUT_DIR}"
echo "                             ${BOUTS_DIR}"
echo "============================================================"

# Load a Python module via LMOD. The exact module name/version available on
# MedicineBow was not independently re-verified from inside this coding
# session — run `module spider python` on a login node to find the right
# name (e.g. something like "python/3.11" or "miniconda3") and update this
# line. NOTE: unlike the quoted placeholders above, this is a bare shell
# command — do not wrap the module name in angle brackets (`<`/`>` are
# redirection operators to bash); just replace PYTHON_MODULE_PLACEHOLDER
# below with the real module name/version, e.g. `module load python/3.11`.
# See DEPLOY_MEDICINEBOW.md "One-time environment setup".
module load PYTHON_MODULE_PLACEHOLDER

# If exiftool is provided as an LMOD module on MedicineBow rather than a
# project-local install, load it here too (see DEPLOY_MEDICINEBOW.md step 5).
# Leave commented if you installed exiftool into the project space instead.
# module load EXIFTOOL_MODULE_PLACEHOLDER

# Activate the project virtualenv. This venv is created ONCE, ahead of time,
# per DEPLOY_MEDICINEBOW.md — this job only activates it, it never runs
# `pip install` itself (compute nodes may not have internet access, and
# installing packages is not compute-bound work anyway).
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "FATAL: virtualenv not found at ${VENV_PATH}" >&2
    echo "       Run the one-time setup in DEPLOY_MEDICINEBOW.md first." >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python:      $(which python)  ($(python --version 2>&1))"

# =============================================================================
# PREFLIGHT CHECKS — abort loudly before burning compute time on bad input.
# =============================================================================
echo "------------------------------------------------------------"
echo "Preflight checks"
echo "------------------------------------------------------------"

# 1) Cheap smoke test that the venv's packages actually import.
python -c "import numpy, cv2, pandas, pydantic, statsmodels, sklearn, skimage, matplotlib" \
    || { echo "FATAL: one or more required Python packages failed to import." >&2; exit 1; }
echo "OK: core Python packages import cleanly."

# 2) exiftool MUST resolve on the compute node (not just the login node —
#    LMOD module state and PATH can differ between them). Without it, the
#    pipeline silently falls back to an inaccurate linear temperature
#    approximation instead of the FLIR Planck-equation conversion — a
#    correctness bug, not a crash, so this MUST be a hard failure here.
if ! command -v exiftool >/dev/null 2>&1; then
    echo "FATAL: exiftool not found on compute node $(hostname)." >&2
    echo "       Temperatures would silently fall back to an inaccurate linear" >&2
    echo "       approximation without it (see README.md). Resolve this before" >&2
    echo "       resubmitting: check 'module spider exiftool', or contact" >&2
    echo "       arcc-help@uwyo.edu to request it be installed/made available." >&2
    exit 1
fi
echo "OK: exiftool resolved at $(command -v exiftool) ($(exiftool -ver))"

# 3) Input directory must exist and actually contain .seq files. Read-only
#    check against Alcova (or the staged copy) — this job never writes here.
if [ ! -d "${ALCOVA_SEQ_INPUT_DIR}" ]; then
    echo "FATAL: ALCOVA_SEQ_INPUT_DIR does not exist or is not readable: ${ALCOVA_SEQ_INPUT_DIR}" >&2
    exit 1
fi
if [ -z "$(find "${ALCOVA_SEQ_INPUT_DIR}" -iname '*.seq' -print -quit)" ]; then
    echo "FATAL: no .seq files found under ${ALCOVA_SEQ_INPUT_DIR}" >&2
    exit 1
fi
echo "OK: .seq input files found under ${ALCOVA_SEQ_INPUT_DIR}"

# 4) Config files and LUT must be present (all treated as fixed, committed
#    inputs — this job never regenerates tracking_config.json's arena_polygon
#    or any other config value).
for f in "${CONFIG_TRACKING}" "${CONFIG_ANALYSIS}" "${LUT}"; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: required input file not found: ${f}" >&2
        exit 1
    fi
done
echo "OK: tracking config, analysis config, and metadata LUT all present."
echo "    tracking_config: ${CONFIG_TRACKING}"
echo "    analysis_config: ${CONFIG_ANALYSIS}"
echo "    LUT:              ${LUT}"

# 5) Output directories must resolve to MedicineBow, never Alcova.
case "${OUTPUT_DIR}" in
    /cluster/alcova/*) echo "FATAL: OUTPUT_DIR points at Alcova (${OUTPUT_DIR}). This job must never write to Alcova. See the STORAGE MODEL note at the top of this file." >&2; exit 1 ;;
esac
case "${BOUTS_DIR}" in
    /cluster/alcova/*) echo "FATAL: BOUTS_DIR points at Alcova (${BOUTS_DIR}). This job must never write to Alcova. See the STORAGE MODEL note at the top of this file." >&2; exit 1 ;;
esac

mkdir -p "${OUTPUT_DIR}" "${BOUTS_DIR}"

echo "------------------------------------------------------------"
echo "Preflight checks passed. Starting pipeline."
echo "------------------------------------------------------------"

# =============================================================================
# STAGE 1 — Tracking (compute-bound; this is nearly all of the ~1h runtime)
# =============================================================================
echo ""
echo "=== [1/4] batch_track_temperatures.py :: START $(date) ==="
python scripts/batch_track_temperatures.py \
    --input-dir "${ALCOVA_SEQ_INPUT_DIR}" \
    --config "${CONFIG_TRACKING}" \
    --output-dir "${OUTPUT_DIR}" \
    ${RECURSIVE_FLAG}
echo "=== [1/4] batch_track_temperatures.py :: END   $(date) ==="

# =============================================================================
# STAGE 2 — Bout detection
# =============================================================================
echo ""
echo "=== [2/4] batch_compute_bouts.py :: START $(date) ==="
python scripts/batch_compute_bouts.py \
    --input-dir "${OUTPUT_DIR}" \
    --config "${CONFIG_ANALYSIS}" \
    --output-dir "${BOUTS_DIR}" \
    --recursive
echo "=== [2/4] batch_compute_bouts.py :: END   $(date) ==="

# =============================================================================
# STAGE 3 — Metadata join
# =============================================================================
echo ""
echo "=== [3/4] join_metadata.py :: START $(date) ==="
python scripts/join_metadata.py \
    --tracking-dir "${OUTPUT_DIR}" \
    --bouts-dir "${BOUTS_DIR}" \
    --metadata "${LUT}" \
    --config "${CONFIG_ANALYSIS}" \
    --output-dir "${BOUTS_DIR}"
echo "=== [3/4] join_metadata.py :: END   $(date) ==="

# =============================================================================
# STAGE 4 — Treatment-effect analysis
# =============================================================================
echo ""
echo "=== [4/4] analyze_treatment_effects.py :: START $(date) ==="
python scripts/analyze_treatment_effects.py \
    --master "${BOUTS_DIR}/master_tracking_with_metadata.csv" \
    --bout-table "${BOUTS_DIR}/bout_table.csv" \
    --config "${CONFIG_ANALYSIS}" \
    --output-dir "${BOUTS_DIR}"
echo "=== [4/4] analyze_treatment_effects.py :: END   $(date) ==="

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "Tracking outputs:      ${OUTPUT_DIR}"
echo "Bout/analysis outputs: ${BOUTS_DIR}"
echo ""
echo "These outputs are on MedicineBow storage ONLY. This job does NOT copy"
echo "them back to Alcova. That is a separate, human-run step — see"
echo "slurm/DEPLOY_MEDICINEBOW.md, step 12, once you've sanity-checked them."
echo "============================================================"

# =============================================================================
# NOT IMPLEMENTED HERE — read only, for context
# =============================================================================
# batch_track_temperatures.py processes .seq files one at a time (confirmed:
# no internal multiprocessing/threading in src/batch.py or elsewhere).
# Tracking is embarrassingly parallel *across files*, so if the dataset grows
# large enough that this single sequential job no longer comfortably fits the
# wall time above, the natural next step is a SLURM JOB ARRAY: one array task
# per session folder (or per .seq file), each running
# `batch_track_temperatures.py --input-dir <one_subfolder>` independently
# against the same Alcova input / MedicineBow output rule above, followed by
# a single small dependent job running stages 2-4 once all array tasks finish
# (`sbatch --dependency=afterok:<array_jobid>`). This is deliberately NOT
# implemented in this script — a single correct sequential job is the right
# first deliverable given the ~1-hour desktop baseline. See
# slurm/DEPLOY_MEDICINEBOW.md for more on this tradeoff.
