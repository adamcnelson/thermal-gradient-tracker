# ProjectBrief_v4 — SLURM deployment of `thermal-gradient-tracker` on MedicineBow (UW ARCC)

## 0. First: review the existing code

Before writing anything, review the code so far in this self-contained
`thermal-gradient-tracker/` folder — this is where all coding sessions took place to write
the source, and it also contains a handful of `.seq` videos used for local test runs. Read
`README.md` end to end, then walk `scripts/` and `src/` so you understand the actual
pipeline entry points, their command-line arguments, config files (`tracking_config.json`,
`analysis_config.json`), the expected input layout (`croppedSeqFiles/`), and the output
trees (`trackingOutputs/`, `bouts/`).

**Ignore the "manual steps."** Some Python scripts include manual, interactive steps that
served as a quality-control measure — e.g. interactive arena-mask drawing
(`create_arena_mask.py`), manual training-frame annotation with labelme
(`select_training_frames.py` / `train_mouse_detector.py`), and any other step requiring a
person at a screen. **These are no longer in use and must not be part of the SLURM job.**
The batch job must be fully non-interactive (fire-and-forget). Treat `tracking_config.json`
(including its already-set `arena_polygon`) as a fixed input artifact that is committed
alongside the code — the job consumes it; it never regenerates it.

Confirm your understanding of the non-interactive pipeline before writing code. The
compute-bound end-to-end path that must run under SLURM is, at minimum:

1. `batch_track_temperatures.py --input-dir <seq> --config tracking_config.json --output-dir trackingOutputs [--recursive]`
2. `batch_compute_bouts.py --input-dir trackingOutputs --config analysis_config.json --output-dir bouts --recursive`
3. `join_metadata.py --tracking-dir trackingOutputs --bouts-dir bouts --metadata metadata/LUT_CLEAN_July6.csv --config analysis_config.json --output-dir bouts`
4. `analyze_treatment_effects.py --master bouts/master_tracking_with_metadata.csv --bout-table bouts/bout_table.csv --config analysis_config.json --output-dir bouts`

If reviewing the scripts reveals the real entry points or arguments differ from the README,
**trust the code**, and note any discrepancy in the manual.

---

## 1. Background & motivation

`thermal-gradient-tracker` is a Python pipeline (built here with Claude Code) that analyzes
FLIR `.seq` thermography videos from a rodent thermal-gradient assay: it tracks the mouse,
extracts surface/floor temperatures, detects stationary bouts, joins experimental metadata,
and fits treatment-effect models.

The pipeline is verified working on a Windows PC desktop on the UW campus. On that machine a
full-dataset run takes **over an hour**, which is why we are moving to SLURM: to run the
full dataset on the University of Wyoming ARCC **MedicineBow** cluster (Alcova), where the
code has been transferred. I can already log into MedicineBow OnDemand
(`https://medicinebow.arcc.uwyo.edu`) and have shell access there.

## 2. Objectives

Produce two deliverables, committed into the repo:

1. **`run_thermal_gradient.sbatch`** — a SLURM batch script that runs the full
   non-interactive pipeline end to end on MedicineBow.
2. **`DEPLOY_MEDICINEBOW.md`** — a clear, concise instruction manual for a CS-oriented team
   member to deploy that script from scratch on MedicineBow.

## 3. Target environment (MedicineBow / ARCC) — constraints to honor

Design both deliverables around these ARCC facts (from ARCC HPC policy and the "Getting
Started" wiki). Call these out explicitly in the manual.

- **Scheduler:** Slurm. Batch jobs are submitted with `sbatch <script>` from a **login
  node**. Submitting the job from the login node is allowed; the actual computation runs on
  a compute node.
- **Never compute on the login node.** No compute-intensive or long-running (>10 min) work
  on login nodes — that's the entire reason for the batch job. The manual must state this.
- **Required directives:** every job must specify `--account=<project>` and `--time`.
  Leave `--account` as a clearly-marked placeholder (`<YOUR_ARCC_PROJECT>`) since it is
  project-specific.
- **Memory must be explicit.** `--mem=0` is disallowed. Request an explicit `--mem` (e.g.
  `--mem=16G`) with a comment on how to raise it if an out-of-memory / `oom-kill` occurs.
- **QoS / wall time.** Default (Normal) queue allows 3-day wall time; the **Fast** queue is
  higher priority but caps at 12 h. A run that takes ~1 h on a single desktop should fit
  comfortably in **Fast** once parallelized or even serial — set a conservative `--time`
  (e.g. `02:00:00`) and comment that shorter wall times get scheduled sooner. Note in the
  manual how to switch QoS/partition if the job is larger than expected.
- **CPU/parallelism.** Tracking is embarrassingly parallel across `.seq` files. Decide, from
  reading the code, whether `batch_track_temperatures.py` already parallelizes internally
  (e.g. a workers flag) or is serial:
  - If it exposes internal multiprocessing, request `--cpus-per-task=N` and pass N through.
  - If it's serial per file, still request a modest `--cpus-per-task` (e.g. 8) and add a
    clearly-commented **optional** section showing how this could later be converted to a
    Slurm **job array** (one array task per session folder or per `.seq` file). Do not
    over-engineer the array now — a single correct serial/multiprocess job is the priority;
    describe the array as a future optimization.
- **Software / environment.** Users have no `sudo`. Two viable paths — pick the one that
  matches what you find, and document it:
  - Check for a Python module via `module spider python` (LMOD). Prefer building a project
    virtualenv (`python -m venv .venv && pip install -r requirements.txt`) on top of a
    loaded Python module, created **once** in the project space, then activated inside the
    sbatch script.
  - **exiftool is a hard dependency** and is *not* a pip package — without it, temperatures
    are computed by an inaccurate linear fallback (see README). It must be available on the
    compute node. In the manual, instruct the deployer to check `module spider exiftool` /
    `which exiftool`, and if absent, to request it from ARCC (`arcc-help@uwyo.edu`) or
    install a local copy — and to **verify `exiftool` resolves on the compute node**, not
    just the login node. Add a preflight check in the sbatch script that fails loudly if
    exiftool is missing rather than silently producing wrong temperatures.
- **Storage & data location.** Home is small (50 GB). Real datasets belong in `/project`
  (persistent, per-project) or `/gscratch` (large, but **purged after 90 days of
  inactivity** — never the only copy). The sbatch script should not hard-code my desktop
  paths; input `.seq` directory, config paths, and output directory should be variables at
  the top of the script (or CLI args) that the deployer sets. Mirror the README's
  philosophy: paths are passed in, code is never edited.
- **Notifications & logging.** Include `--job-name`, `--mail-type=ALL`, and a
  `--mail-user=<placeholder>` line, plus Slurm `--output`/`--error` log paths (e.g.
  `logs/%x_%j.out`). Have the script echo key info (node, start time, resolved paths, tool
  versions, `exiftool` path) to the log for debuggability.

## 4. `run_thermal_gradient.sbatch` — requirements

- Standard `#!/bin/bash` shebang and a header block of `#SBATCH` directives covering:
  `--account` (placeholder), `--job-name`, `--time`, `--partition`/QoS as appropriate,
  `--nodes=1`, `--cpus-per-task`, `--mem` (explicit), `--mail-type`, `--mail-user`
  (placeholder), `--output`, `--error`.
- A clearly-delimited **user-config block** near the top: `SEQ_INPUT_DIR`, `OUTPUT_DIR`,
  `CONFIG_TRACKING`, `CONFIG_ANALYSIS`, `LUT`, `VENV_PATH`. Every path the deployer must set
  lives here and nowhere else.
- Environment setup: load the Python module (or whatever you determined), activate the venv,
  fail fast (`set -euo pipefail`) so a broken step stops the job instead of cascading.
- **Preflight checks** that abort with a clear message: venv activates, `python -c "import ..."`
  smoke test if cheap, `exiftool` present, input dir exists and contains `.seq` files, LUT
  exists.
- Run the four non-interactive stages in order (section 0), each echoing a start/end banner
  and timestamp to the log. **No interactive/manual steps.**
- Exit non-zero on any failure so `--mail-type=ALL` reports it accurately.
- Heavy inline comments — this doubles as a teaching artifact for the team.

## 5. `DEPLOY_MEDICINEBOW.md` — requirements

Write for a CS-literate teammate who has **not** used this pipeline or ARCC before. Concise,
step-ordered, copy-pasteable commands. Cover:

1. **Prerequisites** — being added to an ARCC project (so they have an `--account`), and
   confirming MedicineBow access (OnDemand at `https://medicinebow.arcc.uwyo.edu`, or SSH
   with key + certificate).
2. **Getting the code onto MedicineBow** — clone/transfer the repo into `/project/<...>`;
   note the small home quota; recommend `/project` for code + `.seq` data. Mention OnDemand
   file upload vs. Globus for the large dataset.
3. **One-time environment setup** — load Python module, create venv, `pip install -r
   requirements.txt`, resolve `exiftool`, place `metadata/LUT_CLEAN_July6.csv`, and confirm
   `tracking_config.json` (with its committed `arena_polygon`) is present. Explicitly state
   the manual/interactive steps are **not** run on the cluster.
4. **Configure the sbatch script** — which variables to edit (the user-config block),
   `--account`, `--mail-user`, and where the `.seq` input lives.
5. **Submit & monitor** — `sbatch run_thermal_gradient.sbatch`; check status with `squeue
   --me`; find logs under `logs/`; cancel with `scancel <jobid>`.
6. **Retrieve results** — where outputs land (`trackingOutputs/`, `bouts/`), how to pull
   them back off the cluster.
7. **Troubleshooting** — oom-kill → raise `--mem`; job pending too long → shorter `--time`
   or different QoS; wrong temperatures → exiftool not found on the compute node; join
   misses → LUT filename pattern (point to README's troubleshooting section). Include the
   ARCC help contact (`arcc-help@uwyo.edu`).
8. A short **"what this pipeline does"** paragraph and a pointer to the main `README.md` for
   scientific/config detail (don't duplicate it).

## 6. Guardrails

- Do **not** modify pipeline source in `src/` or `scripts/` to make SLURM work. If a genuine
  blocker requires a code change (e.g. a script hard-codes an interactive call in the
  non-interactive path), stop and flag it to me with the specific file/line rather than
  editing silently.
- Do not invent ARCC-specific values you can't verify (exact partition names, module
  versions, account name). Use clearly-marked placeholders and tell the deployer to confirm
  them with `module spider`, the MedicineBow hardware summary, or ARCC support.
- Keep both deliverables self-contained in the repo root and cross-referenced.

## 7. Deliverables checklist

- [ ] Code reviewed; non-interactive path confirmed; manual/QC steps excluded.
- [ ] `run_thermal_gradient.sbatch` at repo root, heavily commented, paths parameterized,
      `--mem` explicit, preflight checks incl. exiftool, fail-fast.
- [ ] `DEPLOY_MEDICINEBOW.md` at repo root, step-ordered, copy-pasteable, placeholders
      marked.
- [ ] Any discrepancies between README and actual code, or any required source change,
      flagged to me explicitly.
