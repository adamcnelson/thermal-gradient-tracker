# Deploying `thermal-gradient-tracker` on MedicineBow (UW ARCC)

This is a step-by-step guide for a teammate deploying the SLURM batch job for
this pipeline on the University of Wyoming ARCC **MedicineBow** cluster for
the first time. It assumes no prior Slurm or ARCC experience but does assume
comfort with the Linux command line, SSH, and `git`.

## What this pipeline does

`thermal-gradient-tracker` analyzes FLIR `.seq` thermography videos from a
rodent thermal-gradient assay: it tracks the mouse, extracts surface/floor
temperatures, detects stationary rest bouts, joins experimental metadata, and
fits treatment-effect models. See **`README.md`** at the repo root for full
scientific and configuration detail (config parameters, output file formats,
troubleshooting bout thresholds, etc.) — this manual does not duplicate that;
it only covers getting the existing, already-configured pipeline running as
a batch job on MedicineBow.

The batch job (`slurm/run_thermal_gradient.sbatch`) runs four
already-non-interactive stages in sequence:

1. `batch_track_temperatures.py` — tracking + temperature extraction (the vast
   majority of the ~1 hour runtime)
2. `batch_compute_bouts.py` — stationary bout detection
3. `join_metadata.py` — join to experimental metadata
4. `analyze_treatment_effects.py` — treatment-effect models and plots

**Important:** this repo also contains manual/interactive QC scripts
(`create_arena_mask.py`, `select_training_frames.py`, `train_mouse_detector.py`)
used earlier in development to draw the arena mask and tune segmentation.
**These are retired — they are not run on the cluster and have no place in
the batch job.** `tracking_config.json` — including its already-drawn
`arena_polygon` — is a finalized, fixed input; the batch job only reads it.

---

## 1. The storage model — read this before doing anything else

The data and code for this project live in two different places with two
different permission systems, and getting the direction of data flow wrong
is the most likely way to break this pipeline on the cluster:

- **Alcova** (`/cluster/alcova/bedfordlab`) — the bedfordlab storage project.
  Uses **Windows-style (ACL) permissions**. This is where the `.seq` input
  dataset lives, and where finished results ultimately belong.
- **MedicineBow** (`/project`, `/gscratch`) — the compute cluster. Uses
  **POSIX permissions**. This is where the code, the Python environment, and
  the job itself live and run.

**Reading** from Alcova on MedicineBow works reliably. **Writing** from
MedicineBow to Alcova does not — permission translation across the ACL/POSIX
boundary can silently fail. Because of this asymmetry, the workflow is:

1. Code + virtualenv live on MedicineBow `/project` (cloned there directly,
   not synced from Alcova).
2. The job **reads** `.seq` inputs from Alcova (`/cluster/alcova/bedfordlab/...`).
3. The job **writes all outputs** to MedicineBow storage (`/project` or
   `/gscratch`) — **never** to an Alcova path.
4. Once outputs are sanity-checked, a **person** copies them back to Alcova
   as a separate step, after the job has finished — see step 12 below.

**The job never writes to Alcova, under any circumstance.** If you're tempted
to point `--output-dir` at an Alcova path "just this once" to save a copy
step — don't; that write can fail or silently corrupt permissions on the
Alcova side. The copy-back step exists precisely so this never has to happen
inside the job.

---

## 2. Prerequisites

- **ARCC project / account.** You need to be added to an ARCC project to get
  a Slurm `--account` value. This is the project named in the welcome email
  from `arcc-admin@uwyo.edu`, or run `my_accounts` on a login node. If you
  don't have one, ask your PI/lab lead, or request one via ARCC support
  (`arcc-help@uwyo.edu`).
- **MedicineBow access**, either:
  - **OnDemand** (easiest to start with): log into
    `https://medicinebow.arcc.uwyo.edu` with your UWyo credentials. This
    gives you a browser-based shell, file browser, and Jupyter/Desktop apps
    without needing local SSH key setup.
  - **SSH**, if you prefer a terminal: this requires an SSH key plus an ARCC
    login certificate (short-lived, re-issued periodically) — see the ARCC
    "Getting Started" wiki for the current SSH setup steps, or ask ARCC
    support if you don't already have this configured.
- **Alcova access.** Confirm separately that your account can read
  `/cluster/alcova/bedfordlab` from a MedicineBow login node (`ls
  /cluster/alcova/bedfordlab/...`) — Alcova access is granted independently
  of MedicineBow access since it's a different permission system (ACL, not
  POSIX). If it's not visible, ask Adam or ARCC support to confirm your ACL
  grant.
- Confirm you can open a shell on a **login node** (`ssh` prompt or OnDemand
  "Clusters > MedicineBow Shell Access") before continuing.

## 3. Getting the code onto MedicineBow

Your home directory quota is small (**50 GB**) — do not put the repo there.
Use `/project/<your_project>/` (persistent, not purged). The code lives on
MedicineBow permanently; it is not re-synced from Alcova each time.

```bash
# On a MedicineBow login node:
mkdir -p /project/<your_project>/
cd /project/<your_project>/
git clone <YOUR_REPO_URL> thermal-gradient-tracker
cd thermal-gradient-tracker
```

here's what I did (July 28 2026)
```
mkdir -p /project/huddlevidmicro/anelso74/
cd /project/huddlevidmicro/anelso74/
git clone https://github.com/adamcnelson/thermal-gradient-tracker.git
cd thermal-gradient-tracker
```

Since we're actively iterating on the code, pulling updates is just:

```bash
cd /project/<your_project>/thermal-gradient-tracker
git pull
```

This only ever touches the MedicineBow copy of the code — it has no effect
on anything stored on Alcova.

## 4. Confirming access to the data on Alcova

The `.seq` input dataset lives on Alcova, not MedicineBow:

```
/cluster/alcova/bedfordlab/<path to the cropped .seq dataset>
```

This differs from Adam's iMac path (`../croppedSeqFiles/...` in the README)
— confirm the real Alcova path with Adam if you don't already have it. No
source code needs editing for this: the path is set once, in the
`ALCOVA_SEQ_INPUT_DIR` variable near the top of `slurm/run_thermal_gradient.sbatch`
(see step 7).

Sanity-check you can read it from a MedicineBow login node:

```bash
ls /cluster/alcova/bedfordlab/<path to the cropped .seq dataset> | head
```

**Optional — staging a local copy for heavy iterative runs.** If you expect
to re-run the pipeline many times while tuning bout parameters, reading
directly from Alcova on every run works but adds cross-service I/O each time.
You can instead stage a one-time copy of the `.seq` files onto MedicineBow
storage (`/project` or `/gscratch`) and point `ALCOVA_SEQ_INPUT_DIR` at that
copy instead — this is a normal MedicineBow-to-MedicineBow copy once the data
has been read from Alcova, so it's fully decoupled from Alcova after that.
Use `rsync` or Globus for the initial copy-in; ask ARCC support
(`arcc-help@uwyo.edu`) if you're not sure which is appropriate for the
dataset size.

## 5. One-time environment setup

This is done **once** per project space, from a login node (it's not
compute-intensive, so it's fine on the login node — just don't run the
*pipeline itself* there).

```bash
cd /project/<your_project>/thermal-gradient-tracker

# Find the available Python module (name/version will be MedicineBow-specific —
# confirm and note whatever it reports; the exact string wasn't verified from
# outside the cluster and is a placeholder in the sbatch script):
module spider python

# Load whatever module `module spider` recommends, e.g.:
module load python/3.11     # <- replace with the actual module name/version

# Create the project virtualenv ONCE, in the project space:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Resolve `exiftool` (hard dependency)

The pipeline uses `exiftool` to read FLIR Planck calibration constants for
accurate temperature conversion. **Without it, temperatures silently fall
back to an inaccurate linear approximation** — no crash, just wrong numbers —
so this must be confirmed working before a real run. This is important
enough that `run_thermal_gradient.sbatch` also checks for it itself and
**fails the job immediately** if it's missing, rather than letting a run
complete with wrong temperatures.

```bash
module spider exiftool     # check if ARCC provides it as a module
which exiftool              # or check if it's already on PATH
```

- If a module exists, note it — you'll need to `module load` it in the
  sbatch script alongside the Python module (there's a commented-out line
  ready for this).
- If neither works, request it from ARCC support (`arcc-help@uwyo.edu`), or
  install a self-contained copy under your project space and add it to
  `PATH` (exiftool is a single Perl script + libraries; see
  [exiftool.org](https://exiftool.org) for the standalone tarball).

**Critical: verify `exiftool` resolves on a *compute* node, not just the
login node.** Loaded modules and `PATH` can differ between them. Run a
one-off interactive job to check:

```bash
srun --account=<PROJECT_NAME> --time=00:05:00 --mem=2G --pty bash
# once you have a compute-node shell:
module load <exiftool module, if applicable>
which exiftool && exiftool -ver
exit
```

If this fails on the compute node even though it worked on the login node,
you likely need to add the corresponding `module load` line inside
`slurm/run_thermal_gradient.sbatch` itself (compute nodes start with a fresh
environment — they do not inherit your login shell's loaded modules).

## 6. Placing config and metadata

These are fixed, already-finalized inputs — they're read by the job, never
regenerated by it:

```bash
# metadata/LUT_CLEAN_July6.csv should already be in the repo if it was
# committed; if not, copy it in:
cp /path/to/LUT_CLEAN_July6.csv metadata/

# Confirm the fixed config artifacts are present and already configured:
cat tracking_config.json    # should already have a non-null "arena_polygon"
cat analysis_config.json
```

**Do not run** `create_arena_mask.py`, `select_training_frames.py`, or
`train_mouse_detector.py` on the cluster — these are retired, interactive
scripts that require a person at a screen, and are not part of the batch job.
If `arena_polygon` is missing or you believe the geometry has changed, that
has to be redone locally (e.g. on the original desktop) and the updated
`tracking_config.json` re-committed/re-copied — flag this to Adam rather than
trying to regenerate it on MedicineBow.

## 7. Editing the sbatch script

Open `slurm/run_thermal_gradient.sbatch` and edit **only** the placeholders
listed below — every other line is either fixed pipeline logic or a default
that already does the right thing.

**A. `#SBATCH` directives near the top:**

| Placeholder | Set to |
|---|---|
| `--account=<PROJECT_NAME>` | Your ARCC project/account name (step 2) |
| `--mail-user=<USER_EMAIL>` | Your email |
| `--qos=fast` / `--partition=...` | Confirm the "Fast" QoS/partition name with `sinfo` or ARCC support; edit if it differs |

**B. The `USER-CONFIG BLOCK` (plain bash variables, further down):**

| Variable | Set to |
|---|---|
| `REPO_DIR` | Absolute path to this repo on MedicineBow, e.g. `/project/<your_project>/thermal-gradient-tracker` |
| `ALCOVA_SEQ_INPUT_DIR` | Absolute Alcova path to your `.seq` input dataset (step 4) — or the staged MedicineBow copy, if you set that up |
| `CONFIG_TRACKING`, `CONFIG_ANALYSIS`, `LUT` | Usually left as-is (they default to files inside `REPO_DIR`) |
| `VENV_PATH` | Usually left as-is (`${REPO_DIR}/.venv`, created in step 5) |
| `OUTPUT_DIR`, `BOUTS_DIR` | **Leave pointed at MedicineBow** (default: inside `REPO_DIR`, i.e. `/project/...`). Do **not** repoint these at `/cluster/alcova/...` — see the storage-model note in section 1 and the matching comment block in the script itself. The script will refuse to run (fail its preflight check) if either resolves to an Alcova path. |
| `RECURSIVE_FLAG` | `--recursive` if `ALCOVA_SEQ_INPUT_DIR` has session subfolders, or `""` if it's one flat folder of `.seq` files |

Also confirm the `module load PYTHON_MODULE_PLACEHOLDER` line (and, if
needed, uncomment and fill in the `exiftool` module load line) matches what
you found in step 5.

## 8. Smoke test first

Before committing the full dataset, confirm the environment works end-to-end
against the handful of `.seq` test videos included in the repo (or run
interactively via `salloc`/a Debug/short session instead of `sbatch` for
faster turnaround). Point `ALCOVA_SEQ_INPUT_DIR` at just the test-video
subfolder for this run, then re-point it at the full dataset once it passes.

Results from a small subset are **expected** to show up labeled
`PRELIMINARY — insufficient sample for inference` in
`bouts/analysis_README.txt` and related outputs — that's the analysis stage
correctly detecting too little data for statistical inference, not an error.
The same commands run unchanged on the full dataset.

## 9. Submitting the full job

```bash
cd /project/<your_project>/thermal-gradient-tracker
mkdir -p logs                       # Slurm needs this to exist before the job starts
sbatch slurm/run_thermal_gradient.sbatch
```

This prints a job ID, e.g. `Submitted batch job 123456`. **Always submit from
a login node with `sbatch`** — never run the pipeline scripts directly on a
login node; they are compute-intensive and long-running, well over ARCC's
~10-minute login-node limit.

## 10. Monitoring

```bash
squeue --me                                 # see your queued/running jobs
squeue -j <jobid>                           # status of a specific job
tail -f logs/thermal_gradient_<jobid>.out   # follow the log live
scancel <jobid>                             # cancel a job
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode   # post-mortem
```

You'll also get an email at each job state transition (`--mail-type=ALL`):
job start, end (success or failure), and requeue.

## 11. Sanity-checking outputs on MedicineBow

Outputs land under, relative to `REPO_DIR` (i.e. on MedicineBow, not Alcova):

- `trackingOutputs/` — per-file tracking CSVs and QC images/plots (stage 1)
- `bouts/` — bout tables, metadata-joined master table, and
  `analysis_tables/` / `analysis_plots/` / `analysis_README.txt` (stages 2–4)

Before trusting a run, check:

- **`bouts/metadata_join_report.csv`** — every row should show `matched_exact`
  or `matched_token`; there should be **no `unmatched` rows** (flag any to
  Adam).
- **QC overlay images** in `trackingOutputs/qc_images/` — spot-check a few;
  cyan circle on the mouse, green circle on empty floor.
- **Bout diagnostic plots** in `bouts/qc_plots/` — the red dashed threshold
  lines should look sane against the actual centroid/velocity traces.

See `README.md`'s "Output file reference" and "Troubleshooting" sections for
the full column/file breakdown and what to do about join misses or unusual
bout counts.

## 12. Copying results back to Alcova

**This is a separate, human-run step performed only after you've sanity-checked
the outputs in step 11 — it is never done automatically inside the job.**

Copy the finished outputs from MedicineBow into a **run-named folder** on
Alcova, e.g. `results_<runname>_<date>/`, so results from different runs
don't overwrite each other. Outputs are small (CSVs + PNGs), so this is cheap
and easy to verify by hand afterward.

**Resolved with ARCC (2026-07-28): use Globus, not an ad hoc `cp`/`rsync`.**
ARCC's stated reasoning is the same POSIX↔ACL permission asymmetry described
in section 1 — a plain `cp`/`rsync` from MedicineBow can silently mishandle
Alcova's ACL permissions, where Globus handles the translation correctly.
ARCC's recommendation, verbatim: *"try a single run or a small subset with
whatever is easier for you first and then test[ing] before going forward."*

1. Log into https://app.globus.org with your UWyo credentials.
2. Find the ARCC-hosted collection that exposes both MedicineBow
   (`/cluster/medbow/...`) and Alcova (`/cluster/alcova/...`) storage. The
   exact collection name has changed across ARCC's docs (seen referred to as
   both "ARCC Medicinebow" and "ARCC Teton" in different places) — don't
   hardcode one here; confirm the current name from ARCC's own docs:
   - [Globus — ARCC Wiki](https://arccwiki.atlassian.net/wiki/spaces/DOCUMENTAT/pages/1757446145)
   - [Globus Web Interface — arccwiki](https://arccwiki.uwyo.edu/index.php/Globus_Web_Interface)
   - [Data Moving and Access — ARCC Wiki](https://arccwiki.atlassian.net/wiki/spaces/DOCUMENTAT/pages/1559592966)
3. Source: `/cluster/medbow/<your_project>/thermal-gradient-tracker/{trackingOutputs,bouts}`
4. Destination: a new `results_<runname>_<date>/` folder under
   `/cluster/alcova/bedfordlab/ThermalGradient/...`
5. **Test small first, per ARCC's advice**: transfer a small subset (e.g.
   just `bouts/metadata_join_report.csv` and a couple of QC images) and
   confirm on the Alcova side that (a) the transfer succeeded and (b)
   everyone in the lab who should be able to read the results actually can —
   ACL permissions need to carry over correctly, which is exactly what a
   plain `cp`/`rsync` risks getting wrong. Only transfer the full result set
   once that's confirmed.

Until you've done that small-subset test, treat the MedicineBow copy as the
working copy rather than relying on Alcova as the durable copy of a given
run's results.

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Job fails almost immediately, log shows `FATAL: exiftool not found on compute node` | `exiftool` isn't on `PATH` on the *compute* node even if it works on login. Add/fix a `module load` line for it in the sbatch script, or install a project-local copy — see step 5. |
| Job killed, log/`sacct` shows `OUT_OF_MEMORY` / `oom-kill` | Raise `--mem` in the `#SBATCH` header (e.g. `32G` → `64G`) and resubmit. |
| Job stays `PENDING` in `squeue` for a long time | The requested `--time`/QoS may be lower priority or resources are busy. Confirm you're targeting the "Fast" QoS, not "Normal". `squeue -j <jobid> --start` shows the estimated start time; `sprio -j <jobid>` shows priority factors. |
| Temperatures look linear/off even though the job succeeded | `exiftool` passed the preflight check but Planck constants weren't found for some individual files. Check the per-file log output from stage 1 for `exiftool not found or Planck constants missing` warnings. |
| `metadata_join_report.csv` has `unmatched` rows | Not a Slurm issue — a metadata/filename mismatch. See `README.md`'s "Join misses" troubleshooting section (filename pattern vs. LUT `Video_name_SEQ` column). |
| Permission error / failure writing to an Alcova path | **Don't write to Alcova from the job — this should never happen** given the preflight check in the sbatch script. If you see this, something has been misconfigured (e.g. `OUTPUT_DIR`/`BOUTS_DIR` was pointed at `/cluster/alcova/...`). Fix the path in the USER-CONFIG block back to a MedicineBow path; use the copy-back step (12) instead of writing to Alcova directly. |
| Anything else / cluster-specific errors (`module` not found, QoS/partition rejected, account invalid) | Contact ARCC support: **`arcc-help@uwyo.edu`** |

---

For pipeline internals, config parameter meaning, bout-threshold tuning, and
full output schemas, see **`README.md`** at the repo root.
