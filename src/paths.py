"""
Single source of truth for project directories.

All paths are computed relative to the repository root (this file's location
on disk), not the current working directory, so the automated scripts behave
identically whether run locally on Adam's iMac or from a SLURM job on
MedicineBow. The only path a user ever supplies manually is the
croppedSeqFiles input location (via --input / --input-dir) — every directory
below is created automatically on first use via ensure_dir().
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── config files ──────────────────────────────────────────────────────────
DEFAULT_TRACKING_CONFIG = PROJECT_ROOT / "tracking_config.json"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "analysis_config.json"

# ── metadata ──────────────────────────────────────────────────────────────
METADATA_DIR = PROJECT_ROOT / "metadata"
DEFAULT_METADATA_LUT = METADATA_DIR / "LUT_CLEAN_July6.csv"

# ── preview / mask (retired manual steps use mask_preview directly) ───────
PREVIEW_TRACKING_DIR = PROJECT_ROOT / "preview_tracking"

# ── tracking outputs ────────────────────────────────────────────────────────
TRACKING_OUTPUTS_DIR = PROJECT_ROOT / "trackingOutputs"
TRACKING_QC_IMAGES_DIR = TRACKING_OUTPUTS_DIR / "qc_images"
TRACKING_QC_PLOTS_DIR = TRACKING_OUTPUTS_DIR / "qc_plots"
TRACKING_QC_SUMMARIES_DIR = TRACKING_OUTPUTS_DIR / "qc_summaries"
TRACKING_BOUT_EXAMPLES_DIR = TRACKING_OUTPUTS_DIR / "bout_examples"

# ── bout detection + analysis outputs ───────────────────────────────────────
BOUTS_DIR = PROJECT_ROOT / "bouts"
BOUTS_QC_PLOTS_DIR = BOUTS_DIR / "qc_plots"
BOUTS_REFERENCE_DIR = BOUTS_DIR / "reference"
BOUTS_ANALYSIS_TABLES_DIR = BOUTS_DIR / "analysis_tables"
BOUTS_ANALYSIS_PLOTS_DIR = BOUTS_DIR / "analysis_plots"

# ── v7 landmarks: per-session RGB<->thermal homography calibration ─────────
# Small, hand-clicked correspondence files (see scripts/calibrate_homography.py)
# — unlike the generated-output dirs above, these are NOT gitignored: a human
# spent effort producing them and they aren't cheaply regenerable.
HOMOGRAPHY_CALIBRATION_DIR = PROJECT_ROOT / "homography_calibration"

# ── v7 landmarks: Stage 5-7 outputs (RGB tracks, bout/frame tables, QC) ────
LANDMARK_OUTPUTS_DIR = PROJECT_ROOT / "landmark_outputs"


def ensure_dir(path: Path) -> Path:
    """mkdir -p and return the path, for use at the point a directory is written to."""
    path.mkdir(parents=True, exist_ok=True)
    return path
