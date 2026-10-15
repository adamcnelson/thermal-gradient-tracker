# Which SLURM results the analysis reads, and where its figures go.
#
# Each data version names every input file explicitly; DATA_VERSION (env var) picks one, and all
# output goes to output/<DATA_VERSION>/ (+ <window>s_post_entry/), so results from different
# SLURM runs never overwrite each other. scripts/run_all.sh records the inputs' sha256 in
# output/<DATA_VERSION>/INPUTS.sha256 and refuses to rerun into a folder whose inputs changed.
#
#   retrack_2026-10     tracking: SLURM_RESULTS retrack_2026-10 (65 inherited + Test_3/4/7 lanes),
#                       recover_2026-10 (10 lanes), recover_lanes_2026-10 (4 lanes); Stage 7: the
#                       Test_3/4/7, stage7_inherited, recover and recover_lanes runs; joined into
#                       landmark_outputs/. The analysis up to 2026-10-14.
#   retrack_v2_2026-10  slurm/run_retrack_v2.sbatch (job 24611205, 2026-10-15): all 85 lanes
#                       re-tracked with tracker v2 + Stage 7 in one run; joined into
#                       thermalFeatures/retrack_v2_2026-10/landmark_join/.
# Paths are relative to r_analysis_RGB_Thermal/.

DATA_VERSIONS <- list(
  "retrack_2026-10" = list(
    landmark_dir = "../landmark_outputs",
    stage7_configs = c(
      "../stage7_sessions_config.json",
      "../thermalFeatures/stage7_inherited/stage7_sessions_config_inherited.json",
      "../thermalFeatures/recover_2026-10/stage7/stage7_sessions_config_recover.json",
      "../thermalFeatures/recover_lanes_2026-10/stage7/stage7_sessions_config_recover_lanes.json"),
    # Same files, same order as that version's join (--velocity-from): a lane in several is taken from the last.
    tracking_csvs = c(
      "../thermalFeatures/retrack_2026-10/retrack/master_tracking_with_metadata.csv",
      "../thermalFeatures/recover_2026-10/retrack/retrack/master_tracking_with_metadata.csv",
      "../thermalFeatures/recover_lanes_2026-10/retrack/retrack/master_tracking_with_metadata.csv")
  ),
  "retrack_v2_2026-10" = list(
    landmark_dir = "../thermalFeatures/retrack_v2_2026-10/landmark_join",
    stage7_configs = "../thermalFeatures/retrack_v2_2026-10/stage7/stage7_sessions_config_v2.json",
    tracking_csvs = "../thermalFeatures/retrack_v2_2026-10/retrack/master_tracking_with_metadata.csv"
  )
)

DATA_VERSION <- Sys.getenv("DATA_VERSION", "retrack_2026-10")
if (!DATA_VERSION %in% names(DATA_VERSIONS)) {
  stop("Unknown DATA_VERSION '", DATA_VERSION, "'; known: ", paste(names(DATA_VERSIONS), collapse = ", "))
}
DATA_INPUTS <- DATA_VERSIONS[[DATA_VERSION]]
LUT_CSV <- file.path("..", "metadata", "LUT_CLEAN_July6.csv")

# Every file this version reads, for the INPUTS.sha256 manifest.
data_version_files <- function(inputs = DATA_INPUTS) {
  c(file.path(inputs$landmark_dir, c("master_landmarks_with_metadata_bout.csv",
                                     "master_landmarks_with_metadata_frame.csv")),
    inputs$stage7_configs, inputs$tracking_csvs, LUT_CSV)
}
