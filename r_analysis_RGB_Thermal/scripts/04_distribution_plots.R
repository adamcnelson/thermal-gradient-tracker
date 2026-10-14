# Section 4.3 — distribution / histogram plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/tracking_activity.R")
source("R/plot_distributions.R")

frames <- load_frames()
build_distribution_plots(frames, output_dir = file.path(output_root(), "distributions"),
                         tracking_frames = load_activity(frames)$frames)

cat("Done. Figures written to", file.path(output_root(), "distributions"), "\n")
cat("Velocity distributions need velocity_smooth_px_s in the frame table: run\n",
    "../scripts/join_landmark_metadata.py with --velocity-from (see R/plot_distributions.R).\n")
