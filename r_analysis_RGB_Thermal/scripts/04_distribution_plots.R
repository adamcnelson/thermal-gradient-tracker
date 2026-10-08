# Section 4.3 — distribution / histogram plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_distributions.R")

frames <- load_frames()
build_distribution_plots(frames, output_dir = "output/distributions")

cat("Done. Figures written to r_analysis_RGB_Thermal/output/distributions/\n")
cat("Velocity distributions need velocity_smooth_px_s in the frame table: run\n",
    "../scripts/join_landmark_metadata.py with --velocity-from (see R/plot_distributions.R).\n")
