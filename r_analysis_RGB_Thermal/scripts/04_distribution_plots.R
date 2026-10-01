# Section 4.3 — distribution / histogram plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_distributions.R")

frames <- load_frames()
build_distribution_plots(frames, output_dir = "output/distributions")

cat("Done. Figures written to r_analysis_RGB_Thermal/output/distributions/\n")
cat("NOTE: velocity_smooth_px_s distributions deferred -- needs a separate\n")
cat("per-timestamp join against trackingOutputs/*_tracking_every10frames.csv\n")
cat("(see R/plot_distributions.R header comment).\n")
