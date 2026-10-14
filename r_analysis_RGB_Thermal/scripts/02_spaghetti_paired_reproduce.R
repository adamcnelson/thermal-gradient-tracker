# Section 4.2.1 — paired DCZ/Vehicle spaghetti plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/dcz_vehicle_units.R")
source("R/tracking_activity.R")
source("R/plot_spaghetti.R")

frames <- load_frames()
bouts <- load_bouts()
activity <- load_activity(frames)

build_dcz_vehicle_paired_plots(frames, bouts, output_dir = file.path(output_root(), "spaghetti"), activity_lanes = activity$lanes)

cat("Done. Figures written to", file.path(output_root(), "spaghetti"), "\n")
cat("Compare structure against ../r_analysis/output/spaghetti/paired_dcz_vehicle_*.png ",
    "(legacy single-outcome reference).\n")
