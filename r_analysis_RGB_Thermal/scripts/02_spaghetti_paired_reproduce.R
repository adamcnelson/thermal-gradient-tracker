# Section 4.2.1 — paired DCZ/Vehicle spaghetti plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_spaghetti.R")

frames <- load_frames()
bouts <- load_bouts()

build_dcz_vehicle_paired_plots(frames, bouts, output_dir = "output/spaghetti")

cat("Done. Figures written to r_analysis_RGB_Thermal/output/spaghetti/\n")
cat("Compare structure against ../r_analysis/output/spaghetti/paired_dcz_vehicle_*.png ",
    "(legacy single-outcome reference).\n")
