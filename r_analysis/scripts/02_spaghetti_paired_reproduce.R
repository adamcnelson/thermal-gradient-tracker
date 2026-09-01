# Section 3.2.1 — reproduce the 4 existing Python paired DCZ/Vehicle plots.
# Run from r_analysis/.
source("R/data.R")
source("R/plot_spaghetti.R")

master <- load_master()
build_dcz_vehicle_paired_plots(master, output_dir = "output/spaghetti")

cat("Done. Figures written to r_analysis/output/spaghetti/\n")
cat("Compare against ../bouts/analysis_plots/paired_dcz_vehicle_*.png\n")
