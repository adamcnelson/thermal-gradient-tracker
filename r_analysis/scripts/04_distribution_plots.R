# Section 3.3 — distribution / histogram plots. Run from r_analysis/.
source("R/data.R")
source("R/plot_distributions.R")

master <- load_master()
build_distribution_plots(master, output_dir = "output/distributions")

cat("Done. Figures written to r_analysis/output/distributions/\n")
