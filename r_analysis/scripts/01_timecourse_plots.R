# Section 3.1 — time-course plots. Run from r_analysis/.
source("R/data.R")
source("R/plot_timecourse.R")

master <- load_master()
build_timecourse_plots(master, output_dir = "output/timecourse")

cat("Done. Figures written to r_analysis/output/timecourse/\n")
