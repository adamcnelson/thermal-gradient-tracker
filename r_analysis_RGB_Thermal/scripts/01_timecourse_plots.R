# Section 4.1 — time-course plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_timecourse.R")

frames <- load_frames()
bouts <- load_bouts()

build_timecourse_plots(frames, bouts, output_dir = "output/timecourse")

cat("Done. Figures written to r_analysis_RGB_Thermal/output/timecourse/\n")
