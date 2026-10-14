# Section 4.1 — time-course plots. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/tracking_activity.R")
source("R/plot_timecourse.R")

bouts <- load_bouts()
frames <- load_frames()

build_timecourse_plots(frames, bouts, output_dir = file.path(output_root(), "timecourse"),
                       activity_bins = load_activity(frames)$bins)

cat("Done. Figures written to", file.path(output_root(), "timecourse"), "\n")
