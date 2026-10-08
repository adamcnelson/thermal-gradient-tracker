# Section 4.2.2 — craniotomy-effect paired plots. Run from r_analysis_RGB_Thermal/.
# Pre (habituation, Saline) vs Post (experimental, Vehicle): confounded with phase -- see R/plot_craniotomy.R.
source("R/data.R")
source("R/plot_spaghetti.R")      # outcome labels
source("R/plot_paired_common.R")
source("R/plot_craniotomy.R")

frames <- load_frames()
bouts <- load_bouts()
build_craniotomy_plots(frames, bouts, output_dir = "output/craniotomy")
cat("Done. Figures written to r_analysis_RGB_Thermal/output/craniotomy/\n")
