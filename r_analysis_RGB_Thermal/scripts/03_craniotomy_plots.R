# Section 4.2.2 — craniotomy-effect paired plots. Run from r_analysis_RGB_Thermal/.
# Two contrasts, both confounded with time/experience -- see R/plot_craniotomy.R:
#   craniotomy_*.png         Pre (habituation, Saline) vs Post (experimental, Vehicle)
#   craniotomy_saline_*.png  Pre (habituation, Saline) vs Post (rehabituation, Saline)
source("R/data.R")
source("R/plot_spaghetti.R")      # outcome labels
source("R/plot_paired_common.R")
source("R/plot_craniotomy.R")

frames <- load_frames()
bouts <- load_bouts()
build_craniotomy_plots(frames, bouts, output_dir = "output/craniotomy")
build_craniotomy_plots(frames, bouts, output_dir = "output/craniotomy", post_injection = "Saline")
cat("Done. Figures written to r_analysis_RGB_Thermal/output/craniotomy/\n")
