# DCZ vs Vehicle, one dot per experiment (session x lane), per virus x outcome x nestlet tier.
# Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/dcz_vehicle_units.R")
source("R/plot_all_experiments.R")

frames <- load_frames()
bouts <- load_bouts()

build_all_experiments_plots(frames, bouts, output_dir = "output/all_experiments")

cat("Done. Figures written to r_analysis_RGB_Thermal/output/all_experiments/<tier>/\n")
