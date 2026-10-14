# DCZ vs Vehicle, one dot per experiment (session x lane), per virus x outcome x nestlet tier.
# Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/dcz_vehicle_units.R")
source("R/tracking_activity.R")
source("R/plot_all_experiments.R")

frames <- load_frames()
bouts <- load_bouts()
activity <- load_activity(frames)

build_all_experiments_plots(frames, bouts, output_dir = file.path(output_root(), "all_experiments"), activity_lanes = activity$lanes)

cat("Done. Figures written to", file.path(output_root(), "all_experiments"), "\n")
