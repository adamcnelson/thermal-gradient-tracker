# Section 4.2.3 — bout organization (bouts per session, bout duration), DCZ vs Vehicle. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_paired_common.R")
source("R/plot_bout_organization.R")

bouts <- load_bouts()
build_bout_organization_plots(bouts, output_dir = "output/bout_organization")
cat("Done. Figures written to r_analysis_RGB_Thermal/output/bout_organization/\n")
