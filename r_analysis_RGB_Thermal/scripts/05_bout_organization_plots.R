# Section 4.2.3 — bout organization (bouts per session, bout duration), DCZ vs Vehicle. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/plot_paired_common.R")
source("R/plot_bout_organization.R")

bouts <- load_bouts()
build_bout_organization_plots(bouts, output_dir = file.path(output_root(), "bout_organization"))
cat("Done. Figures written to", file.path(output_root(), "bout_organization"), "\n")
