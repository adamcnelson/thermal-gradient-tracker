# Section 3.2.2 (craniotomy effect) and 3.2.3 (bout organization effect).
# Run from r_analysis/.
source("R/data.R")
source("R/plot_craniotomy.R")
source("R/plot_bout_organization.R")

master <- load_master()
bouts <- load_bouts(master)

build_craniotomy_effect_plots(master, output_dir = "output/spaghetti")
build_bout_organization_plots(bouts, output_dir = "output/spaghetti")

cat("Done. Figures written to r_analysis/output/spaghetti/\n")
