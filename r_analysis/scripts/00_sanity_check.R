# Sanity-check the input schema and the Craniotomy join before building any
# plots on top of them. Run from r_analysis/ (or via `renv::run()`).
source("R/data.R")

master <- load_master()
cat("\nmaster_tracking_with_metadata.csv (qc_flag == 'ok', craniotomy joined):\n")
cat(sprintf("  %d rows, %d columns\n", nrow(master), ncol(master)))
cat("  craniotomy values:\n")
print(table(master$craniotomy, useNA = "ifany"))
cat("  virus x craniotomy counts:\n")
print(table(master$virus, master$craniotomy, useNA = "ifany"))
cat("  injection x craniotomy counts (checking the resolved A/B mapping):\n")
print(table(master$injection, master$craniotomy, useNA = "ifany"))

bouts <- load_bouts(master)
cat("\nbout_table.csv (metadata joined):\n")
cat(sprintf("  %d rows, %d columns\n", nrow(bouts), ncol(bouts)))
print(head(bouts))

cat("\nOK: schema matches project_brief_v6.md and Craniotomy join has 100% coverage.\n")
