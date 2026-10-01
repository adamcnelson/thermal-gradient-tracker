# Sanity-check the input schema and the Craniotomy join before building any
# plots on top of them. Run from r_analysis_RGB_Thermal/ (or via renv::run()).
source("R/data.R")

frames <- load_frames()
cat("\nmaster_landmarks_with_metadata_frame.csv (craniotomy joined):\n")
cat(sprintf("  %d rows, %d columns\n", nrow(frames), ncol(frames)))
cat("  craniotomy values:\n")
print(table(frames$craniotomy, useNA = "ifany"))
cat("  virus x injection counts:\n")
print(table(frames$virus, frames$injection, useNA = "ifany"))
cat("  stationary counts:\n")
print(table(frames$stationary, useNA = "ifany"))

cat("\n  outcome applicability (non-NA vs. after the full gate):\n")
for (outcome in c("mouse_surface_temp_mean_c", "warm_spot_temp_c", "tail_delta_t_c")) {
  n_nonNA <- sum(!is.na(frames[[outcome]]))
  n_applicable <- nrow(filter_for_outcome(frames, outcome))
  cat(sprintf("    %-28s non-NA=%-5d applicable=%-5d\n", outcome, n_nonNA, n_applicable))
}

bouts <- load_bouts()
cat("\nmaster_landmarks_with_metadata_bout.csv (craniotomy joined):\n")
cat(sprintf("  %d rows, %d columns\n", nrow(bouts), ncol(bouts)))
cat("  craniotomy values:\n")
print(table(bouts$craniotomy, useNA = "ifany"))
cat("\n  outcome applicability (non-NA vs. after the full gate):\n")
for (outcome in c("dorsal_mean_c", "warm_spot_temp_c", "tail_delta_t_c")) {
  n_nonNA <- sum(!is.na(bouts[[outcome]]))
  n_applicable <- nrow(filter_for_outcome(bouts, outcome))
  cat(sprintf("    %-28s non-NA=%-5d applicable=%-5d\n", outcome, n_nonNA, n_applicable))
}

cat("\nOK: schema loaded and Craniotomy join has full coverage (join_craniotomy() stops on any unmatched row, so reaching this line means it passed).\n")
