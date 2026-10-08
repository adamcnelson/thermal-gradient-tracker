library(dplyr)
library(ggplot2)

# Section 4.2.3 — bout organization, post-craniotomy only, DCZ vs Vehicle, per virus:
#   - bouts per session: stationary bouts detected in one session x lane (after the entry
#     filter); per-mouse mean across that mouse's sessions in each condition
#   - mean bout duration (s): bout_end - bout_start, per-mouse mean over all its bouts
# Bouts come from the thermal tracking, so the three sessions with short RGB video
# (Test8-005, Test10-005, Test11-010) are not undercounted here.

.post_dcz_vehicle <- function(bouts_df) {
  bouts_df |>
    filter(craniotomy == "Post", phase == "experimental", injection %in% c("DCZ", "Vehicle"),
           virus %in% c("Gi", "Gq")) |>
    mutate(cond = factor(injection, levels = c("Vehicle", "DCZ")),
           bout_duration_s = bout_end_thermal_sec - bout_start_thermal_sec)
}

plot_bout_count <- function(bouts_df) {
  mm <- .post_dcz_vehicle(bouts_df) |>
    count(virus, cond, mouse_id, session, track, name = "n_bouts") |>
    group_by(virus, cond, mouse_id) |>
    summarise(value = mean(n_bouts), .groups = "drop")
  plot_paired_generic(mm, facet_vars = "virus", facet_formula = . ~ virus,
                      title = "Bout organization — stationary bouts per session",
                      ylabel = "Bouts per session (per-mouse mean)",
                      subtitle = "Post-craniotomy, experimental phase  |  per-mouse mean across sessions")
}

plot_bout_duration <- function(bouts_df) {
  mm <- .post_dcz_vehicle(bouts_df) |>
    group_by(virus, cond, mouse_id) |>
    summarise(value = mean(bout_duration_s, na.rm = TRUE), .groups = "drop")
  plot_paired_generic(mm, facet_vars = "virus", facet_formula = . ~ virus,
                      title = "Bout organization — mean bout duration",
                      ylabel = "Bout duration (s, per-mouse mean)",
                      subtitle = "Post-craniotomy, experimental phase  |  per-mouse mean over all bouts")
}

build_bout_organization_plots <- function(bouts_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(output_dir, "bout_count_per_session.png"), plot_bout_count(bouts_df), width = 8, height = 5.5, dpi = 150)
  ggsave(file.path(output_dir, "bout_duration.png"), plot_bout_duration(bouts_df), width = 8, height = 5.5, dpi = 150)
}
