library(dplyr)
library(ggplot2)

# Section 4.2.2 — craniotomy effect: per-mouse Pre- vs Post-craniotomy, Gi+Gq combined, split
# stationary / non-stationary (frame-level outcomes) plus bout-level floor preference.
#
# DESIGN (2026-10-08): in this corpus craniotomy is perfectly confounded with phase -- every
# Pre-craniotomy session is habituation (Saline) and every Post-craniotomy session is
# experimental (DCZ/Vehicle). The comparison used is therefore
#     Pre  = habituation, Saline        vs        Post = experimental, Vehicle
# -- both control injections, so DCZ is out of the comparison, but phase / experience / time
# since habituation still differ with craniotomy. A difference here is "craniotomy + phase",
# not craniotomy alone; every figure says so.

CRANIOTOMY_CONTRAST <- c("Pre (habituation, Saline)", "Post (experimental, Vehicle)")

.cranio_subset <- function(df) {
  df |>
    filter(virus %in% c("Gi", "Gq"),
           (craniotomy == "Pre-craniotomy" & injection == "Saline") |
             (craniotomy == "Post" & injection == "Vehicle")) |>
    mutate(cond = factor(ifelse(craniotomy == "Pre-craniotomy", CRANIOTOMY_CONTRAST[1], CRANIOTOMY_CONTRAST[2]),
                         levels = CRANIOTOMY_CONTRAST))
}

.cranio_subtitle <- "Gi+Gq combined  |  per-mouse mean  |  CONFOUNDED: craniotomy coincides with phase (Pre = habituation)"

plot_craniotomy_one <- function(frames_df, outcome) {
  mm <- frames_df |>
    .cranio_subset() |>
    filter_for_outcome(outcome) |>
    mutate(stat_label = factor(ifelse(stationary, "Stationary", "Non-stationary"),
                               levels = c("Stationary", "Non-stationary"))) |>
    group_by(stat_label, cond, mouse_id) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  ylabel <- .spaghetti_outcome_labels[[outcome]]
  plot_paired_generic(mm, facet_vars = "stat_label", facet_formula = . ~ stat_label,
                      title = paste(ylabel, "— craniotomy effect"), ylabel = ylabel, subtitle = .cranio_subtitle)
}

plot_craniotomy_floor <- function(bouts_df) {
  mm <- bouts_df |>
    .cranio_subset() |>
    filter(!is.na(mean_floor_temp_c)) |>
    group_by(cond, mouse_id) |>
    summarise(value = mean(mean_floor_temp_c, na.rm = TRUE), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  plot_paired_generic(mm, title = "Floor temperature preference — craniotomy effect",
                      ylabel = "Mean floor temp at bout (°C)",
                      subtitle = "Gi+Gq combined  |  per-mouse mean across bouts\nCONFOUNDED: craniotomy coincides with phase (Pre = habituation)")
}

build_craniotomy_plots <- function(frames_df, bouts_df, output_dir,
                                   outcomes = names(.spaghetti_outcome_labels)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (outcome in outcomes) {
    p <- plot_craniotomy_one(frames_df, outcome)
    if (is.null(p)) { message("skip (no data): ", outcome); next }
    ggsave(file.path(output_dir, sprintf("craniotomy_%s.png", outcome)), p, width = 9, height = 5.5, dpi = 150)
  }
  p <- plot_craniotomy_floor(bouts_df)
  if (!is.null(p)) ggsave(file.path(output_dir, "craniotomy_floor_preference.png"), p, width = 6.5, height = 5.5, dpi = 150)
}
