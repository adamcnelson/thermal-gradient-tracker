library(dplyr)
library(ggplot2)

# Section 3.3 — distributions / histograms (frame-level).
#
# For Gi and Gq separately: distribution of floor_temp_mean / mouse_surface_temp_mean
# / mouse_minus_floor_temp_mean for DCZ vs Vehicle, as separate figures for
# rest-state (stationary == TRUE) and non-rest-state (stationary == FALSE).
# velocity_smooth_px_s is the exception, per Adam: a single distribution per
# virus x injection, not split by rest state.
#
# Uses raw frame-level values (not per-mouse aggregates) — matches
# src/treatment_plots.py::plot_floor_temp_distribution, which histograms raw
# per-frame values rather than per-mouse means.

.dist_outcome_labels <- c(
  floor_temp_mean = "Floor temperature (°C)",
  mouse_surface_temp_mean = "Mouse surface temperature (°C)",
  mouse_minus_floor_temp_mean = "Mouse − floor temperature (°C)"
)

.dist_base <- function(master_df, outcome) {
  master_df |>
    filter(
      injection %in% c("DCZ", "Vehicle"),
      virus %in% c("Gi", "Gq"),
      !is.na(.data[[outcome]])
    ) |>
    mutate(injection = factor(injection, levels = c("Vehicle", "DCZ")))
}

plot_distribution_one <- function(master_df, outcome, virus_val, stationary_val) {
  df <- .dist_base(master_df, outcome) |>
    filter(virus == virus_val, stationary == stationary_val)
  if (nrow(df) == 0) return(NULL)

  ylabel <- .dist_outcome_labels[[outcome]]
  state_label <- ifelse(stationary_val, "rest (stationary)", "non-rest (non-stationary)")

  ggplot(df, aes(x = .data[[outcome]], fill = injection)) +
    geom_histogram(aes(y = after_stat(density)), position = "identity", alpha = 0.5, bins = 30) +
    labs(
      title = sprintf("%s distribution — %s, %s", ylabel, virus_val, state_label),
      subtitle = "DCZ vs Vehicle; frame-level, qc_flag == 'ok'",
      x = ylabel, y = "Density", fill = "Injection"
    ) +
    theme_minimal()
}

plot_velocity_distribution_one <- function(master_df, virus_val) {
  df <- .dist_base(master_df, "velocity_smooth_px_s") |>
    filter(virus == virus_val)
  if (nrow(df) == 0) return(NULL)

  ggplot(df, aes(x = velocity_smooth_px_s, fill = injection)) +
    geom_histogram(aes(y = after_stat(density)), position = "identity", alpha = 0.5, bins = 30) +
    labs(
      title = sprintf("Velocity (px/s) distribution — %s", virus_val),
      subtitle = "DCZ vs Vehicle; frame-level, qc_flag == 'ok'; not split by rest state (per Adam)",
      x = "Velocity (px/s)", y = "Density", fill = "Injection"
    ) +
    theme_minimal()
}

build_distribution_plots <- function(master_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  for (outcome in names(.dist_outcome_labels)) {
    for (virus_val in c("Gi", "Gq")) {
      for (stationary_val in c(TRUE, FALSE)) {
        p <- plot_distribution_one(master_df, outcome, virus_val, stationary_val)
        if (is.null(p)) {
          message(sprintf("skip (no data): %s / %s / stationary=%s", outcome, virus_val, stationary_val))
          next
        }
        state_slug <- ifelse(stationary_val, "rest", "nonrest")
        fname <- sprintf("dist_%s_%s_%s.png", outcome, virus_val, state_slug)
        ggsave(file.path(output_dir, fname), p, width = 7, height = 4.5, dpi = 150)
      }
    }
  }

  for (virus_val in c("Gi", "Gq")) {
    p <- plot_velocity_distribution_one(master_df, virus_val)
    if (is.null(p)) {
      message(sprintf("skip (no data): velocity_smooth_px_s / %s", virus_val))
      next
    }
    fname <- sprintf("dist_velocity_smooth_px_s_%s.png", virus_val)
    ggsave(file.path(output_dir, fname), p, width = 7, height = 4.5, dpi = 150)
  }
}
