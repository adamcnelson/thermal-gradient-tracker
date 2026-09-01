library(dplyr)
library(tidyr)
library(ggplot2)

# Section 3.2.1 — ggplot2 reimplementation of
# src/treatment_plots.py::plot_dcz_vehicle_paired().
#
# One figure per outcome, faceted virus (rows: Gi, Gq) x bout type (cols:
# Stationary, Non-stationary). Each mouse is a connected line (Vehicle -> DCZ,
# per-mouse mean); group mean +/- SE is overlaid as a black diamond offset to
# the right of the paired points, matching the Python reference plots in
# ../bouts/analysis_plots/paired_dcz_vehicle_*.png.

.spaghetti_outcome_labels <- c(
  floor_temp_mean = "Floor temperature (°C)",
  mouse_surface_temp_mean = "Mouse surface temperature (°C)",
  mouse_minus_floor_temp_mean = "Mouse − floor temperature (°C)",
  velocity_smooth_px_s = "Velocity (px/s)"
)

.sem2 <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

plot_dcz_vehicle_paired_one <- function(master_df, outcome) {
  exp_df <- master_df |>
    filter(
      phase == "experimental",
      injection %in% c("DCZ", "Vehicle"),
      virus %in% c("Gi", "Gq"),
      !is.na(.data[[outcome]])
    ) |>
    mutate(
      stat_label = factor(
        ifelse(stationary, "Stationary", "Non-stationary"),
        levels = c("Stationary", "Non-stationary")
      ),
      mouse_id = as.character(mouse_id),
      injection = factor(injection, levels = c("Vehicle", "DCZ")),
      x_pos = ifelse(injection == "Vehicle", 0, 1)
    )

  if (nrow(exp_df) == 0) return(NULL)

  mouse_means <- exp_df |>
    group_by(virus, stat_label, injection, x_pos, mouse_id) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop")

  group_summary <- mouse_means |>
    group_by(virus, stat_label, injection, x_pos) |>
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sem_value = .sem2(value),
      .groups = "drop"
    ) |>
    mutate(x_offset = x_pos + 0.22)

  ylabel <- .spaghetti_outcome_labels[[outcome]]
  if (is.null(ylabel)) ylabel <- outcome

  ggplot(mouse_means, aes(x = x_pos, y = value)) +
    geom_line(aes(group = mouse_id, color = mouse_id), linewidth = 0.8, alpha = 0.5) +
    geom_point(aes(color = mouse_id), size = 2.5, alpha = 0.9) +
    geom_errorbar(
      data = group_summary,
      aes(x = x_offset, y = mean_value, ymin = mean_value - sem_value, ymax = mean_value + sem_value),
      width = 0.06, color = "black", linewidth = 0.8, inherit.aes = FALSE
    ) +
    geom_point(
      data = group_summary,
      aes(x = x_offset, y = mean_value),
      shape = 23, fill = "black", color = "black", size = 3, inherit.aes = FALSE
    ) +
    facet_grid(virus ~ stat_label) +
    scale_x_continuous(breaks = c(0, 1), labels = c("Vehicle", "DCZ"), limits = c(-0.4, 1.55)) +
    labs(
      title = ylabel,
      subtitle = "DCZ vs Vehicle  |  experimental phase  |  per-mouse mean",
      x = NULL, y = ylabel, color = "Mouse ID"
    ) +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
}

build_dcz_vehicle_paired_plots <- function(master_df, output_dir,
                                            outcomes = names(.spaghetti_outcome_labels)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (outcome in outcomes) {
    p <- plot_dcz_vehicle_paired_one(master_df, outcome)
    if (is.null(p)) {
      message(sprintf("skip (no data): %s", outcome))
      next
    }
    fname <- sprintf("paired_dcz_vehicle_%s.png", outcome)
    ggsave(file.path(output_dir, fname), p, width = 8, height = 8, dpi = 150)
  }
}
