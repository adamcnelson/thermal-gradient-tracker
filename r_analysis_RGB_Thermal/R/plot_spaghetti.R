library(dplyr)
library(tidyr)
library(ggplot2)

# Section 4.2.1 — per-mouse paired DCZ-vs-Vehicle plots, extending
# r_analysis/R/plot_spaghetti.R::plot_dcz_vehicle_paired_one() (which
# reproduces src/treatment_plots.py::plot_dcz_vehicle_paired() for the
# legacy pipeline's single mouse_surface_temp_mean outcome) to the 3
# frame-level feature outcomes + floor preference.
#
# One figure per outcome, faceted virus (rows) x stationary state (cols:
# Stationary, Non-stationary) for the 3 frame-level outcomes -- same visual
# structure as the legacy reference plots. Floor preference is bout-level
# (no stationary axis to facet by -- a bout already IS stationary time by
# construction), so it gets its own function/figure, faceted by virus only.

.spaghetti_outcome_labels <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temperature (°C)",
  warm_spot_temp_c = "Warm-spot temperature (°C)",
  tail_delta_t_c = "Tail ΔT (°C)"
)

.sem2 <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

plot_dcz_vehicle_paired_one <- function(frames_df, outcome) {
  exp_df <- frames_df |>
    filter(phase == "experimental", injection %in% c("DCZ", "Vehicle"), virus %in% c("Gi", "Gq")) |>
    filter_for_outcome(outcome) |>
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
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), n_samples = n(), .groups = "drop")

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
      subtitle = paste0("DCZ vs Vehicle  |  experimental phase  |  per-mouse mean  |  ", .paired_n_label(mouse_means)),
      x = NULL, y = ylabel, color = "Mouse ID"
    ) +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
}

# "paired mice: Gi 4, Gq 4" -- mice per virus with BOTH conditions (the ones drawn as lines),
# computed from the data rather than stated, so the label can't go stale as the corpus grows.
.paired_n_label <- function(mouse_means) {
  n <- mouse_means |>
    group_by(virus, mouse_id) |>
    summarise(both = n_distinct(injection) == 2, .groups = "drop") |>
    group_by(virus) |>
    summarise(n = sum(both), .groups = "drop")
  paste0("paired mice: ", paste(n$virus, n$n, collapse = ", "))
}

# Floor preference: bout-level, no stationary axis -- facet by virus only.
plot_floor_preference_paired <- function(bouts_df) {
  df <- bouts_df |>
    filter(phase == "experimental", injection %in% c("DCZ", "Vehicle"), virus %in% c("Gi", "Gq"),
           !is.na(mean_floor_temp_c)) |>
    mutate(
      mouse_id = as.character(mouse_id),
      injection = factor(injection, levels = c("Vehicle", "DCZ")),
      x_pos = ifelse(injection == "Vehicle", 0, 1)
    )

  if (nrow(df) == 0) return(NULL)

  mouse_means <- df |>
    group_by(virus, injection, x_pos, mouse_id) |>
    summarise(value = mean(mean_floor_temp_c, na.rm = TRUE), n_bouts = n(), .groups = "drop")

  group_summary <- mouse_means |>
    group_by(virus, injection, x_pos) |>
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sem_value = .sem2(value),
      .groups = "drop"
    ) |>
    mutate(x_offset = x_pos + 0.22)

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
    facet_wrap(~virus) +
    scale_x_continuous(breaks = c(0, 1), labels = c("Vehicle", "DCZ"), limits = c(-0.4, 1.55)) +
    labs(
      title = "Floor temperature preference (bout-level mean)",
      subtitle = paste0("DCZ vs Vehicle  |  experimental phase\nper-mouse mean across bouts  |  ", .paired_n_label(mouse_means)),
      x = NULL, y = "Mean floor temp at bout (°C)", color = "Mouse ID"
    ) +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
}

build_dcz_vehicle_paired_plots <- function(frames_df, bouts_df, output_dir,
                                            outcomes = names(.spaghetti_outcome_labels)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (outcome in outcomes) {
    p <- plot_dcz_vehicle_paired_one(frames_df, outcome)
    if (is.null(p)) {
      message(sprintf("skip (no data): %s", outcome))
      next
    }
    fname <- sprintf("paired_dcz_vehicle_%s.png", outcome)
    ggsave(file.path(output_dir, fname), p, width = 8, height = 8, dpi = 150)
  }

  p_floor <- plot_floor_preference_paired(bouts_df)
  if (is.null(p_floor)) {
    message("skip (no data): floor_preference")
  } else {
    ggsave(file.path(output_dir, "paired_dcz_vehicle_floor_preference.png"), p_floor, width = 7, height = 5, dpi = 150)
  }
}
