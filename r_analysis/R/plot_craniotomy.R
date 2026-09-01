library(dplyr)
library(ggplot2)

# Section 3.2.2 — craniotomy effect (new plot, not in the Python reference).
#
# Gi and Gq combined (per the brief: "combining Gi and Gq animals together"),
# split by rest bout (stationary) vs non-rest bout, x = pre- vs post-
# craniotomy, y = temp / temp difference / velocity. Per-mouse paired means
# connected pre -> post, group mean +/- SE overlaid as a black diamond — same
# visual language as the 3.2.1 paired plots for consistency.
#
# Non-rest-bout data has to come from master_tracking_with_metadata.csv
# (stationary == FALSE), not bout_table.csv, which by construction only has
# already-stationary rows (open question C) — this function operates on
# master_df, not the bout table.
#
# Post-craniotomy is restricted to injection %in% c("DCZ", "Vehicle"), which
# also excludes any Saline "rehabituation" trials (open question B) — none
# are present in this run, but the filter is written to hold if any ever are.

.craniotomy_outcome_labels <- c(
  mouse_surface_temp_mean = "Mouse surface temperature (°C)",
  floor_temp_mean = "Floor temperature (°C)",
  mouse_minus_floor_temp_mean = "Mouse − floor temperature (°C)",
  velocity_smooth_px_s = "Velocity (px/s)"
)

.sem3 <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

plot_craniotomy_effect_one <- function(master_df, outcome) {
  df <- master_df |>
    filter(!is.na(craniotomy), !is.na(.data[[outcome]]), virus %in% c("Gi", "Gq")) |>
    filter(craniotomy == "Pre-craniotomy" | injection %in% c("DCZ", "Vehicle")) |>
    mutate(
      stat_label = factor(
        ifelse(stationary, "Stationary", "Non-stationary"),
        levels = c("Stationary", "Non-stationary")
      ),
      mouse_id = as.character(mouse_id),
      craniotomy = factor(craniotomy, levels = c("Pre-craniotomy", "Post")),
      x_pos = ifelse(craniotomy == "Pre-craniotomy", 0, 1)
    )

  if (nrow(df) == 0) return(NULL)

  mouse_means <- df |>
    group_by(stat_label, craniotomy, x_pos, mouse_id) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop")

  group_summary <- mouse_means |>
    group_by(stat_label, craniotomy, x_pos) |>
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sem_value = .sem3(value),
      .groups = "drop"
    ) |>
    mutate(x_offset = x_pos + 0.22)

  ylabel <- .craniotomy_outcome_labels[[outcome]]
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
    facet_wrap(~stat_label) +
    scale_x_continuous(breaks = c(0, 1), labels = c("Pre-craniotomy", "Post"), limits = c(-0.4, 1.55)) +
    labs(
      title = ylabel,
      subtitle = "Pre- vs post-craniotomy  |  Gi + Gq combined  |  post restricted to DCZ/Vehicle  |  per-mouse mean",
      x = NULL, y = ylabel, color = "Mouse ID"
    ) +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
}

build_craniotomy_effect_plots <- function(master_df, output_dir,
                                           outcomes = names(.craniotomy_outcome_labels)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (outcome in outcomes) {
    p <- plot_craniotomy_effect_one(master_df, outcome)
    if (is.null(p)) {
      message(sprintf("skip (no data): %s", outcome))
      next
    }
    fname <- sprintf("craniotomy_effect_%s.png", outcome)
    ggsave(file.path(output_dir, fname), p, width = 8, height = 5, dpi = 150)
  }
}
