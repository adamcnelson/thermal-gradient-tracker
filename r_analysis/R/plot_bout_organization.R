library(dplyr)
library(ggplot2)

# Section 3.2.3 — bout organization effect (new plot, not in the Python
# reference). Post-craniotomy only, Gi and Gq separately: does DCZ vs Vehicle
# affect (a) number of bouts per session, (b) bout duration? Uses the
# bout_table.csv metadata join from open question C (via load_bouts()).
# Same paired-per-mouse visual language as 3.2.1/3.2.2.

.sem4 <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

.filter_post_dcz_vehicle <- function(bout_df) {
  bout_df |>
    filter(
      craniotomy == "Post",
      injection %in% c("DCZ", "Vehicle"),
      virus %in% c("Gi", "Gq")
    ) |>
    mutate(
      mouse_id = as.character(mouse_id),
      injection = factor(injection, levels = c("Vehicle", "DCZ")),
      x_pos = ifelse(injection == "Vehicle", 0, 1)
    )
}

# Shared paired-plot renderer: per-mouse points at x_pos, connected across
# injection, faceted by virus, with a black mean +/- SE diamond offset right.
.render_paired_by_virus <- function(mouse_means, title, ylabel) {
  group_summary <- mouse_means |>
    group_by(virus, injection, x_pos) |>
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sem_value = .sem4(value),
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
      title = title,
      subtitle = "DCZ vs Vehicle  |  post-craniotomy only  |  per-mouse mean",
      x = NULL, y = ylabel, color = "Mouse ID"
    ) +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
}

# (a) Number of bouts per session, averaged per mouse x injection.
plot_bout_count_by_injection <- function(bout_df) {
  df <- .filter_post_dcz_vehicle(bout_df)
  if (nrow(df) == 0) return(NULL)

  per_session <- df |>
    count(video_file, mouse_id, virus, injection, x_pos, name = "n_bouts")

  mouse_means <- per_session |>
    group_by(virus, injection, x_pos, mouse_id) |>
    summarise(value = mean(n_bouts, na.rm = TRUE), .groups = "drop")

  .render_paired_by_virus(
    mouse_means,
    title = "Number of stationary bouts per session",
    ylabel = "Bouts per session — per-mouse mean"
  )
}

# (b) Bout duration, averaged per mouse x injection (pooled across sessions).
plot_bout_duration_by_injection <- function(bout_df) {
  df <- .filter_post_dcz_vehicle(bout_df) |>
    filter(!is.na(bout_duration_sec))
  if (nrow(df) == 0) return(NULL)

  mouse_means <- df |>
    group_by(virus, injection, x_pos, mouse_id) |>
    summarise(value = mean(bout_duration_sec, na.rm = TRUE), .groups = "drop")

  .render_paired_by_virus(
    mouse_means,
    title = "Stationary bout duration",
    ylabel = "Bout duration (s) — per-mouse mean"
  )
}

build_bout_organization_plots <- function(bout_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  p_count <- plot_bout_count_by_injection(bout_df)
  if (!is.null(p_count)) {
    ggsave(file.path(output_dir, "bout_count_by_injection.png"), p_count, width = 7, height = 5, dpi = 150)
  } else {
    message("skip (no data): bout_count_by_injection")
  }

  p_dur <- plot_bout_duration_by_injection(bout_df)
  if (!is.null(p_dur)) {
    ggsave(file.path(output_dir, "bout_duration_by_injection.png"), p_dur, width = 7, height = 5, dpi = 150)
  } else {
    message("skip (no data): bout_duration_by_injection")
  }
}
