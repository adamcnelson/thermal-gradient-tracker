library(dplyr)
library(ggplot2)

# Section 4.1 — time-course plots (frame-level for the 3 feature outcomes;
# bout-level for the floor-preference outcome, which has no frame-level
# equivalent -- see data.R's OUTCOME_COLS note on why floor_temp_mean_c
# can't stand in for mean_floor_temp_c).
#
# Craniotomy split deliberately OMITTED for now (Adam, 2026-09-02): every
# session in the current 6-session corpus is craniotomy=="Post" -- there is
# no pre-craniotomy data to split against, so a pre/post facet would just
# produce empty "pre" figures. Revisit once pre-craniotomy sessions exist
# in the pipeline; filter_for_outcome()/join_craniotomy() already carry
# `craniotomy` through so adding the split back is a small change, not a
# redesign.
#
# Time alignment: same binning approach as r_analysis/R/plot_timecourse.R
# (open question D there) -- bin elapsed_time_thermal_sec into bin_width-
# second intervals, take each mouse's per-bin mean first, then average
# those per-mouse bin means across mice per injection, so a mouse with more
# samples in a bin doesn't dominate. SEM is 0 when only one mouse
# contributes.
TIMECOURSE_BIN_WIDTH_SEC <- 30

.sem <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

.outcome_labels <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temp (°C)",
  warm_spot_temp_c = "Warm-spot temp (°C)",
  tail_delta_t_c = "Tail ΔT (°C)"
)

# One figure: per-mouse thin traces + group mean +/- SEM ribbon, colored by
# injection, for a single (virus, outcome) combination. Restricted to
# injection %in% c("DCZ", "Vehicle") -- matches r_analysis/'s post-craniotomy
# convention; since this corpus is all-Post already, that's the only
# meaningful restriction available (excludes Saline/rehabituation, none of
# which are present here anyway).
plot_timecourse_one <- function(frames_df, virus_val, outcome, bin_width = TIMECOURSE_BIN_WIDTH_SEC) {
  df <- frames_df |>
    filter(virus == virus_val, injection %in% c("DCZ", "Vehicle")) |>
    filter_for_outcome(outcome) |>
    mutate(
      time_bin = floor(elapsed_time_thermal_sec / bin_width) * bin_width,
      mouse_id = as.character(mouse_id)
    )

  if (nrow(df) == 0) return(NULL)

  n_mice <- df |> distinct(injection, mouse_id) |> count(injection, name = "n_mice")
  n_mice_label <- paste(sprintf("%s: n=%d mouse", n_mice$injection, n_mice$n_mice), collapse = ", ")

  mouse_traces <- df |>
    group_by(mouse_id, injection, time_bin) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop")

  group_trace <- mouse_traces |>
    group_by(injection, time_bin) |>
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sem_value = .sem(value),
      .groups = "drop"
    ) |>
    mutate(ymin = mean_value - sem_value, ymax = mean_value + sem_value)

  ylabel <- .outcome_labels[[outcome]]
  if (is.null(ylabel)) ylabel <- outcome

  ggplot() +
    geom_line(
      data = mouse_traces,
      aes(x = time_bin, y = value, color = injection, group = interaction(mouse_id, injection)),
      linewidth = 0.3, alpha = 0.25
    ) +
    geom_ribbon(
      data = group_trace,
      aes(x = time_bin, ymin = ymin, ymax = ymax, fill = injection),
      alpha = 0.2, color = NA
    ) +
    geom_line(
      data = group_trace,
      aes(x = time_bin, y = mean_value, color = injection),
      linewidth = 1.1
    ) +
    labs(
      title = sprintf("%s — %s", ylabel, virus_val),
      subtitle = sprintf(
        "Per-mouse traces (thin) + group mean ± SEM (%ds bins); post-craniotomy only; %s",
        bin_width, n_mice_label
      ),
      x = "Elapsed time (thermal clock, s)", y = ylabel, color = "Injection", fill = "Injection"
    ) +
    theme_minimal()
}

# Floor-preference time-course: bout-level (mean_floor_temp_c), NOT binned
# the way the frame-level outcomes are -- bouts are already discrete events,
# so each bout's mean floor temp is plotted directly at its bout_start_
# thermal_sec, one thin per-mouse line/points per injection (no smoothing —
# bout counts per mouse are too low in this corpus to justify a fitted
# trend line without overclaiming).
plot_floor_preference_timecourse <- function(bouts_df, virus_val) {
  df <- bouts_df |>
    filter(virus == virus_val, injection %in% c("DCZ", "Vehicle"), !is.na(mean_floor_temp_c)) |>
    mutate(mouse_id = as.character(mouse_id))

  if (nrow(df) == 0) return(NULL)

  n_mice <- df |> distinct(injection, mouse_id) |> count(injection, name = "n_mice")
  n_mice_label <- paste(sprintf("%s: n=%d mouse", n_mice$injection, n_mice$n_mice), collapse = ", ")

  ggplot(df, aes(x = bout_start_thermal_sec, y = mean_floor_temp_c, color = injection,
                 group = interaction(mouse_id, injection))) +
    geom_line(linewidth = 0.5, alpha = 0.4) +
    geom_point(size = 2, alpha = 0.8) +
    labs(
      title = sprintf("Floor temperature preference (bout-level) — %s", virus_val),
      subtitle = sprintf(
        "Each point = one stationary bout's mean floor temp; post-craniotomy only; %s",
        n_mice_label
      ),
      x = "Bout start (elapsed thermal time, s)", y = "Mean floor temp at bout (°C)",
      color = "Injection"
    ) +
    theme_minimal()
}

# Builds all 4.1 figures: 2 viruses x 3 frame-level outcomes = 6 time-course
# figures, plus 2 viruses x 1 floor-preference figure = 2 more (8 total).
build_timecourse_plots <- function(frames_df, bouts_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  viruses <- c("Gi", "Gq")
  outcomes <- names(.outcome_labels)

  for (virus_val in viruses) {
    for (outcome in outcomes) {
      p <- plot_timecourse_one(frames_df, virus_val, outcome)
      if (is.null(p)) {
        message(sprintf("skip (no data): %s / %s", outcome, virus_val))
        next
      }
      fname <- sprintf("timecourse_%s_%s.png", outcome, virus_val)
      ggsave(file.path(output_dir, fname), p, width = 8, height = 5, dpi = 150)
    }

    p_floor <- plot_floor_preference_timecourse(bouts_df, virus_val)
    if (is.null(p_floor)) {
      message(sprintf("skip (no data): floor_preference / %s", virus_val))
      next
    }
    fname <- sprintf("timecourse_floor_preference_%s.png", virus_val)
    ggsave(file.path(output_dir, fname), p_floor, width = 8, height = 5, dpi = 150)
  }
}
