library(dplyr)
library(ggplot2)

# Section 4.3 — distributions / histograms (frame-level).
#
# For Gi and Gq separately: distribution of dorsal_mean_c / warm_spot_temp_c
# / tail_delta_t_c for DCZ vs Vehicle, as separate figures for rest-state
# (stationary == TRUE) and non-rest-state (stationary == FALSE) -- same
# structure as r_analysis/R/plot_distributions.R, extended from 1 outcome
# (mouse_surface_temp_mean) to the new pipeline's 3.
#
# velocity_smooth_px_s: one distribution per virus, NOT split by stationary state -- matching
# r_analysis/R/plot_distributions.R::plot_velocity_distribution_one(). It isn't a native Stage 7
# column: ../scripts/join_landmark_metadata.py --velocity-from attaches it from the re-track run's
# master_tracking_with_metadata.csv (compute_velocity() output) by nearest same-lane timestamp.

.dist_outcome_labels <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temperature (°C)",
  warm_spot_temp_c = "Warm-spot temperature (°C)",
  tail_delta_t_c = "Tail ΔT (°C)"
)

.dist_base <- function(frames_df, outcome) {
  frames_df |>
    filter(injection %in% c("DCZ", "Vehicle"), virus %in% c("Gi", "Gq")) |>
    filter_for_outcome(outcome) |>
    mutate(injection = factor(injection, levels = c("Vehicle", "DCZ")))
}

plot_distribution_one <- function(frames_df, outcome, virus_val, stationary_val) {
  df <- .dist_base(frames_df, outcome) |>
    filter(virus == virus_val, stationary == stationary_val)
  if (nrow(df) == 0) return(NULL)

  n_mice <- df |> distinct(injection, mouse_id) |> count(injection, name = "n_mice")
  n_mice_label <- paste(sprintf("%s: n=%d %s", n_mice$injection, n_mice$n_mice, ifelse(n_mice$n_mice == 1, "mouse", "mice")), collapse = ", ")

  ylabel <- .dist_outcome_labels[[outcome]]
  state_label <- ifelse(stationary_val, "rest (stationary)", "non-rest (non-stationary)")

  ggplot(df, aes(x = .data[[outcome]], fill = injection)) +
    geom_histogram(aes(y = after_stat(density)), position = "identity", alpha = 0.5, bins = 30) +
    labs(
      title = sprintf("%s distribution — %s, %s", ylabel, virus_val, state_label),
      subtitle = sprintf("DCZ vs Vehicle; frame-level, %d samples; %s", nrow(df), n_mice_label),
      x = ylabel, y = "Density", fill = "Injection"
    ) +
    theme_minimal()
}

plot_velocity_distribution_one <- function(frames_df, virus_val) {
  if (!"velocity_smooth_px_s" %in% names(frames_df)) return(NULL)
  df <- frames_df |>
    filter(injection %in% c("DCZ", "Vehicle"), virus == virus_val, !is.na(velocity_smooth_px_s)) |>
    mutate(injection = factor(injection, levels = c("Vehicle", "DCZ")))
  if (nrow(df) == 0) return(NULL)
  n_mice <- df |> distinct(injection, mouse_id) |> count(injection, name = "n_mice")
  n_mice_label <- paste(sprintf("%s: n=%d %s", n_mice$injection, n_mice$n_mice, ifelse(n_mice$n_mice == 1, "mouse", "mice")), collapse = ", ")
  ggplot(df, aes(x = velocity_smooth_px_s, fill = injection)) +
    geom_histogram(aes(y = after_stat(density)), position = "identity", alpha = 0.5, bins = 40) +
    labs(
      title = sprintf("Velocity (px/s) distribution — %s", virus_val),
      subtitle = sprintf("DCZ vs Vehicle; frame-level (Stage 7 samples), all states, %d samples\n%s", nrow(df), n_mice_label),
      x = "Velocity (px/s)", y = "Density", fill = "Injection"
    ) +
    theme_minimal()
}

build_distribution_plots <- function(frames_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  for (outcome in names(.dist_outcome_labels)) {
    for (virus_val in c("Gi", "Gq")) {
      for (stationary_val in c(TRUE, FALSE)) {
        p <- plot_distribution_one(frames_df, outcome, virus_val, stationary_val)
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
    p <- plot_velocity_distribution_one(frames_df, virus_val)
    if (is.null(p)) { message("skip (no velocity column/data): ", virus_val); next }
    ggsave(file.path(output_dir, sprintf("dist_velocity_smooth_px_s_%s.png", virus_val)), p, width = 7, height = 4.5, dpi = 150)
  }
}
