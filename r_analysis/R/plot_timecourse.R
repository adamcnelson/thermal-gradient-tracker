library(dplyr)
library(ggplot2)

# Section 3.1 — time-course plots (frame-level).
#
# Time alignment (open question D): trial durations vary across videos
# (~950s to ~4431s in this run), so raw elapsed_time_sec isn't a shared grid
# across mice. We bin elapsed_time_sec into `bin_width`-second intervals,
# take each mouse's per-bin mean first (so a mouse with more sampled frames
# in a bin doesn't dominate), then average those per-mouse bin means across
# mice per injection to get the group mean +/- SEM trace. SEM is 0 when only
# one mouse contributes to a bin (matches the sem()-with-n==1 convention used
# in src/treatment_plots.py).
TIMECOURSE_BIN_WIDTH_SEC <- 30

.sem <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}

.outcome_labels <- c(
  mouse_surface_temp_mean = "Mouse surface temp (°C)",
  floor_temp_mean = "Floor temp (°C)",
  mouse_minus_floor_temp_mean = "Mouse − floor temp (°C)",
  velocity_smooth_px_s = "Velocity (px/s)"
)

# Filters master_df to one virus and one craniotomy status, applying the
# section-3.1 exclusion rules:
#  - drop rows with no resolved craniotomy status
#  - for Post-craniotomy, restrict to injection %in% c("DCZ", "Vehicle")
#    (this also excludes any Saline "rehabituation" trials, open question B —
#    none are present in this run, but the filter is written to hold if any
#    ever are)
filter_for_timecourse <- function(master_df, virus_val, craniotomy_val) {
  df <- master_df |>
    filter(virus == virus_val, craniotomy == craniotomy_val)
  if (craniotomy_val == "Post") {
    df <- df |> filter(injection %in% c("DCZ", "Vehicle"))
  }
  df
}

# One figure: per-mouse thin traces + group mean +/- SEM ribbon, colored by
# injection, for a single (virus, craniotomy, outcome) combination.
plot_timecourse_one <- function(master_df, virus_val, craniotomy_val, outcome,
                                 bin_width = TIMECOURSE_BIN_WIDTH_SEC) {
  df <- filter_for_timecourse(master_df, virus_val, craniotomy_val) |>
    filter(!is.na(.data[[outcome]])) |>
    mutate(time_bin = floor(elapsed_time_sec / bin_width) * bin_width)

  if (nrow(df) == 0) {
    return(NULL)
  }

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
  craniotomy_label <- ifelse(craniotomy_val == "Post", "post-craniotomy", "pre-craniotomy")

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
      title = sprintf("%s — %s, %s", ylabel, virus_val, craniotomy_label),
      subtitle = sprintf(
        "Per-mouse traces (thin) + group mean ± SEM (%ds bins); qc_flag == 'ok'",
        bin_width
      ),
      x = "Elapsed time (s)", y = ylabel, color = "Injection", fill = "Injection"
    ) +
    theme_minimal()
}

# Builds all 3.1 figures (4 outcomes x 2 viruses x 2 craniotomy states = 16
# figures) and writes them to output_dir.
build_timecourse_plots <- function(master_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  outcomes <- names(.outcome_labels)
  viruses <- c("Gi", "Gq")
  craniotomy_states <- c("Pre-craniotomy" = "pre", "Post" = "post")

  for (virus_val in viruses) {
    for (craniotomy_val in names(craniotomy_states)) {
      for (outcome in outcomes) {
        p <- plot_timecourse_one(master_df, virus_val, craniotomy_val, outcome)
        if (is.null(p)) {
          message(sprintf(
            "skip (no data): %s / %s / %s", outcome, virus_val, craniotomy_val
          ))
          next
        }
        fname <- sprintf(
          "timecourse_%s_%s_%s.png",
          outcome, virus_val, craniotomy_states[[craniotomy_val]]
        )
        ggsave(file.path(output_dir, fname), p, width = 8, height = 5, dpi = 150)
      }
    }
  }
}
