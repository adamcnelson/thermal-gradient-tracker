library(dplyr)
library(ggplot2)

# Shared pieces for the DCZ-vs-Vehicle plot families (spaghetti, all_experiments): the
# post-craniotomy experimental subset, per-experiment (session x lane) summaries, and a fixed
# per-mouse color map.

DCZ_VEHICLE_OUTCOME_LABELS <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temperature (°C)",
  warm_spot_temp_c = "Warm-spot temperature (°C)",
  tail_delta_t_c = "Tail ΔT (°C)"
)

STATE_LEVELS <- c("Stationary", "Non-stationary")
COMBINED_STATE <- "Stationary + non-stationary (time-weighted)"

dcz_vehicle_subset <- function(df) {
  df |>
    filter(craniotomy == "Post", phase == "experimental", injection %in% c("DCZ", "Vehicle"),
           virus %in% c("Gi", "Gq")) |>
    mutate(mouse_id = as.character(mouse_id))
}

# Colors follow the mouse, not its position in the panel: assigned once from the full table
# (in ID order within virus), so a mouse keeps its color in every tier even when a tier drops
# other mice. Okabe-Ito hues; each virus has 4 mice, so the 4 hues repeat across viruses but
# never within one figure.
mouse_colors <- function(df) {
  hues <- c("#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
  ids <- df |> distinct(virus, mouse_id = as.character(mouse_id)) |> arrange(virus, mouse_id) |>
    group_by(virus) |> mutate(col = hues[((row_number() - 1) %% length(hues)) + 1]) |> ungroup()
  setNames(ids$col, ids$mouse_id)
}

# One row per experiment (session x lane) x stationary state: mean of the outcome's valid samples.
experiment_means_frame <- function(frames_df, outcome) {
  dcz_vehicle_subset(frames_df) |>
    filter_for_outcome(outcome) |>
    mutate(state = factor(ifelse(stationary, "Stationary", "Non-stationary"), levels = STATE_LEVELS)) |>
    group_by(virus, mouse_id, injection, nestlet, session, track, state) |>
    summarise(value = mean(.data[[outcome]]), n_samples = n(), .groups = "drop")
}

# Dorsal, stationary + non-stationary combined. Not a pooled sample mean: stationary samples are
# ~13 s apart and non-stationary 1 s apart, so pooling would just return the non-stationary mean.
# Each state's experiment mean is weighted by the time spent in it: stationary time = sum of
# bout durations (bout table); non-stationary time = number of 1 Hz non-stationary samples
# (counted before the outcome filter). If an experiment has a measured mean for only one state,
# that mean is used as-is.
experiment_means_dorsal_combined <- function(frames_df, bouts_df) {
  outcome <- "mouse_surface_temp_mean_c"
  key <- c("session", "track")
  stat_time <- dcz_vehicle_subset(bouts_df) |>
    group_by(across(all_of(key))) |>
    summarise(t_stat = sum(bout_end_thermal_sec - bout_start_thermal_sec), .groups = "drop")
  nonstat_time <- dcz_vehicle_subset(frames_df) |>
    filter(!stationary) |>
    count(across(all_of(key)), name = "t_nonstat")

  experiment_means_frame(frames_df, outcome) |>
    select(-n_samples) |>
    tidyr::pivot_wider(names_from = state, values_from = value) |>
    left_join(stat_time, by = key) |>
    left_join(nonstat_time, by = key) |>
    mutate(
      w_s = ifelse(is.na(Stationary), 0, coalesce(t_stat, 0)),
      w_n = ifelse(is.na(`Non-stationary`), 0, coalesce(t_nonstat, 0)),
      value = case_when(
        w_s + w_n > 0 ~ (w_s * coalesce(Stationary, 0) + w_n * coalesce(`Non-stationary`, 0)) / (w_s + w_n),
        TRUE ~ coalesce(Stationary, `Non-stationary`)
      ),
      state = factor(COMBINED_STATE)
    ) |>
    select(virus, mouse_id, injection, nestlet, session, track, state, value)
}

# Floor preference: one row per experiment, mean of its bouts' mean_floor_temp_c.
experiment_means_floor <- function(bouts_df) {
  dcz_vehicle_subset(bouts_df) |>
    filter(!is.na(mean_floor_temp_c)) |>
    group_by(virus, mouse_id, injection, nestlet, session, track) |>
    summarise(value = mean(mean_floor_temp_c), n_bouts = n(), .groups = "drop")
}

# Floor temperature where the mouse was while moving: median, per experiment, of
# tracking_floor_temp_c over the 1 Hz non-stationary samples. tracking_floor_temp_c is the
# tracker's per-frame floor ROI next to the mouse -- the same signal mean_floor_temp_c averages
# over a stationary bout (attached by join_landmark_metadata.py --velocity-from). Median, not
# mean: a moving mouse sweeps the gradient, so the distribution is wide and often skewed.
experiment_medians_floor_moving <- function(frames_df) {
  dcz_vehicle_subset(frames_df) |>
    filter(!stationary, !is.na(tracking_floor_temp_c)) |>
    group_by(virus, mouse_id, injection, nestlet, session, track) |>
    summarise(value = median(tracking_floor_temp_c), n_samples = n(), .groups = "drop")
}

sem <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  sd(x) / sqrt(length(x))
}
