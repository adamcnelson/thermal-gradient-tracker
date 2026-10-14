library(dplyr)
library(ggplot2)

# Shared pieces for the DCZ-vs-Vehicle plot families (spaghetti, all_experiments): the
# post-craniotomy experimental subset, per-experiment (session x lane) summaries, and a fixed
# per-mouse color map.

DCZ_VEHICLE_OUTCOME_LABELS <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temperature (°C)",
  warm_spot_temp_c = "Warm-spot temperature (°C)",
  tail_delta_t_c = "Tail ΔT (°C)",
  velocity_smooth_px_s = "Velocity (px/s)"
)

# Outcomes that also get a stationary + non-stationary combined ("all states") figure, with the
# short name used in its filename/title. Velocity's all-states figure comes from the tracking
# instead (tracking_activity.R), not from Stage 7 samples.
ALL_STATES_OUTCOMES <- c(mouse_surface_temp_mean_c = "dorsal")

STATE_LEVELS <- c("Stationary", "Non-stationary")

# Outcomes drawn as one figure per stationary state instead of one figure faceted by state:
# velocity is ~1 px/s at rest vs ~14 moving, so a shared axis flattens the stationary panel.
SPLIT_STATE_OUTCOMES <- "velocity_smooth_px_s"
STATE_FILE_SUFFIX <- c(Stationary = "stationary", `Non-stationary` = "nonstationary")

# Figures to draw for an outcome: list(state = NULL, suffix = "") for one faceted figure, or one
# entry per state for SPLIT_STATE_OUTCOMES.
state_figures <- function(outcome) {
  if (!outcome %in% SPLIT_STATE_OUTCOMES) return(list(list(state = NULL, suffix = "")))
  lapply(STATE_LEVELS, function(s) list(state = s, suffix = paste0("_", STATE_FILE_SUFFIX[[s]])))
}
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

EXPERIMENT_KEYS <- c("virus", "mouse_id", "injection", "craniotomy", "nestlet", "session", "track")

# One row per experiment (session x lane) x stationary state: mean of the outcome's valid samples.
# `subset` picks the sessions (default: post-craniotomy DCZ/Vehicle).
experiment_means_frame <- function(frames_df, outcome, subset = dcz_vehicle_subset) {
  subset(frames_df) |>
    filter_for_outcome(outcome) |>
    mutate(state = factor(ifelse(stationary, "Stationary", "Non-stationary"), levels = STATE_LEVELS)) |>
    group_by(across(all_of(c(EXPERIMENT_KEYS, "state")))) |>
    summarise(value = mean(.data[[outcome]]), n_samples = n(), .groups = "drop")
}

# Stationary + non-stationary combined (dorsal, velocity). Not a pooled sample mean: stationary
# samples are ~13 s apart and non-stationary 1 s apart, so pooling would just return the
# non-stationary mean.
# Each state's experiment mean is weighted by the time spent in it: stationary time = sum of
# bout durations (bout table); non-stationary time = number of 1 Hz non-stationary samples
# (counted before the outcome filter). If an experiment has a measured mean for only one state,
# that mean is used as-is.
experiment_means_all_states <- function(frames_df, bouts_df, outcome, subset = dcz_vehicle_subset) {
  key <- c("session", "track")
  stat_time <- subset(bouts_df) |>
    group_by(across(all_of(key))) |>
    summarise(t_stat = sum(bout_end_thermal_sec - bout_start_thermal_sec), .groups = "drop")
  nonstat_time <- subset(frames_df) |>
    filter(!stationary) |>
    count(across(all_of(key)), name = "t_nonstat")

  experiment_means_frame(frames_df, outcome, subset) |>
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
    select(all_of(EXPERIMENT_KEYS), state, value)
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
