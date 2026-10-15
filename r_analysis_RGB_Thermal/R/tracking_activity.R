library(readr)
library(dplyr)

# Activity outcomes -- distance travelled and whole-session ("all states") velocity -- computed
# from the thermal tracking itself (~1 Hz, every frame) rather than from Stage 7's samples.
# Stage 7 samples cover only ~70% of a session: QC-invalid frames (~10%) and stationary stretches
# too short to be bouts (~13-16%) are in neither its bout samples nor its non-stationary pass, and
# the split differs between habituation and experimental sessions, so a session-level velocity
# reconstructed from Stage 7 samples is biased (2026-10-14). Per-state Stage 7 velocity (the
# Stationary / Non-stationary panels) is unaffected and still comes from the frame table.
#
# Distance: sum of centroid steps between consecutive valid frames, from arena entry to the end
# of the lane (or of the analysis window, data.R::ANALYSIS_WINDOW_SEC).
#   - valid frame: qc_flag == "ok" and mouse_roi_valid (compute_velocity()'s rule)
#   - steps < DISTANCE_JITTER_PX are dropped: centroid jitter at rest (stationary steps: median
#     1.0 px, 75th pct 2.3 px; moving: median 10.6 px). Insensitive: 3 -> 8 px lowers totals 8%.
#   - a step spanning a QC dropout <= DISTANCE_MAX_GAP_SEC is kept (straight line across it);
#     longer gaps are skipped and their time reported as gap_s per lane (~5% of time overall)
#   - 1 Hz sampling makes it a lower bound on path length. Units: thermal px.
# Velocity: velocity_smooth_px_s (compute_velocity()'s smoothed speed) averaged over valid frames.
# Smoothing makes it ~25% lower than raw step speed (distance / time), so the two aren't
# interchangeable in absolute terms.

DISTANCE_JITTER_PX <- 3
DISTANCE_MAX_GAP_SEC <- 5
DISTANCE_LABEL <- "Distance travelled (thermal px)"
VELOCITY_LABEL <- "Velocity (px/s)"

# The data version's tracking masters (R/data_versions.R), in that version's join order: a lane
# present in several is taken from the last.
if (!exists("DATA_INPUTS")) source("R/data_versions.R")
TRACKING_CSVS <- DATA_INPUTS$tracking_csvs

load_tracking <- function(paths = TRACKING_CSVS) {
  cols <- cols_only(video_file = "c", elapsed_time_sec = "d", mouse_centroid_x = "d",
                    mouse_centroid_y = "d", qc_flag = "c", mouse_roi_valid = "c",
                    velocity_smooth_px_s = "d")
  parts <- lapply(paths, read_csv, col_types = cols, show_col_types = FALSE)
  for (i in seq_along(parts)) {
    later <- unique(unlist(lapply(parts[-seq_len(i)], `[[`, "video_file")))
    parts[[i]] <- parts[[i]] |> filter(!video_file %in% later)
  }
  bind_rows(parts) |>
    mutate(session = sub("_(Front|Back)\\.seq$", "", video_file),
           track = ifelse(grepl("_Front\\.seq$", video_file), "F", "B")) |>
    select(-video_file)
}

# Lane metadata from the (already windowed) frame table, so activity uses exactly the lanes the
# other outcomes use.
lane_info <- function(frames_df) {
  frames_df |>
    distinct(session, track, entry_time_thermal_sec, mouse_id = as.character(mouse_id), virus,
             injection, phase, craniotomy, nestlet)
}

# Valid tracking frames inside [entry, entry + window] for the given lanes, with the step from the
# previous valid frame (0 if below the jitter threshold or across a long gap).
tracking_frames <- function(tracking, lanes) {
  tracking |>
    inner_join(lanes |> distinct(session, track, entry_time_thermal_sec), by = c("session", "track")) |>
    filter(elapsed_time_sec >= entry_time_thermal_sec,
           elapsed_time_sec <= entry_time_thermal_sec + ANALYSIS_WINDOW_SEC,
           qc_flag == "ok", mouse_roi_valid == "True") |>
    arrange(session, track, elapsed_time_sec) |>
    group_by(session, track) |>
    mutate(dt = elapsed_time_sec - lag(elapsed_time_sec),
           step = sqrt((mouse_centroid_x - lag(mouse_centroid_x))^2 +
                         (mouse_centroid_y - lag(mouse_centroid_y))^2)) |>
    ungroup() |>
    mutate(counted = !is.na(dt) & dt <= DISTANCE_MAX_GAP_SEC,
           step = ifelse(counted & step >= DISTANCE_JITTER_PX, step, 0),
           time_since_entry_s = elapsed_time_sec - entry_time_thermal_sec)
}

# Per lane: total distance, mean velocity, time covered, time lost to long gaps.
lane_activity <- function(tf, lanes) {
  tf |>
    group_by(session, track) |>
    summarise(distance_px = sum(step),
              velocity_px_s = mean(velocity_smooth_px_s, na.rm = TRUE),
              covered_s = max(time_since_entry_s),
              gap_s = sum(dt[!counted & !is.na(dt)]), .groups = "drop") |>
    inner_join(lanes, by = c("session", "track"))
}

# Per lane x 1-min bin since entry: distance and mean velocity in the bin, cumulative distance at
# its end. Only bins the lane's tracking reaches the end of (within 1 s), so a partial last
# minute doesn't pull the cumulative mean down.
lane_activity_bins <- function(tf, lanes, bin_s = 60) {
  tf |>
    mutate(bin_end_s = pmax(ceiling(time_since_entry_s / bin_s), 1) * bin_s) |>  # entry frame -> minute 1
    group_by(session, track) |>
    filter(bin_end_s <= max(time_since_entry_s) + 1) |>
    group_by(session, track, bin_end_s) |>
    summarise(distance_px = sum(step), velocity_px_s = mean(velocity_smooth_px_s, na.rm = TRUE),
              .groups = "drop") |>
    group_by(session, track) |>
    arrange(bin_end_s, .by_group = TRUE) |>
    mutate(cumulative_px = cumsum(distance_px)) |>
    ungroup() |>
    inner_join(lanes, by = c("session", "track"))
}

# list(frames = valid tracking frames + metadata, lanes = per-lane totals, bins = 1-min bins).
load_activity <- function(frames_df) {
  lanes <- lane_info(frames_df)
  tf <- tracking_frames(load_tracking(), lanes)
  list(frames = tf |> inner_join(lanes |> select(-entry_time_thermal_sec), by = c("session", "track")),
       lanes = lane_activity(tf, lanes),
       bins = lane_activity_bins(tf, lanes))
}

# The two whole-session activity outcomes: column in load_activity()$lanes -> label / file name.
ACTIVITY_OUTCOMES <- list(
  distance_px = list(label = DISTANCE_LABEL, title = "Distance travelled", file = "distance_travelled"),
  velocity_px_s = list(label = VELOCITY_LABEL, title = "Velocity, all states", file = "velocity_all_states")
)
