library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)

# Raw data: landmark_outputs/ at the repo root, one directory up from this
# workstream's own root -- already fully metadata-joined by
# scripts/join_landmark_metadata.py (project_brief_v8.md §3.2), unlike the
# legacy pipeline's per-file tracking CSVs that r_analysis/ has to join
# itself. No video_file -> lookup join needed here at all.
LANDMARK_DIR <- file.path("..", "landmark_outputs")
BOUT_CSV <- file.path(LANDMARK_DIR, "master_landmarks_with_metadata_bout.csv")
FRAME_CSV <- file.path(LANDMARK_DIR, "master_landmarks_with_metadata_frame.csv")
LUT_CSV <- file.path("..", "metadata", "LUT_CLEAN_July6.csv")

# ── Craniotomy lookup ────────────────────────────────────────────────────────
#
# Craniotomy is a per-(Mouse_ID, Date) LUT property, not selected by the
# Python-side join (src/metadata.py::join_landmark_metadata()) any more than
# it was by the legacy pipeline's join_metadata() -- same deliberate scope
# decision, see project_v8_metadata_join memory / project_brief_v6.md open
# question A. Ported from r_analysis/R/data.R::build_craniotomy_lookup() /
# join_craniotomy() rather than sourced from that sibling project (this
# workstream has its own renv and inputs, per project_brief_v8.md §5's
# "don't share r_analysis/'s renv" instruction -- code included). The only
# real change from the legacy version: the date is parsed from `session`
# (e.g. "07-28-25_4540_B_4541_F_Test3-004"), not `video_file` -- same
# leading MM-DD-YY token, same regex, so the parsing logic itself is
# unchanged.
build_craniotomy_lookup <- function(lut_csv = LUT_CSV) {
  lut <- read_csv(lut_csv, show_col_types = FALSE, col_types = cols(.default = "c"))
  lut |>
    transmute(
      mouse_id = str_trim(Mouse_ID),
      date = mdy(Date),
      craniotomy = str_trim(Craniotomy)
    ) |>
    filter(!is.na(mouse_id), !is.na(date), !is.na(craniotomy)) |>
    distinct()
}

extract_session_date <- function(session) {
  date_token <- str_extract(session, "^\\d{2}-\\d{2}-\\d{2}")
  as.Date(date_token, format = "%m-%d-%y")
}

# Left-joins `craniotomy` ("Pre-craniotomy" / "Post") onto any data frame
# that has `mouse_id` and `session` columns -- i.e. both load_frames() and
# load_bouts() output. Errors out (rather than silently dropping rows) if
# any row fails to match, same as the legacy version.
join_craniotomy <- function(df, lut_csv = LUT_CSV, allow_unmatched = FALSE) {
  lookup <- build_craniotomy_lookup(lut_csv)

  df <- df |>
    mutate(
      .session_date = extract_session_date(session),
      mouse_id = str_trim(as.character(mouse_id))
    ) |>
    left_join(lookup, by = c("mouse_id" = "mouse_id", ".session_date" = "date")) |>
    select(-.session_date)

  n_unmatched <- sum(is.na(df$craniotomy))
  if (n_unmatched > 0) {
    msg <- sprintf(
      "join_craniotomy: %d / %d rows failed to match a Craniotomy value (%.1f%%)",
      n_unmatched, nrow(df), 100 * n_unmatched / nrow(df)
    )
    if (allow_unmatched) {
      warning(msg)
    } else {
      stop(msg, "\nInspect with: df |> filter(is.na(craniotomy)) |> distinct(session, mouse_id)")
    }
  }

  df
}

# ── Frame-level data ─────────────────────────────────────────────────────────
#
# One row per Stage 7 sample (stationary bouts + project_brief_v8.md §3.1's
# non-stationary sampling pass). Primary input for §4.1 time-course, §4.3
# distributions, and the stationary/non-stationary splits used in the
# paired/craniotomy plot families. `stationary` is already a native boolean
# column here (unlike the legacy pipeline, which derives it via
# compute_velocity()/classify_stationary() on the R-loaded master table --
# Stage 7 already did that on the Python side, see project_v8_nonstationary_
# sampling memory).
load_frames <- function(with_craniotomy = TRUE) {
  df <- read_csv(FRAME_CSV, show_col_types = FALSE)
  if (with_craniotomy) df <- join_craniotomy(df)
  df
}

# ── Bout-level data ──────────────────────────────────────────────────────────
#
# One row per stationary bout. Primary input for §4.2.1's paired
# reproduction, §4.2.3's bout-organization plots, and the preference
# outcomes (mean_floor_temp_c, gradient_zone) -- see the OUTCOME_COLS note
# below for why floor_temp_mean_c (frame-level) must NOT be substituted for
# mean_floor_temp_c (bout-level) even though the names look interchangeable.
load_bouts <- function(with_craniotomy = TRUE) {
  df <- read_csv(BOUT_CSV, show_col_types = FALSE)
  if (with_craniotomy) df <- join_craniotomy(df)
  df
}

# ── Outcome applicability (project_brief_v8.md §1 / §4's filtering-
# conventions paragraph) ─────────────────────────────────────────────────────
#
# Each of the three feature outcomes has its OWN applicability rule, not a
# shared QC gate -- centralized here so every plot_*.R / model_timeseries.R
# script applies the identical rule instead of re-deriving it (the brief
# explicitly warns not to paper over warm-spot's sparsity by mixing up which
# gate applies to which outcome). Verified against real data 2026-09-02
# (see project_v8_metadata_join memory):
#   - dorsal (dorsal_mean_c at bout level, mouse_surface_temp_mean_c at
#     frame level -- SAME measurement, DIFFERENT column name per file; not a
#     typo) -- non-NA is the only applicability rule.
#   - warm_spot_temp_c -- non-NA alone is a COMPLETE filter: confirmed
#     non-NA exactly iff posture=="extended", nothing further needed.
#   - tail_delta_t_c -- non-NA is NOT sufficient alone: 649/3272 frame rows
#     with a computed delta_t still fail qc_valid (bout) / qc_flag=="ok"
#     (frame), so that check is required IN ADDITION to non-NA.
#
# Do NOT use frame-level floor_temp_mean_c as a "floor preference" outcome
# -- it's the LOCAL floor annulus next to the tail sample point (an input to
# tail-ΔT), not the same signal as bout-level mean_floor_temp_c (the
# whole-ROI floor mean the brief actually means by "what temperature did
# the mouse choose to rest at"). Floor-preference plots stay bout-level
# (mean_floor_temp_c, gradient_zone), never substitute the frame-level column
# just because the name is similar.
OUTCOME_COLS <- c("dorsal_mean_c", "mouse_surface_temp_mean_c", "warm_spot_temp_c", "tail_delta_t_c")

filter_for_outcome <- function(df, outcome) {
  stopifnot(outcome %in% OUTCOME_COLS)
  df <- df |> filter(!is.na(.data[[outcome]]))
  if (outcome == "tail_delta_t_c") {
    if ("qc_valid" %in% names(df)) {
      df <- df |> filter(qc_valid)
    } else if ("qc_flag" %in% names(df)) {
      df <- df |> filter(qc_flag == "ok")
    }
  }
  df
}
