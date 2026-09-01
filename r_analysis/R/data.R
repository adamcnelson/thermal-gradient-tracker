library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)

# Raw data lives one directory up from the repo root (see project_brief_v6.md section 1).
RAW_DATA_DIR <- normalizePath(
  file.path("..", "..", "SLURM_RESULTS", "results_fullrun_mgms2_2026-07-28", "bouts"),
  mustWork = FALSE
)
MASTER_CSV <- file.path(RAW_DATA_DIR, "master_tracking_with_metadata.csv")
BOUT_CSV <- file.path(RAW_DATA_DIR, "bout_table.csv")
LUT_CSV <- file.path("..", "metadata", "LUT_CLEAN_July6.csv")

# ── Craniotomy lookup (open question A: R-side join) ────────────────────────
#
# Craniotomy is a per-(Mouse_ID, Date) property in the LUT, not currently
# selected by src/metadata.py's join_metadata(). master_tracking_with_metadata.csv
# already carries mouse_id per row (from the Python join), so we only need to
# extract the date out of video_file and join on (mouse_id, date) — no need to
# re-derive mouse_id from the filename the way the Python-side LUT matching does.
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

# video_file looks like '07-28-25_4540_B_4541_F_Test3-004_Front.seq' — the
# leading token is the recording date as MM-DD-YY.
extract_video_date <- function(video_file) {
  date_token <- str_extract(video_file, "^\\d{2}-\\d{2}-\\d{2}")
  as.Date(date_token, format = "%m-%d-%y")
}

# Left-joins `craniotomy` (values "Pre-craniotomy" / "Post") onto any data
# frame that has `mouse_id` and `video_file` columns. Errors out (rather than
# silently dropping rows) if any row fails to match — per the brief, join
# coverage should be spot-checked, not assumed.
join_craniotomy <- function(df, lut_csv = LUT_CSV, allow_unmatched = FALSE) {
  lookup <- build_craniotomy_lookup(lut_csv)

  df <- df |>
    mutate(
      .video_date = extract_video_date(video_file),
      mouse_id = str_trim(as.character(mouse_id))
    ) |>
    left_join(lookup, by = c("mouse_id" = "mouse_id", ".video_date" = "date")) |>
    select(-.video_date)

  n_unmatched <- sum(is.na(df$craniotomy))
  if (n_unmatched > 0) {
    msg <- sprintf(
      "join_craniotomy: %d / %d rows failed to match a Craniotomy value (%.1f%%)",
      n_unmatched, nrow(df), 100 * n_unmatched / nrow(df)
    )
    if (allow_unmatched) {
      warning(msg)
    } else {
      stop(msg, "\nInspect with: df |> filter(is.na(craniotomy)) |> distinct(video_file, mouse_id)")
    }
  }

  df
}

# ── Master (frame-level) data ────────────────────────────────────────────────

# Loads master_tracking_with_metadata.csv, keeps qc_flag == "ok" rows only
# (per the brief: invalid/no-data frames should never be silently treated as
# zero or dropped without comment — this is the one explicit, documented filter),
# and left-joins Craniotomy status.
load_master <- function(qc_ok_only = TRUE, with_craniotomy = TRUE) {
  df <- read_csv(MASTER_CSV, show_col_types = FALSE)

  if (qc_ok_only) {
    n_before <- nrow(df)
    df <- df |> filter(qc_flag == "ok")
    message(sprintf(
      "load_master: kept %d / %d rows with qc_flag == 'ok' (dropped %d)",
      nrow(df), n_before, n_before - nrow(df)
    ))
  }

  if (with_craniotomy) {
    df <- join_craniotomy(df)
  }

  df
}

# ── Bout-level data ──────────────────────────────────────────────────────────

# bout_table.csv has video_file but no mouse_id/virus/injection/phase/craniotomy
# (open question C). Build a deduplicated video_file -> metadata lookup from the
# master data and join it in.
video_metadata_lookup <- function(master_df) {
  master_df |>
    distinct(video_file, mouse_id, virus, injection, phase, craniotomy)
}

load_bouts <- function(master_df = NULL) {
  if (is.null(master_df)) {
    master_df <- load_master()
  }
  lookup <- video_metadata_lookup(master_df)

  bouts <- read_csv(BOUT_CSV, show_col_types = FALSE)

  n_before <- nrow(bouts)
  bouts <- bouts |> left_join(lookup, by = "video_file")
  n_unmatched <- sum(is.na(bouts$mouse_id))
  if (n_unmatched > 0) {
    stop(sprintf(
      "load_bouts: %d / %d bout rows failed to match video-level metadata",
      n_unmatched, n_before
    ))
  }

  bouts
}
