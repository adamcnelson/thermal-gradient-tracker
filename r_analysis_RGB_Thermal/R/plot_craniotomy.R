library(dplyr)
library(ggplot2)

# Section 4.2.2 — craniotomy effect: per-mouse Pre- vs Post-craniotomy, Gi+Gq combined, split
# stationary / non-stationary (frame-level outcomes) plus bout-level floor preference.
#
# DESIGN (2026-10-08): in this corpus craniotomy is perfectly confounded with phase -- every
# Pre-craniotomy session is habituation (Saline) and every Post-craniotomy session is
# experimental (DCZ/Vehicle). The comparison used is therefore
#     Pre  = habituation, Saline        vs        Post = experimental, Vehicle
# -- both control injections, so DCZ is out of the comparison, but phase / experience / time
# since habituation still differ with craniotomy. A difference here is "craniotomy + phase",
# not craniotomy alone; every figure says so.
#
# Second contrast (2026-10-13): Post = rehabituation, Saline (recovered sessions, 6 mice) --
# the same injection on both sides, so only craniotomy + time/experience differ.

CRANIOTOMY_POST <- list(
  Vehicle = list(label = "Post (experimental, Vehicle)", file_prefix = "craniotomy_",
                 confound = "CONFOUNDED: craniotomy coincides with phase (Pre = habituation)"),
  Saline  = list(label = "Post (rehabituation, Saline)", file_prefix = "craniotomy_saline_",
                 confound = "Saline both sides; still CONFOUNDED with time/experience (Pre = habituation)")
)
CRANIOTOMY_PRE_LABEL <- "Pre (habituation, Saline)"

.cranio_subset <- function(df, post_injection = "Vehicle") {
  levels <- c(CRANIOTOMY_PRE_LABEL, CRANIOTOMY_POST[[post_injection]]$label)
  df |>
    filter(virus %in% c("Gi", "Gq"),
           (craniotomy == "Pre-craniotomy" & injection == "Saline") |
             (craniotomy == "Post" & injection == post_injection)) |>
    mutate(cond = factor(ifelse(craniotomy == "Pre-craniotomy", levels[1], levels[2]), levels = levels))
}

# state = NULL: one figure faceted by state; "Stationary" / "Non-stationary": that state only.
plot_craniotomy_one <- function(frames_df, outcome, post_injection = "Vehicle", state = NULL) {
  mm <- frames_df |>
    .cranio_subset(post_injection) |>
    filter_for_outcome(outcome) |>
    mutate(stat_label = factor(ifelse(stationary, "Stationary", "Non-stationary"),
                               levels = c("Stationary", "Non-stationary"))) |>
    filter(is.null(state) | stat_label %in% state) |>
    group_by(stat_label, cond, mouse_id) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[[outcome]]
  plot_paired_generic(mm, facet_vars = "stat_label", facet_formula = if (is.null(state)) . ~ stat_label,
                      title = paste(ylabel, if (!is.null(state)) paste("—", state), "— craniotomy effect"), ylabel = ylabel,
                      subtitle = paste0("Gi+Gq combined  |  per-mouse mean\n", CRANIOTOMY_POST[[post_injection]]$confound))
}

# Stationary + non-stationary combined, time-weighted per session (dcz_vehicle_units.R), then
# per-mouse mean of sessions.
plot_craniotomy_all_states <- function(frames_df, bouts_df, outcome, post_injection = "Vehicle") {
  lv <- c(CRANIOTOMY_PRE_LABEL, CRANIOTOMY_POST[[post_injection]]$label)
  mm <- experiment_means_all_states(frames_df, bouts_df, outcome,
                                    subset = function(d) .cranio_subset(d, post_injection)) |>
    mutate(cond = factor(ifelse(craniotomy == "Pre-craniotomy", lv[1], lv[2]), levels = lv)) |>
    group_by(cond, mouse_id) |>
    summarise(value = mean(value), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[[outcome]]
  plot_paired_generic(mm, title = paste(ylabel, "— all states — craniotomy effect"), ylabel = ylabel,
                      subtitle = paste0("Gi+Gq combined  |  stationary + non-stationary (time-weighted), per-mouse mean of sessions\n",
                                        CRANIOTOMY_POST[[post_injection]]$confound))
}

plot_craniotomy_activity <- function(activity_lanes, col, post_injection = "Vehicle") {
  spec <- ACTIVITY_OUTCOMES[[col]]
  mm <- activity_lanes |>
    .cranio_subset(post_injection) |>
    group_by(cond, mouse_id) |>
    summarise(value = mean(.data[[col]]), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  plot_paired_generic(mm, title = paste(spec$title, "— craniotomy effect"), ylabel = spec$label,
                      subtitle = paste0("Gi+Gq combined  |  per-mouse mean of sessions, tracking from arena entry\n",
                                        CRANIOTOMY_POST[[post_injection]]$confound))
}

plot_craniotomy_floor <- function(bouts_df, post_injection = "Vehicle") {
  mm <- bouts_df |>
    .cranio_subset(post_injection) |>
    filter(!is.na(mean_floor_temp_c)) |>
    group_by(cond, mouse_id) |>
    summarise(value = mean(mean_floor_temp_c, na.rm = TRUE), .groups = "drop")
  if (nrow(mm) == 0) return(NULL)
  plot_paired_generic(mm, title = "Floor temperature preference — craniotomy effect",
                      ylabel = "Mean floor temp at bout (°C)",
                      subtitle = paste0("Gi+Gq combined  |  per-mouse mean across bouts\n",
                                        CRANIOTOMY_POST[[post_injection]]$confound))
}

build_craniotomy_plots <- function(frames_df, bouts_df, output_dir, activity_lanes,
                                   outcomes = names(DCZ_VEHICLE_OUTCOME_LABELS), post_injection = "Vehicle") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- CRANIOTOMY_POST[[post_injection]]$file_prefix
  for (outcome in outcomes) for (fig in state_figures(outcome)) {
    p <- plot_craniotomy_one(frames_df, outcome, post_injection, fig$state)
    if (is.null(p)) { message("skip (no data): ", outcome); next }
    ggsave(file.path(output_dir, sprintf("%s%s%s.png", prefix, outcome, fig$suffix)), p,
           width = if (is.null(fig$state)) 9 else 6.5, height = 5.5, dpi = 150)
  }
  for (outcome in names(ALL_STATES_OUTCOMES)) {
    p <- plot_craniotomy_all_states(frames_df, bouts_df, outcome, post_injection)
    if (!is.null(p)) ggsave(file.path(output_dir, sprintf("%s%s_all_states.png", prefix, ALL_STATES_OUTCOMES[[outcome]])),
                            p, width = 7.5, height = 5.5, dpi = 150)
  }
  for (col in names(ACTIVITY_OUTCOMES)) {
    p <- plot_craniotomy_activity(activity_lanes, col, post_injection)
    if (!is.null(p)) ggsave(file.path(output_dir, paste0(prefix, ACTIVITY_OUTCOMES[[col]]$file, ".png")), p,
                            width = 6.5, height = 5.5, dpi = 150)
  }
  p <- plot_craniotomy_floor(bouts_df, post_injection)
  if (!is.null(p)) ggsave(file.path(output_dir, paste0(prefix, "floor_preference.png")), p, width = 6.5, height = 5.5, dpi = 150)
}
