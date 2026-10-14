library(dplyr)
library(tidyr)
library(ggplot2)

# Section 4.2.1 — per-mouse paired DCZ-vs-Vehicle plots, extending
# r_analysis/R/plot_spaghetti.R::plot_dcz_vehicle_paired_one() (which
# reproduces src/treatment_plots.py::plot_dcz_vehicle_paired() for the
# legacy pipeline's single mouse_surface_temp_mean outcome) to the 3
# frame-level feature outcomes + floor preference.
#
# One figure per virus x outcome (Gi and Gq are separate files), faceted by stationary state
# for the 3 frame-level outcomes, plus a dorsal figure combining both states (time-weighted,
# see dcz_vehicle_units.R). Floor preference is bout-level (a bout already IS stationary time),
# so it has no state facet. Every figure is drawn for each nestlet tier (NESTLET_TIERS) into
# its own subdirectory. Needs R/data.R and R/dcz_vehicle_units.R sourced.

# "paired mice: 4" -- mice with BOTH conditions (the ones drawn as lines), computed from the
# data rather than stated, so the label can't go stale as the corpus grows.
.paired_n_label <- function(mouse_means) {
  n <- mouse_means |>
    group_by(mouse_id) |>
    summarise(both = n_distinct(injection) == 2, .groups = "drop")
  sprintf("paired mice: %d", sum(n$both))
}

# mouse_means: one row per mouse x injection (x facet), columns mouse_id, injection, value.
.draw_spaghetti <- function(mouse_means, colors, title, subtitle, ylabel, facet = NULL) {
  mm <- mouse_means |>
    mutate(injection = factor(injection, levels = c("Vehicle", "DCZ")),
           x_pos = ifelse(injection == "Vehicle", 0, 1))
  group_summary <- mm |>
    group_by(across(all_of(c(facet, "injection", "x_pos")))) |>
    summarise(mean_value = mean(value, na.rm = TRUE), sem_value = sem(value), .groups = "drop") |>
    mutate(x_offset = x_pos + 0.22)

  p <- ggplot(mm, aes(x = x_pos, y = value)) +
    geom_line(aes(group = mouse_id, color = mouse_id), linewidth = 0.8, alpha = 0.5) +
    geom_point(aes(color = mouse_id), size = 2.5, alpha = 0.9) +
    geom_errorbar(
      data = group_summary,
      aes(x = x_offset, ymin = mean_value - sem_value, ymax = mean_value + sem_value),
      width = 0.06, color = "black", linewidth = 0.8, inherit.aes = FALSE
    ) +
    geom_point(data = group_summary, aes(x = x_offset, y = mean_value),
               shape = 23, fill = "black", color = "black", size = 3, inherit.aes = FALSE) +
    scale_x_continuous(breaks = c(0, 1), labels = c("Vehicle", "DCZ"), limits = c(-0.4, 1.55)) +
    scale_color_manual(values = colors) +
    labs(title = title, subtitle = subtitle, x = NULL, y = ylabel, color = "Mouse ID") +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
  if (!is.null(facet)) p <- p + facet_wrap(facet)
  p
}

.subtitle <- function(virus_name, tier, what, mouse_means) {
  paste0(virus_name, "  |  ", NESTLET_TIERS[[tier]], "  |  post-craniotomy, experimental phase\n",
         what, "  |  ", .paired_n_label(mouse_means))
}

plot_dcz_vehicle_paired_one <- function(frames_df, outcome, virus_name, tier, colors) {
  df <- dcz_vehicle_subset(frames_df) |>
    filter(virus == virus_name) |>
    filter_nestlet_tier(tier) |>
    filter_for_outcome(outcome) |>
    mutate(state = factor(ifelse(stationary, "Stationary", "Non-stationary"), levels = STATE_LEVELS))
  if (nrow(df) == 0) return(NULL)

  mouse_means <- df |>
    group_by(state, injection, mouse_id) |>
    summarise(value = mean(.data[[outcome]]), .groups = "drop")
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[[outcome]]
  .draw_spaghetti(mouse_means, colors, title = paste(virus_name, "—", ylabel),
                  subtitle = .subtitle(virus_name, tier, "per-mouse mean of all samples", mouse_means),
                  ylabel = ylabel, facet = "state")
}

plot_dorsal_combined_paired <- function(frames_df, bouts_df, virus_name, tier, colors) {
  mouse_means <- experiment_means_dorsal_combined(frames_df, bouts_df) |>
    filter(virus == virus_name) |>
    filter_nestlet_tier(tier) |>
    group_by(injection, mouse_id) |>
    summarise(value = mean(value), .groups = "drop")
  if (nrow(mouse_means) == 0) return(NULL)
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[["mouse_surface_temp_mean_c"]]
  .draw_spaghetti(mouse_means, colors, title = paste(virus_name, "—", ylabel, "— all states"),
                  subtitle = .subtitle(virus_name, tier,
                                       "stationary + non-stationary (time-weighted)\nper-mouse mean of session means",
                                       mouse_means),
                  ylabel = ylabel)
}

plot_floor_preference_paired <- function(bouts_df, virus_name, tier, colors) {
  mouse_means <- dcz_vehicle_subset(bouts_df) |>
    filter(virus == virus_name, !is.na(mean_floor_temp_c)) |>
    filter_nestlet_tier(tier) |>
    group_by(injection, mouse_id) |>
    summarise(value = mean(mean_floor_temp_c), .groups = "drop")
  if (nrow(mouse_means) == 0) return(NULL)
  .draw_spaghetti(mouse_means, colors, title = paste(virus_name, "— Floor temperature preference"),
                  subtitle = .subtitle(virus_name, tier, "per-mouse mean across bouts", mouse_means),
                  ylabel = "Mean floor temp at bout (°C)")
}

.save_or_skip <- function(p, path, width, height) {
  if (is.null(p)) {
    message(sprintf("skip (no data): %s", path))
    return(invisible())
  }
  ggsave(path, p, width = width, height = height, dpi = 150)
}

build_dcz_vehicle_paired_plots <- function(frames_df, bouts_df, output_dir,
                                            outcomes = names(DCZ_VEHICLE_OUTCOME_LABELS)) {
  colors <- mouse_colors(dcz_vehicle_subset(frames_df))
  for (tier in names(NESTLET_TIERS)) {
    tier_dir <- file.path(output_dir, tier)
    dir.create(tier_dir, recursive = TRUE, showWarnings = FALSE)
    for (virus_name in c("Gi", "Gq")) {
      for (outcome in outcomes) {
        .save_or_skip(plot_dcz_vehicle_paired_one(frames_df, outcome, virus_name, tier, colors),
                      file.path(tier_dir, sprintf("paired_dcz_vehicle_%s_%s.png", virus_name, outcome)), 8, 5)
      }
      .save_or_skip(plot_dorsal_combined_paired(frames_df, bouts_df, virus_name, tier, colors),
                    file.path(tier_dir, sprintf("paired_dcz_vehicle_%s_dorsal_all_states.png", virus_name)), 5.5, 5)
      .save_or_skip(plot_floor_preference_paired(bouts_df, virus_name, tier, colors),
                    file.path(tier_dir, sprintf("paired_dcz_vehicle_%s_floor_preference.png", virus_name)), 5.5, 5)
    }
  }
}
