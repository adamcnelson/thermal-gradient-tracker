library(dplyr)
library(ggplot2)

# DCZ vs Vehicle, one dot per experiment (session x lane) rather than one per mouse: the
# spaghetti plots show within-mouse change, these show the repeated measures behind it.
# No connecting lines; dots jittered and colored by mouse; mean ± SEM across experiments drawn
# beside each cloud. DCZ left, Vehicle right. One figure per virus x outcome x nestlet tier,
# same set as plot_spaghetti.R. Needs R/data.R and R/dcz_vehicle_units.R sourced.
#
# The SEM treats experiments as independent, which they aren't (3-4 per mouse) -- it describes
# the spread of the dots, it is not an inferential error bar.

.X_LEVELS <- c("DCZ", "Vehicle")

.draw_all_experiments <- function(exp_means, colors, title, subtitle, ylabel, facet = NULL) {
  em <- exp_means |>
    mutate(injection = factor(injection, levels = .X_LEVELS),
           x_pos = as.numeric(injection) - 1)
  group_summary <- em |>
    group_by(across(all_of(c(facet, "injection", "x_pos")))) |>
    summarise(mean_value = mean(value), sem_value = sem(value), n = n(), .groups = "drop") |>
    mutate(x_offset = x_pos + 0.3)
  n_labels <- group_summary |> mutate(label = sprintf("n=%d", n))

  p <- ggplot(em, aes(x = x_pos, y = value)) +
    geom_point(aes(color = mouse_id), size = 2.5, alpha = 0.85,
               position = position_jitter(width = 0.1, height = 0, seed = 1)) +
    geom_errorbar(data = group_summary,
                  aes(x = x_offset, ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.06, color = "black", linewidth = 0.8, inherit.aes = FALSE) +
    geom_point(data = group_summary, aes(x = x_offset, y = mean_value),
               shape = 23, fill = "black", color = "black", size = 3, inherit.aes = FALSE) +
    geom_text(data = n_labels, aes(x = x_pos, label = label), y = -Inf, vjust = -0.5,
              size = 3, color = "grey30", inherit.aes = FALSE) +
    scale_x_continuous(breaks = c(0, 1), labels = .X_LEVELS, limits = c(-0.35, 1.5)) +
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.05))) +
    scale_color_manual(values = colors) +
    labs(title = title, subtitle = subtitle, x = NULL, y = ylabel, color = "Mouse ID") +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
  if (!is.null(facet)) p <- p + facet_wrap(facet)
  p
}

.exp_subtitle <- function(virus_name, tier, what, exp_means) {
  paste0(virus_name, "  |  ", NESTLET_TIERS[[tier]], "  |  post-craniotomy, experimental phase\n",
         what, "  |  ", n_distinct(exp_means$mouse_id), " mice\nblack = mean ± SEM across experiments")
}

.tiered <- function(df, virus_name, tier) df |> filter(virus == virus_name) |> filter_nestlet_tier(tier)

plot_all_experiments_one <- function(frames_df, outcome, virus_name, tier, colors) {
  em <- .tiered(experiment_means_frame(frames_df, outcome), virus_name, tier)
  if (nrow(em) == 0) return(NULL)
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[[outcome]]
  .draw_all_experiments(em, colors, title = paste(virus_name, "—", ylabel, "— each experiment"),
                        subtitle = .exp_subtitle(virus_name, tier, "one dot = session mean", em),
                        ylabel = ylabel, facet = "state")
}

plot_all_experiments_dorsal_combined <- function(frames_df, bouts_df, virus_name, tier, colors) {
  em <- .tiered(experiment_means_dorsal_combined(frames_df, bouts_df), virus_name, tier)
  if (nrow(em) == 0) return(NULL)
  ylabel <- DCZ_VEHICLE_OUTCOME_LABELS[["mouse_surface_temp_mean_c"]]
  .draw_all_experiments(em, colors, title = paste(virus_name, "— Dorsal, all states — each experiment"),
                        subtitle = .exp_subtitle(virus_name, tier,
                                                 "one dot = session mean,\nstationary + non-stationary (time-weighted)", em),
                        ylabel = ylabel)
}

plot_all_experiments_floor <- function(bouts_df, virus_name, tier, colors) {
  em <- .tiered(experiment_means_floor(bouts_df), virus_name, tier)
  if (nrow(em) == 0) return(NULL)
  .draw_all_experiments(em, colors, title = paste(virus_name, "— Floor preference — each experiment"),
                        subtitle = .exp_subtitle(virus_name, tier, "one dot = session mean across bouts", em),
                        ylabel = "Mean floor temp at bout (°C)")
}

build_all_experiments_plots <- function(frames_df, bouts_df, output_dir,
                                        outcomes = names(DCZ_VEHICLE_OUTCOME_LABELS)) {
  colors <- mouse_colors(dcz_vehicle_subset(frames_df))
  save <- function(p, path, width, height) {
    if (is.null(p)) return(message(sprintf("skip (no data): %s", path)))
    ggsave(path, p, width = width, height = height, dpi = 150)
  }
  for (tier in names(NESTLET_TIERS)) {
    tier_dir <- file.path(output_dir, tier)
    dir.create(tier_dir, recursive = TRUE, showWarnings = FALSE)
    for (virus_name in c("Gi", "Gq")) {
      for (outcome in outcomes) {
        save(plot_all_experiments_one(frames_df, outcome, virus_name, tier, colors),
             file.path(tier_dir, sprintf("experiments_dcz_vehicle_%s_%s.png", virus_name, outcome)), 8, 5)
      }
      save(plot_all_experiments_dorsal_combined(frames_df, bouts_df, virus_name, tier, colors),
           file.path(tier_dir, sprintf("experiments_dcz_vehicle_%s_dorsal_all_states.png", virus_name)), 5.5, 5)
      save(plot_all_experiments_floor(bouts_df, virus_name, tier, colors),
           file.path(tier_dir, sprintf("experiments_dcz_vehicle_%s_floor_preference.png", virus_name)), 5.5, 5)
    }
  }
}
