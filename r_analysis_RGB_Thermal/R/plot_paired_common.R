library(dplyr)
library(ggplot2)

# Shared drawing + test for the paired (two-condition, per-mouse) plot families -- §4.2.2
# craniotomy and §4.2.3 bout organization. Same visual language as plot_spaghetti.R (§4.2.1):
# colored per-mouse lines, black diamond = group mean ± SE offset to the right.
#
# Each panel is annotated with a paired t-test on the per-mouse values (mice with both
# conditions only), the same test as src/stats_models.py::paired_t_summary on the Python side.
# n = mice, so power is low: the CI is shown with the p-value.

paired_t_label <- function(wide, cond_a, cond_b) {
  d <- wide[[cond_b]] - wide[[cond_a]]
  d <- d[!is.na(d)]
  n <- length(d)
  if (n < 2 || sd(d) == 0) return(sprintf("n=%d paired mice", n))
  tt <- t.test(d)
  # "right - left" rather than the condition names: those are on the x-axis, and long names
  # (the craniotomy contrast) overflow the panel.
  sprintf("Δ right−left %+.2f [%+.2f, %+.2f]\npaired t p=%.3g, n=%d",
          mean(d), tt$conf.int[1], tt$conf.int[2], tt$p.value, n)
}

# mouse_means: one row per mouse x condition (x panel), columns mouse_id, cond (factor with 2
# levels, left -> right), value, plus any facet columns named in facet_vars.
plot_paired_generic <- function(mouse_means, facet_vars = character(0), title, ylabel, subtitle,
                                 facet_formula = NULL) {
  lv <- levels(mouse_means$cond)
  stopifnot(length(lv) == 2)
  mm <- mouse_means |> mutate(x_pos = ifelse(cond == lv[1], 0, 1), mouse_id = as.character(mouse_id))

  group_summary <- mm |>
    group_by(across(all_of(c(facet_vars, "cond", "x_pos")))) |>
    summarise(mean_value = mean(value, na.rm = TRUE),
              sem_value = if (sum(!is.na(value)) > 1) sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))) else 0,
              .groups = "drop") |>
    mutate(x_offset = x_pos + 0.22)

  labels <- mm |>
    group_by(across(all_of(facet_vars))) |>
    group_modify(function(d, key) {
      wide <- tidyr::pivot_wider(d, id_cols = mouse_id, names_from = cond, values_from = value)
      if (!all(lv %in% names(wide))) return(tibble(label = "one condition only"))
      tibble(label = paired_t_label(wide, lv[1], lv[2]))
    }) |>
    ungroup()

  p <- ggplot(mm, aes(x = x_pos, y = value)) +
    geom_line(aes(group = mouse_id, color = mouse_id), linewidth = 0.8, alpha = 0.5) +
    geom_point(aes(color = mouse_id), size = 2.5, alpha = 0.9) +
    geom_errorbar(data = group_summary,
                  aes(x = x_offset, ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.06, color = "black", linewidth = 0.8, inherit.aes = FALSE) +
    geom_point(data = group_summary, aes(x = x_offset, y = mean_value),
               shape = 23, fill = "black", color = "black", size = 3, inherit.aes = FALSE) +
    geom_text(data = labels, aes(label = label), x = -0.38, y = Inf, hjust = 0, vjust = 1.2,
              size = 2.8, inherit.aes = FALSE, lineheight = 0.9) +
    scale_x_continuous(breaks = c(0, 1), labels = lv, limits = c(-0.4, 1.55)) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.25))) +
    labs(title = title, subtitle = subtitle, x = NULL, y = ylabel, color = "Mouse ID") +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines"))
  if (!is.null(facet_formula)) p <- p + facet_grid(facet_formula)
  p
}
