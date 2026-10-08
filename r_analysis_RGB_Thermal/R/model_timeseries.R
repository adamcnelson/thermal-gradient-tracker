library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(mgcv)

# Section 4.4 — time-series models: do the DCZ and Vehicle trajectories diverge over the trial,
# not just differ in their marginal means? Per outcome, per virus, post-craniotomy experimental
# sessions (DCZ vs Vehicle).
#
# LMM (primary):  value ~ time_c * injection + (1 + time_c | mouse_id) + (1 | session_lane)
#   time_c = minutes since start, centred at TIME_CENTER_MIN, so
#     injectionDCZ        = DCZ - Vehicle difference at mid-session
#     time_c              = Vehicle slope (units/min)
#     time_c:injectionDCZ = DCZ - Vehicle slope difference  <- the divergence test
#   If the random-slope model is singular or fails to converge it is refitted with random
#   intercepts only; the `random_effects` column records which model was used.
# GAMM (non-linear check): value ~ injection + s(time) + s(time, by = injection) + random
#   intercepts for mouse and session; the by-smooth (ordered factor) is the DCZ-minus-Vehicle
#   difference curve, so its p-value tests whether the trajectory SHAPE differs.
#
# Unit of analysis -- a deliberate departure from fitting raw ~1 Hz frame samples (brief §4.4):
# frame-level outcomes are first averaged per session x lane x BIN_MIN-minute bin. Consecutive
# frames are strongly autocorrelated; fitting them directly would make every p-value tiny.
# Floor preference is bout-level already (one row per bout, time = bout start).

BIN_MIN <- 1
TIME_CENTER_MIN <- 15

.frame_outcome_labels <- c(
  mouse_surface_temp_mean_c = "Dorsal surface temp (°C)",
  warm_spot_temp_c = "Warm-spot temp (°C)",
  tail_delta_t_c = "Tail ΔT (°C)"
)

.post_exp <- function(df, virus_val) {
  df |>
    filter(craniotomy == "Post", phase == "experimental", injection %in% c("DCZ", "Vehicle"),
           virus == virus_val) |>
    mutate(injection = factor(injection, levels = c("Vehicle", "DCZ")),
           mouse_id = factor(mouse_id),
           session_lane = factor(paste(session, track, sep = "_")))
}

# Model-ready rows: one per session x lane x time bin (frame outcomes) or per bout (floor).
model_data <- function(frames_df, bouts_df, virus_val, outcome) {
  if (outcome == "mean_floor_temp_c") {
    return(.post_exp(bouts_df, virus_val) |>
             filter(!is.na(mean_floor_temp_c)) |>
             transmute(mouse_id, session_lane, injection, value = mean_floor_temp_c,
                       time_min = bout_start_thermal_sec / 60))
  }
  .post_exp(frames_df, virus_val) |>
    filter_for_outcome(outcome) |>
    mutate(time_min = (floor(elapsed_time_thermal_sec / 60 / BIN_MIN) + 0.5) * BIN_MIN) |>
    group_by(mouse_id, session_lane, injection, time_min) |>
    summarise(value = mean(.data[[outcome]], na.rm = TRUE), n_samples = n(), .groups = "drop")
}

.fit_lmm <- function(d) {
  d <- d |> mutate(time_c = time_min - TIME_CENTER_MIN)
  fits <- list(
    "(1 + time_c | mouse_id) + (1 | session_lane)" =
      value ~ time_c * injection + (1 + time_c | mouse_id) + (1 | session_lane),
    "(1 | mouse_id) + (1 | session_lane)" =
      value ~ time_c * injection + (1 | mouse_id) + (1 | session_lane)
  )
  for (re in names(fits)) {
    warn <- character(0)
    fit <- withCallingHandlers(
      tryCatch(lmerTest::lmer(fits[[re]], data = d, REML = TRUE), error = function(e) NULL),
      warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") })
    if (is.null(fit)) next
    if (!isSingular(fit) && length(warn) == 0) return(list(fit = fit, re = re, note = ""))
    if (re == names(fits)[length(fits)]) {
      return(list(fit = fit, re = re, note = paste(c(if (isSingular(fit)) "singular fit", warn), collapse = "; ")))
    }
  }
  NULL
}

.fit_gamm <- function(d) {
  d <- d |> mutate(inj_o = factor(injection, levels = c("Vehicle", "DCZ"), ordered = TRUE))
  contrasts(d$inj_o) <- "contr.treatment"
  tryCatch(
    gam(value ~ inj_o + s(time_min, k = 6) + s(time_min, by = inj_o, k = 6) +
          s(mouse_id, bs = "re") + s(session_lane, bs = "re"),
        data = d, method = "REML"),
    error = function(e) NULL)
}

fit_timeseries_models <- function(frames_df, bouts_df,
                                   outcomes = c(names(.frame_outcome_labels), "mean_floor_temp_c"),
                                   viruses = c("Gi", "Gq")) {
  lmm_rows <- list(); gamm_rows <- list(); fits <- list()
  for (outcome in outcomes) for (virus_val in viruses) {
    d <- model_data(frames_df, bouts_df, virus_val, outcome)
    base <- tibble(outcome = outcome, virus = virus_val, n_rows = nrow(d),
                   n_mice = n_distinct(d$mouse_id), n_sessions = n_distinct(d$session_lane))
    if (nrow(d) < 20 || n_distinct(d$injection) < 2 || n_distinct(d$mouse_id) < 3) {
      lmm_rows[[length(lmm_rows) + 1]] <- mutate(base, term = "not fitted", note = "too little data")
      next
    }
    lm_res <- .fit_lmm(d)
    if (is.null(lm_res)) {
      lmm_rows[[length(lmm_rows) + 1]] <- mutate(base, term = "fit failed")
    } else {
      fe <- broom.mixed::tidy(lm_res$fit, effects = "fixed", conf.int = TRUE) |>
        filter(term != "(Intercept)") |>
        transmute(term = recode(term, injectionDCZ = "DCZ - Vehicle at mid-session",
                                time_c = "Vehicle slope (per min)",
                                `time_c:injectionDCZ` = "slope difference DCZ - Vehicle (per min)"),
                  estimate, std.error, df, p.value, conf.low, conf.high)
      lmm_rows[[length(lmm_rows) + 1]] <- bind_cols(base[rep(1, nrow(fe)), ], fe) |>
        mutate(random_effects = lm_res$re, note = lm_res$note)
      fits[[paste(outcome, virus_val)]] <- list(data = d, fit = lm_res$fit)
    }
    g <- .fit_gamm(d)
    if (!is.null(g)) {
      st <- summary(g)$s.table
      diff_row <- grep("inj_o", rownames(st))
      pt <- summary(g)$p.table
      gamm_rows[[length(gamm_rows) + 1]] <- mutate(base,
        dcz_minus_vehicle = pt["inj_oDCZ", "Estimate"], dcz_minus_vehicle_p = pt["inj_oDCZ", "Pr(>|t|)"],
        difference_smooth_edf = st[diff_row, "edf"], difference_smooth_F = st[diff_row, "F"],
        difference_smooth_p = st[diff_row, "p-value"])
    }
  }
  list(lmm = bind_rows(lmm_rows), gamm = bind_rows(gamm_rows), fits = fits)
}

# Binned group means +/- SEM (per mouse first) with the LMM's population-level fitted lines.
plot_timeseries_fit <- function(fits, outcome, ylabel, bin_plot_min = 2) {
  pts <- list(); lines <- list()
  for (key in names(fits)) {
    parts <- strsplit(key, " ")[[1]]
    if (parts[1] != outcome) next
    d <- fits[[key]]$data; fit <- fits[[key]]$fit
    pts[[key]] <- d |>
      mutate(tb = (floor(time_min / bin_plot_min) + 0.5) * bin_plot_min) |>
      group_by(injection, mouse_id, tb) |> summarise(v = mean(value), .groups = "drop") |>
      group_by(injection, tb) |>
      summarise(mean_v = mean(v), sem_v = if (n() > 1) sd(v) / sqrt(n()) else 0, n_mice = n(), .groups = "drop") |>
      filter(n_mice >= 2) |> mutate(virus = parts[2])
    grid <- expand.grid(time_min = seq(min(d$time_min), max(d$time_min), length.out = 60),
                        injection = factor(c("Vehicle", "DCZ"), levels = c("Vehicle", "DCZ")))
    grid$time_c <- grid$time_min - TIME_CENTER_MIN
    grid$pred <- predict(fit, newdata = grid, re.form = NA)
    lines[[key]] <- mutate(grid, virus = parts[2])
  }
  if (length(pts) == 0) return(NULL)
  pts <- bind_rows(pts); lines <- bind_rows(lines)
  ggplot() +
    geom_errorbar(data = pts, aes(x = tb, ymin = mean_v - sem_v, ymax = mean_v + sem_v, color = injection),
                  width = 0, alpha = 0.4) +
    geom_point(data = pts, aes(x = tb, y = mean_v, color = injection), size = 1.4, alpha = 0.8) +
    geom_line(data = lines, aes(x = time_min, y = pred, color = injection), linewidth = 1.1) +
    facet_wrap(~virus, scales = "free_y") +
    labs(title = paste(ylabel, "— time-series model"),
         subtitle = sprintf("Points: %d-min bin means ± SEM across mice (bins with ≥2 mice)  |  lines: LMM fixed-effect fit",
                            bin_plot_min),
         x = "Elapsed time (thermal clock, min)", y = ylabel, color = "Injection") +
    theme_minimal()
}

build_timeseries_models <- function(frames_df, bouts_df, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  res <- fit_timeseries_models(frames_df, bouts_df)
  readr::write_csv(res$lmm, file.path(output_dir, "timeseries_lmm.csv"))
  readr::write_csv(res$gamm, file.path(output_dir, "timeseries_gamm.csv"))
  labels <- c(.frame_outcome_labels, mean_floor_temp_c = "Floor temp at bout (°C)")
  for (outcome in names(labels)) {
    p <- plot_timeseries_fit(res$fits, outcome, labels[[outcome]])
    if (!is.null(p)) ggsave(file.path(output_dir, sprintf("timeseries_%s.png", outcome)), p,
                            width = 9, height = 4.5, dpi = 150)
  }
  res
}
