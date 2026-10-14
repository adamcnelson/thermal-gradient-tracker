# Section 4.4 — time-series models (LMM + GAMM check), DCZ vs Vehicle trajectories. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/tracking_activity.R")
source("R/model_timeseries.R")

bouts <- load_bouts()
frames <- load_frames()
res <- build_timeseries_models(frames, bouts, output_dir = file.path(output_root(), "models"),
                               activity = load_activity(frames))

cat("\nLMM fixed effects (DCZ - Vehicle; slope difference = trajectory divergence):\n")
print(as.data.frame(res$lmm[, intersect(c("outcome", "virus", "term", "estimate", "conf.low", "conf.high",
                                          "p.value", "random_effects", "note"), names(res$lmm))]), digits = 3)
cat("\nGAMM difference smooth (does the trajectory SHAPE differ?):\n")
print(as.data.frame(res$gamm), digits = 3)
cat("\nWhole-session activity from tracking (LMM, random intercept per mouse):\n")
print(as.data.frame(res$activity), digits = 3)
cat("\nDone. Tables + figures written to", file.path(output_root(), "models"), "\n")
