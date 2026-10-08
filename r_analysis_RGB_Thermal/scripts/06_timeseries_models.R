# Section 4.4 — time-series models (LMM + GAMM check), DCZ vs Vehicle trajectories. Run from r_analysis_RGB_Thermal/.
source("R/data.R")
source("R/model_timeseries.R")

frames <- load_frames()
bouts <- load_bouts()
res <- build_timeseries_models(frames, bouts, output_dir = "output/models")

cat("\nLMM fixed effects (DCZ - Vehicle; slope difference = trajectory divergence):\n")
print(as.data.frame(res$lmm[, intersect(c("outcome", "virus", "term", "estimate", "conf.low", "conf.high",
                                          "p.value", "random_effects", "note"), names(res$lmm))]), digits = 3)
cat("\nGAMM difference smooth (does the trajectory SHAPE differ?):\n")
print(as.data.frame(res$gamm), digits = 3)
cat("\nDone. Tables + figures written to r_analysis_RGB_Thermal/output/models/\n")
