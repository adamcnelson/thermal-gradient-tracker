#!/usr/bin/env bash
# Runs the whole R analysis twice: whole sessions -> output/, and every lane cut to arena entry +
# WINDOW s (lanes too short dropped) -> output/<WINDOW>s_post_entry/. Run from r_analysis_RGB_Thermal/.
# Usage: scripts/run_all.sh [WINDOW_SEC]   (default 1750)
set -euo pipefail
cd "$(dirname "$0")/.."
WINDOW="${1:-1750}"

SCRIPTS=(01_timecourse_plots 02_spaghetti_paired_reproduce 03_craniotomy_plots 04_distribution_plots
         05_bout_organization_plots 06_timeseries_models 07_all_experiments_plots)

for window in "" "$WINDOW"; do
  echo "=== window: ${window:-whole sessions}"
  for s in "${SCRIPTS[@]}"; do
    echo "--- $s"
    ANALYSIS_WINDOW_SEC="${window:-Inf}" Rscript "scripts/$s.R"
  done
done
