#!/usr/bin/env bash
# Runs the whole R analysis twice for one data version (R/data_versions.R): whole sessions ->
# output/<DATA_VERSION>/, and every lane cut to arena entry + WINDOW s (lanes too short dropped) ->
# output/<DATA_VERSION>/<WINDOW>s_post_entry/. Run from r_analysis_RGB_Thermal/.
# Usage: DATA_VERSION=retrack_v2_2026-10 scripts/run_all.sh [WINDOW_SEC]   (default version
#        retrack_2026-10, default window 1750)
#
# Concordance: the sha256 of every input file goes to output/<DATA_VERSION>/INPUTS.sha256. If
# that manifest already exists and the inputs no longer match it, this stops instead of mixing
# figures from different inputs in one folder.
set -euo pipefail
cd "$(dirname "$0")/.."
WINDOW="${1:-1750}"
export DATA_VERSION="${DATA_VERSION:-retrack_2026-10}"
OUT="${OUTPUT_ROOT:-output/${DATA_VERSION}}"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
# Tagged lines only: renv may print startup notices on stdout.
inputs=$(Rscript -e 'source("R/data_versions.R"); cat(paste0("INPUT ", data_version_files()), sep = "\n")' \
         | sed -n 's/^INPUT //p')
[ -n "${inputs}" ] || { echo "FATAL: no inputs for DATA_VERSION=${DATA_VERSION} (see R/data_versions.R)" >&2; exit 1; }
for f in ${inputs}; do [ -f "$f" ] || { echo "FATAL: ${DATA_VERSION} input missing: $f" >&2; exit 1; }; done
manifest=$(sha ${inputs})
mkdir -p "${OUT}"
if [ -f "${OUT}/INPUTS.sha256" ]; then
  if ! diff -q <(echo "${manifest}") "${OUT}/INPUTS.sha256" >/dev/null; then
    echo "FATAL: ${OUT}/INPUTS.sha256 does not match the current ${DATA_VERSION} inputs:" >&2
    diff <(echo "${manifest}") "${OUT}/INPUTS.sha256" >&2 || true
    echo "Use a new data version (R/data_versions.R) rather than overwriting these results." >&2
    exit 1
  fi
else
  echo "${manifest}" > "${OUT}/INPUTS.sha256"
fi
echo "=== data version ${DATA_VERSION} -> ${OUT}/ (inputs: ${OUT}/INPUTS.sha256)"

SCRIPTS=(01_timecourse_plots 02_spaghetti_paired_reproduce 03_craniotomy_plots 04_distribution_plots
         05_bout_organization_plots 06_timeseries_models 07_all_experiments_plots)

for window in "" "$WINDOW"; do
  echo "=== window: ${window:-whole sessions}"
  for s in "${SCRIPTS[@]}"; do
    echo "--- $s"
    ANALYSIS_WINDOW_SEC="${window:-Inf}" Rscript "scripts/$s.R"
  done
done
