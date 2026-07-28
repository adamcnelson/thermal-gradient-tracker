"""
Run treatment-effect analysis and generate all analysis tables and plots.

Usage:
    python scripts/analyze_treatment_effects.py [--overwrite]

--master, --bout-table, --config, and --output-dir all default to the
project-root-relative locations in src/paths.py and only need to be passed
to point at a non-default location.

Outputs:
    bouts/analysis_tables/  — group means, LMM coefficients, contrasts
    bouts/analysis_plots/   — all required treatment-effect plots
    bouts/analysis_README.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paths
from src.analysis_config import AnalysisConfig
from src.logging_utils import setup_logger
from src.stats_models import (
    PRELIMINARY_TAG,
    check_lmm_feasible,
    descriptive_summary,
    fit_lmm,
)
from src.treatment_plots import (
    plot_bout_count_by_group,
    plot_bout_duration_by_group,
    plot_dcz_vehicle_contrasts,
    plot_dcz_vehicle_paired,
    plot_floor_temp_distribution,
    plot_model_coefficients,
    plot_preferred_floor_temp,
    plot_surface_temp,
)


def _load_csv(path: str, label: str, log):
    p = Path(path)
    if not p.exists():
        log.error(f"{label} not found: {path}")
        return None
    try:
        df = pd.read_csv(str(p))
        log.info(f"Loaded {label}: {len(df)} rows")
        return df
    except Exception as exc:
        log.error(f"Failed to load {label}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Treatment-effect analysis and plots")
    parser.add_argument("--master", default=None,
                        help="Path to master tracking+metadata CSV (default: bouts/master_tracking_with_metadata.csv under the project root)")
    parser.add_argument("--bout-table", default=None,
                        help="Path to bout table CSV (default: bouts/bout_table.csv under the project root)")
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json (default: analysis_config.json under the project root)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: bouts/ under the project root)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log = setup_logger("analyze_treatment_effects")

    config_path = args.config or str(paths.DEFAULT_ANALYSIS_CONFIG)
    try:
        config = AnalysisConfig.load(config_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    out = Path(args.output_dir) if args.output_dir else paths.BOUTS_DIR
    master_path = args.master or str(out / "master_tracking_with_metadata.csv")
    bout_table_path = args.bout_table or str(out / "bout_table.csv")
    tables_dir = out / "analysis_tables"
    plots_dir = out / "analysis_plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    readme_path = out / "analysis_README.txt"
    if readme_path.exists() and not args.overwrite:
        log.error(f"{readme_path} exists. Use --overwrite to replace.")
        sys.exit(1)

    master_df = _load_csv(master_path, "master tracking table", log)
    bout_df = _load_csv(bout_table_path, "bout table", log)

    readme_lines = [
        "THERMAL-GRADIENT ANALYSIS PIPELINE — RESULTS SUMMARY",
        "=" * 60,
        f"{PRELIMINARY_TAG}",
        "",
    ]

    factors = config.analysis.group_factors
    random_effect = config.analysis.random_effect
    min_animals = config.analysis.min_animals_for_lmm
    min_obs = config.analysis.min_obs_per_group_for_lmm

    # ── Bout plots ─────────────────────────────────────────────────────────────
    if bout_df is not None:
        n = len(bout_df)
        log.info(f"\nGenerating bout plots (n={n} bouts)...")
        readme_lines.append(f"BOUTS: {n} stationary bouts across all files")

        # Plot 1: bout duration
        plot_bout_duration_by_group(bout_df, str(plots_dir), factors=factors)
        log.info("  bout_duration_by_*.png saved")

        # Plot 2: bout count and rate
        # Compute per-file summary if bout_summary_per_file.csv exists
        summary_csv = out / "bout_summary_per_file.csv"
        if summary_csv.exists():
            summary_df = pd.read_csv(str(summary_csv))
            if "mouse_id" not in summary_df.columns and bout_df is not None:
                # Try to merge mouse_id from bout_df
                if "mouse_id" in bout_df.columns and "video_file" in bout_df.columns:
                    id_map = bout_df.groupby("video_file")["mouse_id"].first().reset_index()
                    summary_df = summary_df.merge(id_map, on="video_file", how="left")
            plot_bout_count_by_group(summary_df, str(plots_dir), factors=factors)
            log.info("  bout_count_by_*.png saved")

        # Bout LMM
        readme_lines.append("\nBOUT DURATION MODEL:")
        for factor in factors:
            if bout_df is not None and factor in bout_df.columns:
                feasible, reason = check_lmm_feasible(
                    bout_df, "bout_duration_sec", [factor], random_effect, min_animals, min_obs
                )
                if feasible:
                    result = fit_lmm(bout_df, "bout_duration_sec", [factor], random_effect)
                    if result["success"]:
                        coef_df = result["coef_df"]
                        coef_df.to_csv(str(tables_dir / f"lmm_bout_duration_{factor}.csv"))
                        plot_model_coefficients(
                            coef_df, f"bout_duration_{factor}", str(plots_dir), n
                        )
                        readme_lines.append(f"  {factor}: LMM fit (n_animals computed from data)")
                        for w in result["warnings"]:
                            readme_lines.append(f"    WARNING: {w}")
                    else:
                        readme_lines.append(f"  {factor}: LMM failed — {result['warnings']}")
                else:
                    log.info(f"  LMM for bout_duration ~ {factor}: fallback — {reason}")
                    desc = descriptive_summary(bout_df, "bout_duration_sec", [factor])
                    desc.to_csv(str(tables_dir / f"descriptive_bout_duration_{factor}.csv"), index=False)
                    readme_lines.append(f"  {factor}: {PRELIMINARY_TAG} ({reason})")

    # ── Master table plots ─────────────────────────────────────────────────────
    if master_df is not None:
        n_master = len(master_df)
        log.info(f"\nGenerating master-table plots (n={n_master} samples)...")
        readme_lines.append(f"\nMASTER TABLE: {n_master} frame-samples")

        # Add stationary column if missing
        if "stationary" not in master_df.columns:
            log.warning("'stationary' column not in master table — cannot split stationary/non-stationary")
            master_df["stationary"] = False

        # Plot 3: preferred floor temp
        plot_preferred_floor_temp(master_df, str(plots_dir), factors=factors)
        log.info("  floor_temp_by_*.png saved")

        # Plot 4: surface temp
        plot_surface_temp(master_df, str(plots_dir), factors=factors)
        log.info("  mouse_surface_temp_by_*.png saved")

        # Plot 5: DCZ vs vehicle (aggregate box)
        plot_dcz_vehicle_contrasts(master_df, str(plots_dir))
        log.info("  contrast_dcz_vehicle_*.png saved")

        # Plot 8: paired DCZ vs Vehicle per-mouse, by virus × bout type (primary hypothesis)
        paired_outcomes = [
            "floor_temp_mean",
            "mouse_surface_temp_mean",
            "mouse_minus_floor_temp_mean",
            "velocity_smooth_px_s",
        ]
        plot_dcz_vehicle_paired(master_df, str(plots_dir), outcomes=paired_outcomes)
        log.info("  paired_dcz_vehicle_*.png saved")

        # Descriptive tables: per-mouse means for each outcome × virus × bout type
        exp_mask = (
            master_df["phase"].eq("experimental")
            & master_df["injection"].isin(["DCZ", "Vehicle"])
            & master_df["virus"].isin(["Gi", "Gq"])
        )
        if "qc_flag" in master_df.columns:
            exp_mask &= master_df["qc_flag"] == "ok"
        exp_df = master_df[exp_mask].copy()

        for outcome in paired_outcomes:
            if outcome not in exp_df.columns:
                continue
            exp_df[outcome] = pd.to_numeric(exp_df[outcome], errors="coerce")
            for stat_val, stat_label in [(True, "stationary"), (False, "non_stationary")]:
                sub = exp_df[exp_df["stationary"] == stat_val]
                if sub.empty:
                    continue
                mouse_means = (
                    sub.dropna(subset=[outcome, "mouse_id"])
                    .groupby(["mouse_id", "virus", "injection"], observed=True)[outcome]
                    .mean()
                    .reset_index()
                    .rename(columns={outcome: f"{outcome}_mean"})
                )
                fname = f"paired_{outcome}_{stat_label}.csv"
                mouse_means.to_csv(str(tables_dir / fname), index=False)
        readme_lines.append("\nPAIRED DCZ vs VEHICLE (per-mouse means, experimental phase):")
        readme_lines.append(f"  Plots: paired_dcz_vehicle_{{outcome}}.png (4 outcomes × 2×2 grid)")
        readme_lines.append(f"  Tables: paired_{{outcome}}_{{stationary|non_stationary}}.csv")

        # Plot 6: floor temp distribution
        plot_floor_temp_distribution(master_df, str(plots_dir), factors=factors)
        log.info("  floor_temp_dist_by_*.png saved")

        # Temperature LMMs (stationary subset only)
        readme_lines.append("\nTEMPERATURE MODELS (stationary subset):")
        stat_df = master_df[master_df["stationary"] == True]
        for outcome in ["floor_temp_mean", "mouse_surface_temp_mean", "mouse_minus_floor_temp_mean"]:
            if outcome not in stat_df.columns:
                continue
            # Scope to experimental phase + DCZ/Vehicle
            scope = stat_df[
                stat_df.get("phase", pd.Series(dtype=str)).isin(["experimental"])
            ] if "phase" in stat_df.columns else stat_df
            if len(scope) < 2:
                scope = stat_df

            for factor in factors:
                if factor not in scope.columns:
                    continue
                feasible, reason = check_lmm_feasible(
                    scope, outcome, [factor], random_effect, min_animals, min_obs
                )
                desc = descriptive_summary(scope, outcome, [factor])
                desc.to_csv(
                    str(tables_dir / f"descriptive_{outcome}_{factor}.csv"), index=False
                )
                if feasible:
                    result = fit_lmm(scope, outcome, [factor], random_effect)
                    if result["success"]:
                        coef_df = result["coef_df"]
                        coef_df.to_csv(str(tables_dir / f"lmm_{outcome}_{factor}.csv"))
                        plot_model_coefficients(
                            coef_df, f"{outcome}_{factor}", str(plots_dir), len(scope)
                        )
                        readme_lines.append(f"  {outcome} ~ {factor}: LMM fit ok")
                    else:
                        readme_lines.append(
                            f"  {outcome} ~ {factor}: LMM failed — {result['warnings']}"
                        )
                else:
                    readme_lines.append(
                        f"  {outcome} ~ {factor}: {PRELIMINARY_TAG} ({reason})"
                    )

    # ── Write README ───────────────────────────────────────────────────────────
    readme_lines += [
        "",
        "=" * 60,
        f"{PRELIMINARY_TAG}",
        "Re-run identical commands on the full dataset to obtain valid inference.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n")
    log.info(f"\nAnalysis README written to {readme_path}")
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
