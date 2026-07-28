"""Treatment-effect plots for the thermal-gradient analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRELIMINARY_TAG = "PRELIMINARY — insufficient sample for inference"
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _add_preliminary(fig: plt.Figure, n: int) -> None:
    fig.text(
        0.5, 0.01,
        f"{PRELIMINARY_TAG}  (n={n})",
        ha="center", va="bottom", fontsize=8, color="red", style="italic",
        wrap=True,
    )


def _save(fig: plt.Figure, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _mouse_colors(df: pd.DataFrame) -> dict:
    """Map mouse_id → color."""
    if "mouse_id" not in df.columns:
        return {}
    ids = sorted(df["mouse_id"].dropna().unique())
    return {mid: _PALETTE[i % len(_PALETTE)] for i, mid in enumerate(ids)}


def _jitter(n: int, width: float = 0.15) -> np.ndarray:
    return np.random.default_rng(42).uniform(-width, width, n)


# ── Plot 1: bout duration by group ────────────────────────────────────────────

def plot_bout_duration_by_group(
    bout_df: pd.DataFrame,
    output_dir: str,
    factors: Optional[List[str]] = None,
) -> None:
    """Strip + box plot of stationary bout duration, faceted by factor."""
    factors = factors or ["virus", "injection", "phase"]
    colors = _mouse_colors(bout_df)
    n = len(bout_df)

    for factor in factors:
        if factor not in bout_df.columns:
            continue
        levels = sorted(bout_df[factor].dropna().unique())
        if not levels:
            continue

        fig, ax = plt.subplots(figsize=(max(5, len(levels) * 2), 5))
        for xi, level in enumerate(levels):
            sub = bout_df[bout_df[factor] == level]["bout_duration_sec"].dropna()
            if len(sub) == 0:
                continue

            # Box
            bp = ax.boxplot(
                sub.values, positions=[xi], widths=0.4,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="whitesmoke", alpha=0.7),
                medianprops=dict(color="black", lw=2),
            )

            # Individual points coloured by mouse_id
            jit = _jitter(len(sub))
            sub_df = bout_df[bout_df[factor] == level][["bout_duration_sec", "mouse_id"]].dropna()
            for _, row in sub_df.iterrows():
                c = colors.get(row.get("mouse_id", ""), "gray")
                ax.scatter(xi + _jitter(1)[0], row["bout_duration_sec"],
                           color=c, s=30, alpha=0.8, zorder=3)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlabel(factor)
        ax.set_ylabel("Bout duration (s)")
        ax.set_title(f"Stationary bout duration by {factor}")

        if colors:
            handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, label=mid, markersize=8)
                       for mid, c in colors.items()]
            ax.legend(handles=handles, title="Mouse ID", fontsize=7, loc="upper right")

        _add_preliminary(fig, n)
        _save(fig, str(Path(output_dir) / f"bout_duration_by_{factor}.png"))


# ── Plot 2: bout count and rate ────────────────────────────────────────────────

def plot_bout_count_by_group(
    summary_df: pd.DataFrame,
    output_dir: str,
    factors: Optional[List[str]] = None,
) -> None:
    """Bar chart of bout count and rate per 1000s."""
    factors = factors or ["virus", "injection", "phase"]
    n = len(summary_df)

    for metric, label in [
        ("n_stationary_bouts", "N stationary bouts"),
        ("bouts_per_1000s", "Bouts per 1000 s"),
        ("fraction_time_stationary", "Fraction of time stationary"),
    ]:
        if metric not in summary_df.columns:
            continue
        for factor in factors:
            if factor not in summary_df.columns:
                continue
            sub = summary_df[[factor, metric, "mouse_id"]].dropna(subset=[metric])
            levels = sorted(sub[factor].dropna().unique())
            if not levels:
                continue

            fig, ax = plt.subplots(figsize=(max(5, len(levels) * 2), 5))
            colors = _mouse_colors(sub)
            for xi, level in enumerate(levels):
                grp = sub[sub[factor] == level][metric].dropna()
                if len(grp) == 0:
                    continue
                ax.bar(xi, grp.mean(), yerr=grp.sem() if len(grp) > 1 else 0,
                       color="steelblue", alpha=0.6, capsize=4)
                for _, row in sub[sub[factor] == level].iterrows():
                    c = colors.get(row.get("mouse_id", ""), "gray")
                    ax.scatter(xi + _jitter(1)[0], row[metric],
                               color=c, s=40, zorder=3, alpha=0.85)

            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels)
            ax.set_xlabel(factor)
            ax.set_ylabel(label)
            ax.set_title(f"{label} by {factor}")
            _add_preliminary(fig, n)
            _save(fig, str(Path(output_dir) / f"{metric}_by_{factor}.png"))


# ── Plot 3: preferred floor temperature ───────────────────────────────────────

def plot_preferred_floor_temp(
    master_df: pd.DataFrame,
    output_dir: str,
    factors: Optional[List[str]] = None,
) -> None:
    """Preferred floor temperature, faceted by stationary vs non-stationary."""
    factors = factors or ["virus", "injection", "phase"]
    if "floor_temp_mean" not in master_df.columns:
        return
    if "stationary" not in master_df.columns:
        return

    n = len(master_df)
    colors = _mouse_colors(master_df)

    for factor in factors:
        if factor not in master_df.columns:
            continue
        levels = sorted(master_df[factor].dropna().unique())
        if not levels:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(levels) * 3), 5), sharey=True)
        for ax, state in zip(axes, [True, False]):
            sub = master_df[master_df["stationary"] == state].copy()
            sub["floor_temp_mean"] = pd.to_numeric(sub["floor_temp_mean"], errors="coerce")
            label = "Stationary" if state else "Non-stationary"
            # Aggregate to per-mouse means — the animal is the biological unit
            mouse_means = (
                sub.dropna(subset=["floor_temp_mean", "mouse_id"])
                .groupby(["mouse_id", factor], observed=True)["floor_temp_mean"]
                .mean()
                .reset_index()
            )
            for xi, level in enumerate(levels):
                grp = mouse_means[mouse_means[factor] == level]["floor_temp_mean"].dropna()
                if len(grp) == 0:
                    continue
                ax.boxplot(grp.values, positions=[xi], widths=0.4,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor="lightyellow", alpha=0.7),
                           medianprops=dict(color="black"))
                for _, row in mouse_means[mouse_means[factor] == level].iterrows():
                    v = row["floor_temp_mean"]
                    if pd.isna(v):
                        continue
                    c = colors.get(row.get("mouse_id", ""), "gray")
                    ax.scatter(xi + _jitter(1)[0], v, color=c, s=40, alpha=0.85, zorder=3)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels)
            ax.set_xlabel(factor)
            ax.set_ylabel("Floor temp (°C) — per-mouse mean")
            ax.set_title(f"Floor temp — {label}")
        _add_preliminary(fig, n)
        _save(fig, str(Path(output_dir) / f"floor_temp_by_{factor}.png"))


# ── Plot 4: surface temp and mouse-minus-floor ────────────────────────────────

def plot_surface_temp(
    master_df: pd.DataFrame,
    output_dir: str,
    factors: Optional[List[str]] = None,
) -> None:
    """Mouse surface temperature and mouse-minus-floor by group."""
    factors = factors or ["virus", "injection", "phase"]
    n = len(master_df)
    colors = _mouse_colors(master_df)

    for outcome, label in [
        ("mouse_surface_temp_mean", "Mouse surface temp (°C)"),
        ("mouse_minus_floor_temp_mean", "Mouse − floor temp (°C)"),
    ]:
        if outcome not in master_df.columns:
            continue
        df_out = master_df.copy()
        df_out[outcome] = pd.to_numeric(df_out[outcome], errors="coerce")
        # Aggregate to per-mouse means — the animal is the biological unit
        for factor in factors:
            if factor not in df_out.columns:
                continue
            levels = sorted(df_out[factor].dropna().unique())
            if not levels:
                continue

            mouse_means = (
                df_out.dropna(subset=[outcome, "mouse_id"])
                .groupby(["mouse_id", factor], observed=True)[outcome]
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(max(5, len(levels) * 2), 5))
            for xi, level in enumerate(levels):
                grp = mouse_means[mouse_means[factor] == level][outcome].dropna()
                if len(grp) == 0:
                    continue
                ax.boxplot(grp.values, positions=[xi], widths=0.4,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor="lightcyan"),
                           medianprops=dict(color="black"))
                for _, row in mouse_means[mouse_means[factor] == level].iterrows():
                    v = row[outcome]
                    if pd.isna(v):
                        continue
                    c = colors.get(row.get("mouse_id", ""), "gray")
                    ax.scatter(xi + _jitter(1)[0], v, color=c, s=40, alpha=0.85, zorder=3)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels)
            ax.set_xlabel(factor)
            ax.set_ylabel(f"{label} — per-mouse mean")
            ax.set_title(f"{label} by {factor}")
            _add_preliminary(fig, n)
            _save(fig, str(Path(output_dir) / f"{outcome}_by_{factor}.png"))


# ── Plot 5: DCZ vs vehicle contrasts ─────────────────────────────────────────

def plot_dcz_vehicle_contrasts(
    master_df: pd.DataFrame,
    output_dir: str,
    outcomes: Optional[List[str]] = None,
) -> None:
    """Paired DCZ vs Vehicle contrast plots, split by virus."""
    outcomes = outcomes or [
        "floor_temp_mean", "mouse_surface_temp_mean",
        "mouse_minus_floor_temp_mean",
    ]
    if "injection" not in master_df.columns or "virus" not in master_df.columns:
        return

    n = len(master_df)
    sub = master_df[master_df["injection"].isin(["DCZ", "Vehicle"])]
    viruses = sorted(sub["virus"].dropna().unique())

    for outcome in outcomes:
        if outcome not in sub.columns:
            continue
        sub_o = sub.copy()
        sub_o[outcome] = pd.to_numeric(sub_o[outcome], errors="coerce")
        sub_o = sub_o.dropna(subset=[outcome])
        if len(sub_o) == 0:
            continue

        fig, axes = plt.subplots(1, max(1, len(viruses)),
                                 figsize=(4 * max(1, len(viruses)), 5), sharey=True)
        if len(viruses) == 1:
            axes = [axes]
        for ax, virus in zip(axes, viruses):
            vdf = sub_o[sub_o["virus"] == virus]
            for xi, inj in enumerate(["Vehicle", "DCZ"]):
                grp = vdf[vdf["injection"] == inj][outcome].dropna()
                if len(grp) == 0:
                    continue
                ax.boxplot(grp.values, positions=[xi], widths=0.4,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=["lightgreen", "salmon"][xi]),
                           medianprops=dict(color="black"))
                ax.scatter([xi + _jitter(1)[0]] * len(grp), grp.values,
                           color="k", s=20, alpha=0.6, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Vehicle", "DCZ"])
            ax.set_title(f"Virus={virus}")
            ax.set_ylabel(outcome)
        fig.suptitle(f"DCZ vs Vehicle — {outcome}")
        _add_preliminary(fig, n)
        _save(fig, str(Path(output_dir) / f"contrast_dcz_vehicle_{outcome}.png"))


# ── Plot 6: floor temperature distribution ───────────────────────────────────

def plot_floor_temp_distribution(
    master_df: pd.DataFrame,
    output_dir: str,
    factors: Optional[List[str]] = None,
) -> None:
    """Histogram / KDE of occupied floor temperatures (stationary) per group."""
    if "floor_temp_mean" not in master_df.columns or "stationary" not in master_df.columns:
        return
    factors = factors or ["virus", "injection", "phase"]
    n = len(master_df)

    stat_df = master_df[master_df["stationary"] == True]
    ft = pd.to_numeric(stat_df["floor_temp_mean"], errors="coerce").dropna()
    if ft.empty:
        return

    for factor in factors:
        if factor not in stat_df.columns:
            continue
        levels = sorted(stat_df[factor].dropna().unique())
        if not levels:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        for level, color in zip(levels, _PALETTE):
            vals = pd.to_numeric(
                stat_df[stat_df[factor] == level]["floor_temp_mean"], errors="coerce"
            ).dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=20, alpha=0.5, label=str(level), color=color, density=True)
        ax.set_xlabel("Floor temp (°C)")
        ax.set_ylabel("Density")
        ax.set_title(f"Floor temp distribution (stationary) by {factor}")
        ax.legend(fontsize=8)
        _add_preliminary(fig, n)
        _save(fig, str(Path(output_dir) / f"floor_temp_dist_by_{factor}.png"))


# ── Plot 8: paired DCZ vs Vehicle per-mouse, by virus and bout type ──────────

_OUTCOME_LABELS = {
    "floor_temp_mean": "Floor temperature (°C)",
    "mouse_surface_temp_mean": "Mouse surface temperature (°C)",
    "mouse_minus_floor_temp_mean": "Mouse − floor temperature (°C)",
    "velocity_smooth_px_s": "Velocity (px/s)",
}


def plot_dcz_vehicle_paired(
    master_df: pd.DataFrame,
    output_dir: str,
    outcomes: Optional[List[str]] = None,
    valid_qc_only: bool = True,
) -> None:
    """
    Paired Vehicle→DCZ slope plots for each outcome, split by virus (Gi/Gq) and
    bout type (stationary/non-stationary).

    One figure per outcome (4 figures), 2×2 panel grid:
      rows = virus (Gi, Gq)
      cols = bout type (Stationary, Non-stationary)

    Each mouse is a connected line; the group mean ± SE is overlaid as a black diamond.
    Restricted to the experimental phase (where DCZ/Vehicle contrast exists).
    """
    outcomes = outcomes or list(_OUTCOME_LABELS.keys())

    required = {"injection", "virus", "phase", "stationary", "mouse_id"}
    if not required.issubset(master_df.columns):
        return

    # Experimental phase, DCZ and Vehicle only, DREADD animals only
    mask = (
        (master_df["phase"] == "experimental")
        & (master_df["injection"].isin(["DCZ", "Vehicle"]))
        & (master_df["virus"].isin(["Gi", "Gq"]))
    )
    if valid_qc_only and "qc_flag" in master_df.columns:
        mask &= master_df["qc_flag"] == "ok"
    exp = master_df[mask].copy()

    if exp.empty:
        return

    viruses = sorted(exp["virus"].dropna().unique())
    bout_types = [(True, "Stationary"), (False, "Non-stationary")]
    n_rows, n_cols = len(viruses), len(bout_types)

    colors_map = _mouse_colors(exp)

    for outcome in outcomes:
        if outcome not in exp.columns:
            continue
        exp[outcome] = pd.to_numeric(exp[outcome], errors="coerce")
        ylabel = _OUTCOME_LABELS.get(outcome, outcome)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"{ylabel}\nDCZ vs Vehicle  |  experimental phase  |  per-mouse mean",
            fontsize=11,
        )

        for ri, virus in enumerate(viruses):
            for ci, (stat_val, stat_label) in enumerate(bout_types):
                ax = axes[ri][ci]
                ax.set_title(f"{virus}  —  {stat_label}", fontsize=10)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Vehicle", "DCZ"])
                if ci == 0:
                    ax.set_ylabel(ylabel)

                sub = exp[(exp["virus"] == virus) & (exp["stationary"] == stat_val)]
                if sub.empty or sub[outcome].isna().all():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    continue

                # Per-mouse mean for each injection condition
                mouse_means = (
                    sub.dropna(subset=[outcome, "mouse_id"])
                    .groupby(["mouse_id", "injection"], observed=True)[outcome]
                    .mean()
                    .unstack("injection")
                )

                # 1. Individual mouse means as colored dots at x=0 (Vehicle) and x=1 (DCZ)
                labels_added: set = set()
                for xi, inj in enumerate(["Vehicle", "DCZ"]):
                    if inj not in mouse_means.columns:
                        continue
                    for mouse_id, val in mouse_means[inj].items():
                        if pd.isna(val):
                            continue
                        c = colors_map.get(str(mouse_id), "gray")
                        lbl = str(mouse_id) if str(mouse_id) not in labels_added else None
                        ax.scatter(xi, val, color=c, s=70, zorder=4, alpha=0.9, label=lbl)
                        if lbl:
                            labels_added.add(lbl)

                # 2. Connecting lines for mice that have both conditions
                for mouse_id, row in mouse_means.iterrows():
                    v = row.get("Vehicle", np.nan)
                    d = row.get("DCZ", np.nan)
                    if pd.isna(v) or pd.isna(d):
                        continue
                    c = colors_map.get(str(mouse_id), "gray")
                    ax.plot([0, 1], [v, d], "-", color=c, lw=1.5, alpha=0.5, zorder=3)

                # 3. Group mean ± SE offset to the right (x=0.22, x=1.22)
                for xi, inj in enumerate(["Vehicle", "DCZ"]):
                    if inj not in mouse_means.columns:
                        continue
                    vals = mouse_means[inj].dropna()
                    if len(vals) == 0:
                        continue
                    ax.errorbar(
                        xi + 0.22, vals.mean(),
                        yerr=vals.sem() if len(vals) > 1 else 0,
                        fmt="D", color="black", ms=9, capsize=5, lw=2, zorder=5,
                    )

                ax.set_xlim(-0.4, 1.55)

        # Shared legend: collect labeled artists across all panels (deduped by label)
        handles, labels = [], []
        seen_labels: set = set()
        for ax_row in axes:
            for ax in ax_row:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l and l not in seen_labels:
                        handles.append(h)
                        labels.append(l)
                        seen_labels.add(l)
        if handles:
            fig.legend(handles, labels, title="Mouse ID", fontsize=8,
                       loc="lower center", ncol=min(len(labels), 6),
                       bbox_to_anchor=(0.5, -0.04))

        _add_preliminary(fig, len(exp))
        fig.tight_layout()
        _save(fig, str(Path(output_dir) / f"paired_dcz_vehicle_{outcome}.png"))


# ── Plot 7: model coefficients ────────────────────────────────────────────────

def plot_model_coefficients(
    coef_df: pd.DataFrame,
    outcome: str,
    output_dir: str,
    n_obs: int = 0,
) -> None:
    """Coefficient plot with 95% CIs from an LMM fit."""
    if coef_df is None or coef_df.empty:
        return
    # Drop intercept for readability
    plot_df = coef_df[~coef_df.index.str.lower().str.contains("intercept")].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(plot_df) * 0.6 + 1)))
    y = range(len(plot_df))
    ax.errorbar(
        plot_df["coef"], y,
        xerr=[
            plot_df["coef"] - plot_df["ci_lower"],
            plot_df["ci_upper"] - plot_df["coef"],
        ],
        fmt="o", color="steelblue", capsize=4,
    )
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df.index.tolist(), fontsize=8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title(f"LMM coefficients — {outcome}")
    _add_preliminary(fig, n_obs)
    _save(fig, str(Path(output_dir) / f"coef_plot_{outcome}.png"))
