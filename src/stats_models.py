"""Linear mixed models with assumption checks and graceful fallback to descriptives."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRELIMINARY_TAG = "PRELIMINARY — insufficient sample for inference"


def check_lmm_feasible(
    df: pd.DataFrame,
    outcome: str,
    group_factors: List[str],
    random_effect: str,
    min_animals: int = 3,
    min_obs_per_group: int = 2,
) -> Tuple[bool, str]:
    """
    Return (feasible, reason_string).

    Checks:
      - outcome column exists and has enough non-null values
      - random_effect column exists with >= min_animals unique levels
      - each group_factor column present with >= 2 populated levels
      - each group has >= min_obs_per_group observations
    """
    if outcome not in df.columns:
        return False, f"Outcome column '{outcome}' not found"

    sub = df[pd.to_numeric(df[outcome], errors="coerce").notna()].copy()
    if len(sub) < min_animals * min_obs_per_group:
        return False, f"Only {len(sub)} non-null rows for outcome '{outcome}'"

    if random_effect not in sub.columns:
        return False, f"Random effect column '{random_effect}' not found"

    n_animals = sub[random_effect].nunique()
    if n_animals < min_animals:
        return False, f"Only {n_animals} unique animals (need ≥{min_animals})"

    for f in group_factors:
        if f not in sub.columns:
            return False, f"Factor '{f}' not found"
        n_levels = sub[f].nunique()
        if n_levels < 2:
            return False, f"Factor '{f}' has only {n_levels} populated level(s)"
        # Check per-group observations
        for _, grp in sub.groupby(f):
            if len(grp) < min_obs_per_group:
                return False, f"Factor '{f}': some group has < {min_obs_per_group} observations"

    return True, "ok"


def fit_lmm(
    df: pd.DataFrame,
    outcome: str,
    fixed_effects: List[str],
    random_effect: str,
) -> Dict:
    """
    Fit a linear mixed model using statsmodels.

    Returns a dict with keys:
      success, model_summary, coef_df, residuals, shapiro_p, levene_p, warnings
    """
    result: Dict = {
        "success": False,
        "model_summary": None,
        "coef_df": None,
        "residuals": None,
        "shapiro_p": None,
        "levene_p": None,
        "warnings": [],
    }

    try:
        import statsmodels.formula.api as smf
        from scipy import stats as scipy_stats
    except ImportError:
        result["warnings"].append("statsmodels or scipy not installed — cannot fit LMM")
        return result

    sub = df.copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=[outcome] + fixed_effects + [random_effect])

    if len(sub) == 0:
        result["warnings"].append("No complete cases after dropping NaN")
        return result

    # Encode categorical fixed effects
    for fe in fixed_effects:
        if sub[fe].dtype == object or str(sub[fe].dtype) == "category":
            sub[fe] = sub[fe].astype("category")

    fe_str = " + ".join(f"C({fe})" if sub[fe].dtype.name == "category" else fe
                        for fe in fixed_effects)
    formula = f"{outcome} ~ {fe_str}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, sub, groups=sub[random_effect])
            fit = model.fit(reml=True)

        coef_df = pd.DataFrame({
            "coef": fit.params,
            "se": fit.bse,
            "z": fit.tvalues,
            "p": fit.pvalues,
            "ci_lower": fit.conf_int()[0],
            "ci_upper": fit.conf_int()[1],
        })

        resid = fit.resid

        # Assumption checks
        shapiro_p = None
        if len(resid) >= 3 and len(resid) <= 5000:
            try:
                _, shapiro_p = scipy_stats.shapiro(resid)
            except Exception:
                pass

        levene_p = None
        if len(fixed_effects) >= 1:
            try:
                groups = [grp[outcome].values for _, grp in sub.groupby(fixed_effects[0])]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) >= 2:
                    _, levene_p = scipy_stats.levene(*groups)
            except Exception:
                pass

        result.update({
            "success": True,
            "model_summary": fit.summary().as_text(),
            "coef_df": coef_df,
            "residuals": resid,
            "shapiro_p": shapiro_p,
            "levene_p": levene_p,
        })

        if shapiro_p is not None and shapiro_p < 0.05:
            result["warnings"].append(
                f"Residuals non-normal (Shapiro-Wilk p={shapiro_p:.3f}); interpret carefully"
            )
        if levene_p is not None and levene_p < 0.05:
            result["warnings"].append(
                f"Heteroscedasticity detected (Levene p={levene_p:.3f})"
            )

    except Exception as exc:
        result["warnings"].append(f"LMM fit failed: {exc}")

    return result


def descriptive_summary(
    df: pd.DataFrame,
    outcome: str,
    group_factors: List[str],
    add_preliminary: bool = True,
) -> pd.DataFrame:
    """
    Compute group mean ± SE for an outcome.

    Always adds a `note` column with PRELIMINARY_TAG when add_preliminary=True.
    """
    if outcome not in df.columns:
        return pd.DataFrame()

    sub = df.copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=[outcome])

    valid_factors = [f for f in group_factors if f in sub.columns]

    if not valid_factors:
        agg = sub[outcome].agg(["mean", "std", "count"]).to_frame().T
        agg.columns = ["mean", "sd", "n"]
        agg["se"] = agg["sd"] / np.sqrt(agg["n"])
    else:
        agg = (
            sub.groupby(valid_factors)[outcome]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        agg["se"] = agg["sd"] / np.sqrt(agg["n"])

    agg["outcome"] = outcome
    if add_preliminary:
        agg["note"] = PRELIMINARY_TAG

    return agg
