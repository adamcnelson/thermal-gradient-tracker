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


# ── DCZ vs Vehicle within each virus (repeated measures) ─────────────────────
# Sign convention for everything below: DCZ − Vehicle (positive = higher under DCZ).
# The pooled lmm_*_injection tables above report the opposite (injection[T.Vehicle]).

def paired_t_summary(dcz, vehicle) -> Dict:
    """Paired t-test on matched per-mouse means: mean DCZ−Vehicle difference, its 95% CI, t, p."""
    from scipy import stats as scipy_stats

    d = np.asarray(dcz, dtype=float) - np.asarray(vehicle, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out = dict(n_mice=n, mean_diff=np.nan, sd_diff=np.nan, se_diff=np.nan, t=np.nan, df=np.nan,
               p=np.nan, ci_lower=np.nan, ci_upper=np.nan)
    if n == 0:
        return out
    out["mean_diff"] = float(d.mean())
    if n < 2:
        return out
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    half = float(scipy_stats.t.ppf(0.975, n - 1)) * se
    out.update(sd_diff=sd, se_diff=se, df=n - 1, ci_lower=out["mean_diff"] - half, ci_upper=out["mean_diff"] + half)
    if se > 0:
        t = out["mean_diff"] / se
        out.update(t=float(t), p=float(2 * scipy_stats.t.sf(abs(t), n - 1)))
    return out


def per_mouse_means(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """mouse_id × {DCZ, Vehicle} table of per-mouse means (the spaghetti-plot points)."""
    sub = df.dropna(subset=[outcome, "mouse_id"])
    return sub.groupby(["mouse_id", "injection"], observed=True)[outcome].mean().unstack("injection")


def paired_dcz_vehicle_tests(exp_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Per virus × bout type: paired t-test of DCZ vs Vehicle on per-mouse means, i.e. the
    test behind each paired_dcz_vehicle_{outcome}.png panel. Only mice with both
    conditions count. n = mice, so power is low (4 per virus here): read the CI too.
    """
    rows = []
    for virus in sorted(exp_df["virus"].dropna().unique()):
        for stat_val, stat_label in [(True, "stationary"), (False, "non_stationary")]:
            sub = exp_df[(exp_df["virus"] == virus) & (exp_df["stationary"] == stat_val)]
            m = per_mouse_means(sub, outcome) if not sub.empty else pd.DataFrame()
            if {"DCZ", "Vehicle"}.issubset(m.columns):
                m = m[["DCZ", "Vehicle"]].dropna()
                res = paired_t_summary(m["DCZ"], m["Vehicle"])
            else:
                res = paired_t_summary([], [])
            rows.append(dict(outcome=outcome, virus=virus, bout_type=stat_label, test="paired t (per-mouse means)",
                             contrast="DCZ - Vehicle", **res))
    return pd.DataFrame(rows)


def _find_param(params: pd.Index, *needles: str) -> Optional[str]:
    hits = [p for p in params if all(n in p for n in needles)]
    return hits[0] if len(hits) == 1 else None


def session_level_injection_lmm(
    exp_df: pd.DataFrame, outcome: str, session_col: str = "video_file", mouse_col: str = "mouse_id",
) -> pd.DataFrame:
    """
    Per bout type: mixed model on SESSION-level means (one value per mouse × session),
        outcome ~ injection × virus + (1 | mouse)
    reporting the DCZ − Vehicle effect within Gi and within Gq, and the interaction
    (Gq effect − Gi effect). The session is the unit of replication -- unlike the
    frame-level lmm_* tables, whose ~10^5 correlated rows make every p≈0.

    Same model, two parameterizations: virus + virus:injection gives each virus's DCZ
    effect directly; injection × virus gives the interaction.
    """
    import statsmodels.formula.api as smf

    inj = "C(injection, Treatment('Vehicle'))"
    rows = []
    for stat_val, stat_label in [(True, "stationary"), (False, "non_stationary")]:
        sub = exp_df[exp_df["stationary"] == stat_val].dropna(subset=[outcome, mouse_col, session_col])
        sess = (sub.groupby([session_col, mouse_col, "virus", "injection"], observed=True)[outcome]
                .mean().reset_index())
        base = dict(outcome=outcome, bout_type=stat_label, model=f"{outcome} ~ injection * virus + (1|mouse), "
                    "session-level means", n_sessions=len(sess), n_mice=sess[mouse_col].nunique())
        if sess["virus"].nunique() < 2 or sess["injection"].nunique() < 2 or sess[mouse_col].nunique() < 3:
            rows.append(dict(base, term="not fitted", note="needs 2 viruses, 2 injections, >=3 mice"))
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                simple = smf.mixedlm(f"{outcome} ~ C(virus) + C(virus):{inj}", sess,
                                     groups=sess[mouse_col]).fit(reml=True)
                inter = smf.mixedlm(f"{outcome} ~ {inj} * C(virus, Treatment('Gi'))", sess,
                                    groups=sess[mouse_col]).fit(reml=True)
            note = "; ".join(sorted({str(w.message).split("\n")[0] for w in caught}))[:300]
        except Exception as exc:
            rows.append(dict(base, term="fit failed", note=str(exc)[:300]))
            continue

        terms = [(f"DCZ effect | {v}", simple, _find_param(simple.params.index, f"[{v}]", "[T.DCZ]"))
                 for v in sorted(sess["virus"].unique())]
        terms.append(("interaction: Gq effect - Gi effect", inter, _find_param(inter.params.index, "[T.DCZ]", ":")))
        for label, fit, name in terms:
            if name is None:
                rows.append(dict(base, term=label, note="term not found"))
                continue
            ci = fit.conf_int().loc[name]
            rows.append(dict(base, term=label, estimate=float(fit.params[name]), se=float(fit.bse[name]),
                             z=float(fit.tvalues[name]), p=float(fit.pvalues[name]),
                             ci_lower=float(ci[0]), ci_upper=float(ci[1]),
                             converged=bool(fit.converged), note=note))
    return pd.DataFrame(rows)
