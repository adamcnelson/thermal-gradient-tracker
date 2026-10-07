"""Tests for the within-virus DCZ vs Vehicle tests in src/stats_models.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_models import paired_dcz_vehicle_tests, paired_t_summary, session_level_injection_lmm


def test_paired_t_summary_matches_scipy():
    dcz, veh = np.array([28.5, 24.5, 24.1, 23.4]), np.array([27.4, 27.0, 24.1, 24.0])
    res = paired_t_summary(dcz, veh)
    ref = stats.ttest_rel(dcz, veh)
    assert res["mean_diff"] == pytest.approx((dcz - veh).mean())
    assert res["t"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["n_mice"] == 4 and res["df"] == 3
    assert res["ci_lower"] < res["mean_diff"] < res["ci_upper"]


def test_paired_t_summary_degenerate_inputs():
    assert np.isnan(paired_t_summary([1.0], [0.5])["p"])          # n=1: difference only
    assert paired_t_summary([1.0], [0.5])["mean_diff"] == 0.5
    assert paired_t_summary([], [])["n_mice"] == 0


def _synthetic(effects={"Gi": 1.0, "Gq": -0.5}, n_mice=4, n_sess=4, seed=0, frames=30):
    """Frame-level rows: mouse intercept + DCZ effect (virus-specific) + session + frame noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for virus, eff in effects.items():
        for m in range(n_mice):
            mouse = f"{virus}{m}"
            base = 25 + rng.normal(0, 1.5)
            for inj in ("DCZ", "Vehicle"):
                for s in range(n_sess):
                    sess_mean = base + (eff if inj == "DCZ" else 0) + rng.normal(0, 0.3)
                    for st in (True, False):
                        for v in sess_mean + rng.normal(0, 1.0, frames):
                            rows.append(dict(mouse_id=mouse, virus=virus, injection=inj, stationary=st,
                                             video_file=f"{mouse}_{inj}_{s}.seq", y=v))
    return pd.DataFrame(rows)


def test_paired_tests_one_row_per_virus_and_bout_type():
    out = paired_dcz_vehicle_tests(_synthetic(), "y")
    assert len(out) == 4 and set(out.virus) == {"Gi", "Gq"}
    gi = out[(out.virus == "Gi") & (out.bout_type == "stationary")].iloc[0]
    assert gi.n_mice == 4 and gi.mean_diff == pytest.approx(1.0, abs=0.4)


def test_paired_tests_use_only_mice_with_both_conditions():
    df = _synthetic()
    df = df[~((df.mouse_id == "Gi0") & (df.injection == "DCZ"))]
    out = paired_dcz_vehicle_tests(df, "y")
    assert out[(out.virus == "Gi") & (out.bout_type == "stationary")].iloc[0].n_mice == 3


def test_session_lmm_recovers_virus_specific_effects_and_interaction():
    out = session_level_injection_lmm(_synthetic(n_mice=6, seed=1), "y").set_index(["bout_type", "term"])
    st = out.loc["stationary"]
    assert st.loc["DCZ effect | Gi", "estimate"] == pytest.approx(1.0, abs=0.35)
    assert st.loc["DCZ effect | Gq", "estimate"] == pytest.approx(-0.5, abs=0.35)
    assert st.loc["interaction: Gq effect - Gi effect", "estimate"] == pytest.approx(-1.5, abs=0.5)
    assert st.loc["DCZ effect | Gi", "n_sessions"] == 6 * 2 * 4 * 2  # mice x injections x sessions x viruses
    assert st.loc["DCZ effect | Gi", "p"] < 0.05
