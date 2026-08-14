"""Tests for src/landmarks/validation_set.py — no real data required."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.validation_set import select_validation_frames


def _make_candidates(n_sessions=6, n_frames_per_session_track=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for session_i in range(n_sessions):
        nestlet = session_i % 2 == 0
        for track in ["B", "F"]:
            for frame_i in range(n_frames_per_session_track):
                zone = rng.choice(["hot", "mid", "cool"], p=[0.33, 0.34, 0.33])
                on_nestlet = bool(nestlet and (rng.random() < 0.1))
                rows.append(
                    {
                        "session": f"s{session_i}",
                        "track": track,
                        "frame": frame_i,
                        "gradient_zone": zone,
                        "nestlet_present": nestlet,
                        "on_nestlet": on_nestlet,
                    }
                )
    return pd.DataFrame(rows)


class TestSelectValidationFrames:
    def test_selects_requested_total_when_pool_is_large_enough(self):
        candidates = _make_candidates()
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        assert len(result.rows) == 75

    def test_spreads_across_available_sessions_not_concentrated(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        counts = result.rows["session"].value_counts()
        assert result.rows["session"].nunique() == 6
        # no single session should dominate — roughly even split expected
        assert counts.max() - counts.min() <= 2

    def test_warns_when_fewer_sessions_than_target(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        assert any("Only 6 session" in w for w in result.warnings)

    def test_overs_samples_hot_and_mid_over_cool(self):
        candidates = _make_candidates(n_sessions=10, n_frames_per_session_track=200)
        result = select_validation_frames(
            candidates, n_total=75, n_sessions_target=10, on_nestlet_col="on_nestlet"
        )
        dist = result.rows["gradient_zone"].value_counts(normalize=True)
        assert dist["cool"] < dist["hot"]
        assert dist["cool"] < dist["mid"]

    def test_includes_both_nestlet_and_no_nestlet_sessions(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        represented = result.rows["nestlet_present"].unique()
        assert True in represented and False in represented

    def test_prioritizes_on_nestlet_frames_when_column_given(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        assert result.rows["on_nestlet"].sum() > 0

    def test_missing_on_nestlet_col_produces_warning_not_error(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25)
        assert any("on_nestlet_col" in w for w in result.warnings)

    def test_missing_posture_col_produces_warning(self):
        candidates = _make_candidates(n_sessions=6)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        assert any("posture" in w.lower() for w in result.warnings)

    def test_invalid_on_nestlet_col_raises(self):
        candidates = _make_candidates(n_sessions=6)
        with pytest.raises(ValueError):
            select_validation_frames(candidates, on_nestlet_col="does_not_exist")

    def test_missing_required_column_raises(self):
        candidates = _make_candidates(n_sessions=6).drop(columns=["gradient_zone"])
        with pytest.raises(ValueError):
            select_validation_frames(candidates)

    def test_small_pool_selects_all_available_and_warns(self):
        candidates = _make_candidates(n_sessions=2, n_frames_per_session_track=5)
        result = select_validation_frames(candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet")
        assert len(result.rows) < 75
        assert any("selected" in w for w in result.warnings)

    def test_deterministic_given_same_seed(self):
        candidates = _make_candidates(n_sessions=6)
        r1 = select_validation_frames(
            candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet", random_seed=42
        )
        r2 = select_validation_frames(
            candidates, n_total=75, n_sessions_target=25, on_nestlet_col="on_nestlet", random_seed=42
        )
        pd.testing.assert_frame_equal(
            r1.rows.sort_values(["session", "track", "frame"]).reset_index(drop=True),
            r2.rows.sort_values(["session", "track", "frame"]).reset_index(drop=True),
        )
