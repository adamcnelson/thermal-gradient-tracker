"""Tests for the 2026-10-14 moving-sample temperature-dip fixes (no .seq file required)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis_config import BoutsConfig
from src.mouse_segmentation import BackgroundModel, segment_mouse
from src.tracking import LowBlobTempGate
from src.velocity import compute_velocity


def _frame_with_mouse_and_nestlet():
    """Flat background; a big lukewarm square (nestlet) near the last centroid, a smaller hot mouse far away."""
    bg = np.full((43, 1000), 1000, dtype=np.uint16)
    frame = bg.copy()
    frame[5:35, 200:240] = 1600   # nestlet: 1200 px, near last centroid
    frame[10:30, 40:70] = 2000    # mouse: 600 px, warm, far away
    return frame, BackgroundModel(bg)


class TestBlobSelection:
    def test_largest_latches_onto_nestlet(self):
        frame, bg = _frame_with_mouse_and_nestlet()
        _, (cx, _), _, _ = segment_mouse(frame, bg, 100, 10000, last_centroid=(215, 20), search_radius=13)
        assert cx > 200

    def test_warmest_picks_mouse(self):
        frame, bg = _frame_with_mouse_and_nestlet()
        _, (cx, _), _, _ = segment_mouse(frame, bg, 100, 10000, last_centroid=(215, 20),
                                         search_radius=13, blob_selection="warmest")
        assert 40 <= cx <= 70

    def test_unknown_selection_raises(self):
        frame, bg = _frame_with_mouse_and_nestlet()
        with pytest.raises(ValueError):
            segment_mouse(frame, bg, 100, 10000, blob_selection="biggest")


class TestLowBlobTempGate:
    def _warm_gate(self):
        g = LowBlobTempGate(margin_c=2.0, n_history=15)
        for t in [29.0, 29.5, 30.0, 29.2, 29.8]:
            g.accept(t)
        return g

    def test_no_flag_until_history(self):
        g = LowBlobTempGate(margin_c=2.0, n_history=15)
        g.accept(30.0)
        assert g.check(10.0) is None

    def test_flags_cold_blob(self):
        assert self._warm_gate().check(20.0) == pytest.approx(29.5)

    def test_passes_warm_core(self):
        # Real mouse whose ROI mean dips still has a warm 95th percentile
        assert self._warm_gate().check(28.0) is None

    def test_resets_after_long_run(self):
        g = LowBlobTempGate(margin_c=2.0, n_history=15, reset_after=3)
        for t in [30.0] * 5:
            g.accept(t)
        assert [g.check(20.0) is not None for _ in range(4)] == [True, True, True, False]


class TestTempSmoothing:
    def test_smooth_skips_invalid_and_keeps_raw(self):
        n = 9
        temps = np.array([30, 30, 26, 30, 30, 10, 30, 30, 30], dtype=float)
        qc = ["ok"] * n
        qc[5] = "low_blob_temp"
        df = pd.DataFrame({
            "frame_number": np.arange(n) * 10, "elapsed_time_sec": np.arange(n, dtype=float),
            "mouse_centroid_x": np.linspace(50, 60, n), "mouse_centroid_y": np.full(n, 20.0),
            "qc_flag": qc, "mouse_roi_valid": [q == "ok" for q in qc],
            "mouse_surface_temp_mean": temps,
        })
        out = compute_velocity(df, BoutsConfig(temperature_smoothing_window_samples=3))
        assert out.loc[2, "mouse_surface_temp_smooth"] == 30      # single jag softened
        assert np.isnan(out.loc[5, "mouse_surface_temp_smooth"])  # flagged sample stays out
        assert out.loc[4, "mouse_surface_temp_smooth"] == 30      # flagged 10 C never smeared in
        assert out.loc[2, "mouse_surface_temp_mean"] == 26        # raw column untouched
