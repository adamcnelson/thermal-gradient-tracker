"""
Tests for src/landmarks/sync.py — no real data required.

The cross-correlation sign convention was wrong on the first pass (caught by
the spike test below, not by inspection) — scipy.signal.correlate's peak
index does not point where a formula-level reading suggests. Keeping that
test in the suite so a future refactor can't silently reintroduce it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.sync import (
    WindowedSyncResult,
    cross_correlate_lag,
    fit_windowed_lag,
    passes_acceptance,
    thermal_motion_energy,
)


class TestThermalMotionEnergy:
    def test_extracts_and_sorts_by_time(self):
        df = pd.DataFrame(
            {
                "elapsed_time_sec": [2.0, 0.0, 1.0],
                "velocity_smooth_px_s": [20.0, 5.0, 10.0],
            }
        )
        trace = thermal_motion_energy(df)
        assert list(trace.index) == [0.0, 1.0, 2.0]
        assert list(trace.values) == [5.0, 10.0, 20.0]

    def test_drops_nan_rows(self):
        df = pd.DataFrame(
            {
                "elapsed_time_sec": [0.0, 1.0, 2.0],
                "velocity_smooth_px_s": [5.0, np.nan, 10.0],
            }
        )
        trace = thermal_motion_energy(df)
        assert len(trace) == 2


class TestCrossCorrelateLagSignConvention:
    def test_spike_delayed_in_b_gives_positive_lag(self):
        """
        b's feature occurs strictly after a's in absolute time -> b's clock
        reads later for the same event -> positive lag, by this module's
        documented convention.
        """
        dt = 1.0
        n = 300
        a = pd.Series(np.zeros(n), index=np.arange(n) * dt)
        b_vals = np.zeros(n)
        b_vals[120] = 1.0
        a_vals = np.zeros(n)
        a_vals[100] = 1.0
        a = pd.Series(a_vals, index=np.arange(n) * dt)
        b = pd.Series(b_vals, index=np.arange(n) * dt)

        lag = cross_correlate_lag(a, b, dt=dt)
        assert lag == pytest.approx(20.0, abs=1e-3)

    def test_spike_delayed_in_a_gives_negative_lag(self):
        dt = 1.0
        n = 300
        a_vals = np.zeros(n)
        a_vals[150] = 1.0
        b_vals = np.zeros(n)
        b_vals[100] = 1.0
        a = pd.Series(a_vals, index=np.arange(n) * dt)
        b = pd.Series(b_vals, index=np.arange(n) * dt)

        lag = cross_correlate_lag(a, b, dt=dt)
        assert lag == pytest.approx(-50.0, abs=1e-3)

    def test_no_overlap_raises(self):
        a = pd.Series([1.0, 2.0], index=[0.0, 1.0])
        b = pd.Series([1.0, 2.0], index=[1000.0, 1001.0])
        with pytest.raises(ValueError):
            cross_correlate_lag(a, b, dt=0.1)

    def test_constant_trace_raises(self):
        a = pd.Series(np.ones(50), index=np.arange(50) * 0.1)
        b = pd.Series(np.arange(50, dtype=float), index=np.arange(50) * 0.1)
        with pytest.raises(ValueError):
            cross_correlate_lag(a, b, dt=0.1)


def _realistic_motion_signal(session_len=2000.0, fine_dt=0.02, seed=42):
    """
    Smoothed-noise + several bigger, IRREGULARLY-spaced bursts, closer to
    real locomotion motion-energy than either isolated spikes or evenly
    (periodically) spaced ones. Both alternatives were tried and rejected:
    isolated/sparse spikes starve short correlation windows of information
    and produce noisy lag estimates; evenly-spaced (periodic) bursts create
    correlation-function aliasing — multiple near-equal peaks a window's
    width apart — that can lock onto the wrong peak entirely. Irregular
    spacing avoids both failure modes and is what real motion looks like.
    """
    rng = np.random.default_rng(seed)
    t_fine = np.arange(0, session_len, fine_dt)
    base = np.abs(gaussian_filter1d(rng.normal(size=len(t_fine)), sigma=8))
    burst_times = [80, 210, 430, 520, 760, 990, 1120, 1380, 1590, 1740, 1900]
    bursts = np.zeros_like(t_fine)
    for bt in burst_times:
        bursts += 8.0 * np.exp(-0.5 * ((t_fine - bt) / 6.0) ** 2)
    signal = base + bursts

    def sample(t_query):
        return np.interp(t_query, t_fine, signal)

    return sample


class TestFitWindowedLag:
    def test_recovers_offset_and_drift_within_tolerance(self):
        sample = _realistic_motion_signal()
        offset, drift = 0.5, 0.0015
        session_len = 2000.0

        t_thermal = np.arange(0, session_len, 1 / 8)
        thermal = pd.Series(sample(t_thermal), index=t_thermal)

        t_true_rgb = np.arange(0, session_len, 1 / 25)
        rgb = pd.Series(sample(t_true_rgb), index=offset + (1 + drift) * t_true_rgb)

        result = fit_windowed_lag(thermal, rgb, n_windows=6, window_frac=0.2, dt=0.05)

        assert result.offset_sec == pytest.approx(offset, abs=0.05)
        assert result.drift_slope == pytest.approx(drift, abs=2e-4)
        assert result.r_squared > 0.95
        assert passes_acceptance(result, thermal_frame_sec=1 / 8)

    def test_zero_drift_pure_offset(self):
        sample = _realistic_motion_signal(seed=7)
        offset = 0.3
        session_len = 2000.0

        t_thermal = np.arange(0, session_len, 1 / 8)
        thermal = pd.Series(sample(t_thermal), index=t_thermal)
        t_true_rgb = np.arange(0, session_len, 1 / 25)
        rgb = pd.Series(sample(t_true_rgb), index=offset + t_true_rgb)

        result = fit_windowed_lag(thermal, rgb, n_windows=5, window_frac=0.25, dt=0.05)
        assert result.offset_sec == pytest.approx(offset, abs=0.05)
        assert abs(result.drift_slope) < 1e-3
        # A flat (near-zero-slope) fit can have a poorly-defined R² even when
        # the offset itself is accurately recovered — acceptance must pass
        # via the "drift ~ zero" branch, not the R² branch, in this regime.
        assert passes_acceptance(result, thermal_frame_sec=1 / 8, drift_r2_min=0.9)

    def test_no_overlap_raises(self):
        thermal = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        rgb = pd.Series([1.0, 2.0, 3.0], index=[500.0, 501.0, 502.0])
        with pytest.raises(ValueError):
            fit_windowed_lag(thermal, rgb)


class TestPassesAcceptance:
    def _result(self, residual, r2, slope):
        return WindowedSyncResult(
            window_centers_sec=np.array([0.0, 1.0]),
            window_lags_sec=np.array([0.0, slope]),
            offset_sec=0.0,
            drift_slope=slope,
            r_squared=r2,
            residual_max_sec=residual,
        )

    def test_passes_when_residual_small_and_r2_high(self):
        r = self._result(residual=0.05, r2=0.95, slope=0.01)
        assert passes_acceptance(r, thermal_frame_sec=0.125)

    def test_fails_when_residual_too_large(self):
        r = self._result(residual=0.2, r2=0.99, slope=0.01)
        assert not passes_acceptance(r, thermal_frame_sec=0.125)

    def test_passes_via_zero_drift_branch_despite_low_r2(self):
        r = self._result(residual=0.05, r2=0.1, slope=1e-6)
        assert passes_acceptance(r, thermal_frame_sec=0.125)

    def test_fails_when_drift_nonzero_and_r2_low(self):
        r = self._result(residual=0.05, r2=0.1, slope=0.01)
        assert not passes_acceptance(r, thermal_frame_sec=0.125)
