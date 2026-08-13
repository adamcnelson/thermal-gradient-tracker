"""
Unit tests for the raw-radiometric conversion path — v7 brief §6 Stage 0
deliverable ("read_seq_radiometric() + a unit test against a frame of
known temperature").

seq_io.raw_to_celsius() + read_planck_constants() together already ARE
that reader (tracking.py has called them since before this branch) — this
file is what was missing: a test that (a) pins the Planck-equation math
itself, independent of any real file, and (b) confirms against a real
.seq file that exiftool actually resolves Planck constants rather than
silently falling back to the linear approximation, which tracking.py only
warns about in a log line that's easy to miss in a batch run.

Caveat: this is not calibration validation against a physical reference
target (no such ground-truth-temperature footage exists for this rig) —
it verifies the formula is implemented correctly and that real files
resolve real constants, not that the resulting °C values are accurate.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.seq_io import raw_to_celsius, read_planck_constants

# A real cropped session file, used only to confirm exiftool/Planck resolution
# against genuine FLIR metadata. Skipped (not failed) if unavailable, since
# it lives outside the repo and is desktop-local dev data, not a portable
# fixture — see project_brief_v7.md §10 for the dev-data convention.
REAL_SEQ_PATH = Path(
    "/Users/adamnelson/Documents/ClaudeCode/ThermalGradient/croppedSeqFiles/"
    "07-28-25_Test_3/07-28-25_4540_B_4541_F_Test3-004_Front.seq"
)


class TestRawToCelsiusPlanckFormula:
    def test_matches_manual_planck_computation(self):
        planck = {"R1": 21714.436, "R2": 0.018001635, "B": 1463.3, "O": -4382, "F": 1.05}
        raw = np.array([[13812]], dtype=np.uint16)

        expected_k = planck["B"] / math.log(
            planck["R1"] / (planck["R2"] * (13812 + planck["O"])) + planck["F"]
        )
        expected_c = expected_k - 273.15

        result = raw_to_celsius(raw, planck)
        assert result.dtype == np.float32
        assert result[0, 0] == pytest.approx(expected_c, abs=1e-3)

    def test_monotonic_in_raw_value(self):
        """Higher raw counts must map to higher temperature (sanity guard on formula sign)."""
        planck = {"R1": 21714.436, "R2": 0.018001635, "B": 1463.3, "O": -4382, "F": 1.05}
        raw = np.array([[10000, 20000, 30000]], dtype=np.uint16)
        result = raw_to_celsius(raw, planck)
        assert result[0, 0] < result[0, 1] < result[0, 2]

    def test_no_planck_falls_back_to_documented_linear_approximation(self):
        raw = np.array([[10000]], dtype=np.uint16)
        result = raw_to_celsius(raw, planck=None)
        assert result[0, 0] == pytest.approx(10000 * 0.04 - 273.15, abs=1e-4)


@pytest.mark.skipif(not REAL_SEQ_PATH.exists(), reason="Real .seq dev fixture not present on this machine")
class TestRealFilePlanckResolution:
    def test_planck_constants_resolve_not_silently_falling_back(self):
        planck = read_planck_constants(str(REAL_SEQ_PATH))
        assert planck is not None, (
            "exiftool failed to resolve Planck constants — tracking.py would silently "
            "fall back to the linear approximation and produce wrong temperatures."
        )
        for key in ("R1", "R2", "B", "O", "F"):
            assert key in planck
            assert isinstance(planck[key], float)

    def test_real_constants_match_confirmed_values(self):
        """
        Pins the specific constants confirmed via `exiftool -PlanckR1 ... -json`
        during the Stage 0 audit (2026-08-12), so a future exiftool/file change
        that silently alters calibration is caught rather than passing quietly.
        """
        planck = read_planck_constants(str(REAL_SEQ_PATH))
        assert planck["R1"] == pytest.approx(21714.436)
        assert planck["R2"] == pytest.approx(0.018001635)
        assert planck["B"] == pytest.approx(1463.3)
        assert planck["O"] == pytest.approx(-4382)
        assert planck["F"] == pytest.approx(1.05)
