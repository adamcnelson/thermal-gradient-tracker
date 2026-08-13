"""
Golden-output regression test for the tracking stage — v7 brief §5 item 4.

Runs the existing (pre-landmarks) tracking pipeline against a small
deterministic synthetic .seq fixture and asserts the output matches a
stored golden CSV. This is what proves standing tracking-stage behavior
survives v7 landmark work; it is not itself proof that landmarks work,
only that nothing outside src/landmarks/ silently changed.

Scope: covers track_file() (the tracking stage) only, since that is the
stage landmark extraction sits beside per the brief's "annotation layer on
existing bout output" design (Stage 1) — bout detection and analysis are
explicitly reused as-is and are not touched by this branch.

No .seq file required — the fixture is generated on the fly by
tests/fixtures/synth_seq.py, so this test runs identically on ARCC/CI.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arena_mask import ArenaMask, TrackingConfig
from src.tracking import track_file
from tests.fixtures.synth_seq import make_moving_blob_frames, write_synthetic_seq

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "golden" / "tracking_golden.csv"
FRAME_SHAPE = (30, 50)  # (height, width)

# Numeric tolerance for floats stored at %.4f precision in the golden CSV.
FLOAT_ATOL = 1e-3


def _fixture_config() -> TrackingConfig:
    return TrackingConfig(
        sampling_interval_frames=1,
        mouse_roi_radius_px=3,
        floor_roi_radius_px=3,
        max_floor_roi_shift_px=10,
        floor_roi_search_step_px=1,
        min_mouse_area_px=10,
        max_mouse_area_px=300,
        training_frame_count=5,
        random_seed=123,
        background_n_frames=8,
        segmentation_threshold_sigma=2.0,
        camera_fps=10.0,
        auto_detect_tracking_start=False,
        enable_local_fallback=True,
        local_fallback_search_radius_px=15,
        local_fallback_threshold_percentile=80.0,
        local_fallback_min_area_px=10,
    )


def _run_fixture(tmp_path: Path) -> pd.DataFrame:
    seq_path = tmp_path / "test_synth.seq"
    write_synthetic_seq(str(seq_path), make_moving_blob_frames())

    arena_mask = ArenaMask.from_full_image(FRAME_SHAPE, border_px=1)
    _df, csv_path, _entry_info = track_file(
        seq_path=str(seq_path),
        config=_fixture_config(),
        arena_mask=arena_mask,
        output_dir=str(tmp_path / "out"),
        overwrite=True,
        save_qc_images=False,
    )
    # Read back through CSV (rather than using the in-memory df) so dtype
    # inference matches the golden file, which is itself CSV-round-tripped.
    return pd.read_csv(csv_path)


class TestTrackingGoldenRegression:
    def test_matches_golden_output(self, tmp_path):
        golden = pd.read_csv(GOLDEN_CSV)
        actual = _run_fixture(tmp_path)

        assert list(actual.columns) == list(golden.columns), (
            "Tracking output columns changed — this is a schema change and "
            "must be called out explicitly, not silently absorbed by this test."
        )
        assert len(actual) == len(golden)

        numeric_cols = golden.select_dtypes(include="number").columns
        other_cols = [c for c in golden.columns if c not in numeric_cols]

        pd.testing.assert_frame_equal(
            actual[numeric_cols].reset_index(drop=True),
            golden[numeric_cols].reset_index(drop=True),
            check_dtype=False,
            atol=FLOAT_ATOL,
        )
        pd.testing.assert_frame_equal(
            actual[other_cols].reset_index(drop=True),
            golden[other_cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_fixture_is_a_meaningful_regression_probe(self, tmp_path):
        """
        Guard against the fixture silently degrading into a no-op (e.g. mouse
        never detected) that would make the equality check above vacuous.
        """
        df = _run_fixture(tmp_path)
        assert (df["qc_flag"] == "ok").all()
        assert df["mouse_roi_valid"].all()
        assert df["floor_roi_valid"].all()
