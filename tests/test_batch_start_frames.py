"""Tests for process_batch's per-file start_frames override (src/batch.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.batch as batch
import src.seq_io as seq_io
from src.arena_mask import TrackingConfig


class _FakeReader:
    frame_shape = (43, 412)

    def __init__(self, path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_track_file(seq_path, config, **kwargs):
        seen[Path(seq_path).stem] = config
        return None, f"{seq_path}.csv", {}

    monkeypatch.setattr(seq_io, "SeqReader", _FakeReader)
    monkeypatch.setattr(batch.ArenaMask, "from_config", staticmethod(lambda config, shape: None))
    monkeypatch.setattr(batch, "track_file", fake_track_file)
    return seen


def _run(tmp_path, start_frames):
    files = [tmp_path / "A_Front.seq", tmp_path / "A_Back.seq"]
    config = TrackingConfig()
    results = batch.process_batch(files, config, str(tmp_path / "out"), save_qc_summary=False,
                                  save_plots=False, start_frames=start_frames)
    return config, results


def test_listed_file_gets_manual_start_frame_others_keep_config(tmp_path, captured):
    config, results = _run(tmp_path, {"A_Front": 401})
    assert all(r["status"] == "ok" for r in results)
    assert captured["A_Front"].manual_tracking_start_frame == 401
    assert captured["A_Back"] is config
    assert config.manual_tracking_start_frame is None  # shared config not mutated


def test_no_start_frames_is_unchanged_behavior(tmp_path, captured):
    config, _ = _run(tmp_path, None)
    assert captured["A_Front"] is config and captured["A_Back"] is config
