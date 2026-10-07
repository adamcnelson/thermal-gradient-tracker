"""Tests for scripts/build_stage7_inherited_config.py's session-label parsing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_stage7_inherited_config import mouse_id_for_lane


@pytest.mark.parametrize("label,lane,expected", [
    ("07-08-25_4539_B_4540_F", "B", 4539),
    ("07-08-25_4539_B_4540_F", "F", 4540),
    ("08-19-25_4541_F_4550_B_Test9", "B", 4550),
    ("07-31-25_4547_F_4552_B_Test4-009", "F", 4547),
    ("07-11-25_4552_B", "B", 4552),
])
def test_mouse_id_for_lane(label, lane, expected):
    assert mouse_id_for_lane(label, lane) == expected


def test_missing_lane_raises():
    with pytest.raises(ValueError):
        mouse_id_for_lane("07-11-25_4552_B", "F")
