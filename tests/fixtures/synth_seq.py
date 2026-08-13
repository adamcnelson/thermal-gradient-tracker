"""
Build a minimal synthetic FLIR FFF .seq file for pipeline tests.

Implements only the subset of the format that src/seq_io.py's SeqReader
parses (one raw-data block per frame, no camera-params block) — enough to
exercise the real .seq read path end-to-end without needing an actual
recorded .seq file.
"""

import struct
from typing import List

import numpy as np

FRAME_MAGIC = b"FFF\x00"
APP_HEADER = b"ResearchIR"
TAG_RAW_DATA = 0x00020001
BLOCK_TABLE_START = 0x40
BLOCK_ENTRY_SIZE = 32
DATA_SUBHEADER_SIZE = 32


def _frame_bytes(pixels: np.ndarray) -> bytes:
    height, width = pixels.shape
    raw_offset = BLOCK_TABLE_START + BLOCK_ENTRY_SIZE  # one block-table entry, no gap
    data_size = DATA_SUBHEADER_SIZE + pixels.size * 2

    header = bytearray(BLOCK_TABLE_START)
    header[0:4] = FRAME_MAGIC
    header[4:14] = APP_HEADER
    header[0x1C:0x20] = struct.pack("<I", 1)  # num_blocks

    block_entry = bytearray(BLOCK_ENTRY_SIZE)
    block_entry[0:4] = struct.pack("<I", TAG_RAW_DATA)
    block_entry[12:16] = struct.pack("<I", raw_offset)
    block_entry[16:20] = struct.pack("<I", data_size)

    subheader = bytearray(DATA_SUBHEADER_SIZE)
    subheader[2:4] = struct.pack("<H", width)
    subheader[4:6] = struct.pack("<H", height)

    pixel_bytes = pixels.astype("<u2").tobytes()

    return bytes(header) + bytes(block_entry) + bytes(subheader) + pixel_bytes


def write_synthetic_seq(path: str, frames: List[np.ndarray]) -> None:
    """Write a list of uint16 (height, width) arrays as a synthetic .seq file."""
    with open(path, "wb") as f:
        for frame in frames:
            f.write(_frame_bytes(frame.astype(np.uint16)))


def make_moving_blob_frames(
    n_frames: int = 12,
    height: int = 30,
    width: int = 50,
    blob_radius: int = 4,
    floor_base: int = 1000,
    floor_gradient_per_px: int = 10,
    blob_delta: int = 800,
    start_x: int = 10,
    step_x: int = 3,
    cy: int = 15,
) -> List[np.ndarray]:
    """
    Deterministic synthetic scene: a horizontal raw-value gradient (stand-in
    for the thermal-gradient floor) with a warm circular blob sweeping
    left-to-right at a fixed row, one step per frame. The blob moves so no
    single pixel is warm in every sampled frame — otherwise it would bias
    into the temporal-median background model instead of standing out
    against it.
    """
    x = np.arange(width)
    row = (floor_base + x * floor_gradient_per_px).astype(np.uint16)
    floor = np.tile(row, (height, 1))

    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for t in range(n_frames):
        frame = floor.copy()
        cx = start_x + t * step_x
        blob = (xx - cx) ** 2 + (yy - cy) ** 2 <= blob_radius**2
        frame[blob] = frame[blob] + blob_delta
        frames.append(frame)
    return frames
