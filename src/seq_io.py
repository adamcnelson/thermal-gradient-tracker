"""
FLIR ResearchIR .seq file reader — read-only version for the tracking pipeline.

Format: FFF (FLIR File Format) frames concatenated sequentially.
Each frame has a variable-length header followed by 16-bit raw pixel data.

Temperature calibration
-----------------------
FLIR ResearchIR stores data in the raw-data block as radiometric uint16 values,
not raw sensor ADU counts. Each unit represents 0.04 K (25 mK resolution).

Conversion formula (confirmed from lab's seq_process pipeline):
    T_celsius = raw_uint16 * 0.04 - 273.15

Use raw_to_celsius() to convert a uint16 frame to a float32 Celsius array.
Segmentation and display use the raw uint16 values directly (no calibration
needed for relative comparisons or normalization).

Block table structure (at frame offset 0x40, entries 32 bytes each):
  Tag 0x00000020 : camera-params block
  Tag 0x00020001 : raw thermal data block with uint16 pixel values
"""

import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

FRAME_MAGIC = b"FFF\x00"

BLOCK_COUNT_OFFSET = 0x1C
BLOCK_TABLE_START = 0x40
BLOCK_ENTRY_SIZE = 32

_BTE_TAG = 0
_BTE_DATA_OFFSET = 12
_BTE_DATA_SIZE = 16

TAG_CAMERA_PARAMS = 0x00000020
TAG_RAW_DATA = 0x00020001

DATA_SUBHEADER_SIZE = 32
_SH_WIDTH_OFF = 2
_SH_HEIGHT_OFF = 4
_SH_MAX_X_OFF = 12
_SH_MAX_Y_OFF = 16


def _ru16(buf: Union[bytes, bytearray], offset: int) -> int:
    return struct.unpack_from("<H", buf, offset)[0]


def _ru32(buf: Union[bytes, bytearray], offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _parse_block_table(header: Union[bytes, bytearray]) -> List[Tuple[int, int, int]]:
    """Return list of (tag, data_offset, data_size) for every non-zero block."""
    num_blocks = _ru32(header, BLOCK_COUNT_OFFSET)
    blocks = []
    for i in range(num_blocks):
        entry_start = BLOCK_TABLE_START + i * BLOCK_ENTRY_SIZE
        tag = _ru32(header, entry_start + _BTE_TAG)
        data_offset = _ru32(header, entry_start + _BTE_DATA_OFFSET)
        data_size = _ru32(header, entry_start + _BTE_DATA_SIZE)
        if tag == 0 and data_offset == 0 and data_size == 0:
            break
        blocks.append((tag, data_offset, data_size))
    return blocks


class FrameMetadata:
    """Metadata extracted from a single FFF frame header (best-effort)."""
    __slots__ = ("frame_index", "width", "height")

    def __init__(self, frame_index: int, width: int, height: int):
        self.frame_index = frame_index
        self.width = width
        self.height = height


class SeqReader:
    """Stream frames from a FLIR ResearchIR .seq file."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        self._fh = open(self.path, "rb")
        self._width, self._height = self._detect_dims()

    def _detect_dims(self) -> Tuple[int, int]:
        self._fh.seek(0)
        probe = self._fh.read(BLOCK_TABLE_START + 16 * BLOCK_ENTRY_SIZE)
        if probe[:4] != FRAME_MAGIC:
            raise ValueError(f"Not a FLIR FFF .seq file: {self.path}")
        if not probe[4:14].startswith(b"ResearchIR"):
            raise ValueError(f"Expected ResearchIR application header: {self.path}")
        blocks = _parse_block_table(probe)
        raw = next((b for b in blocks if b[0] == TAG_RAW_DATA), None)
        if raw is None:
            raise ValueError("Raw data block (0x00020001) not found in first frame")
        _, raw_offset, _ = raw
        self._fh.seek(raw_offset)
        subheader = self._fh.read(DATA_SUBHEADER_SIZE)
        w = _ru16(subheader, _SH_WIDTH_OFF)
        h = _ru16(subheader, _SH_HEIGHT_OFF)
        if w == 0 or h == 0:
            raise ValueError(f"Invalid frame dimensions: {w}x{h}")
        return w, h

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """(height, width) — matches numpy array axis order."""
        return (self._height, self._width)

    def count_frames(self) -> int:
        """Estimate total frame count by comparing first two frame sizes."""
        self._fh.seek(0)
        data = self._fh.read(BLOCK_TABLE_START + 4 * BLOCK_ENTRY_SIZE)
        blocks = _parse_block_table(data)
        raw = next((b for b in blocks if b[0] == TAG_RAW_DATA), None)
        if raw is None:
            return 0
        _, raw_off, raw_size = raw
        frame0_size = raw_off + raw_size

        self._fh.seek(frame0_size)
        data2 = self._fh.read(BLOCK_TABLE_START + 4 * BLOCK_ENTRY_SIZE)
        if len(data2) < 4 or data2[:4] != FRAME_MAGIC:
            return 1
        blocks2 = _parse_block_table(data2)
        raw2 = next((b for b in blocks2 if b[0] == TAG_RAW_DATA), None)
        if raw2 is None:
            return 1
        _, raw_off2, raw_size2 = raw2
        frame1_size = raw_off2 + raw_size2

        file_size = self.path.stat().st_size
        remaining = file_size - frame0_size
        return 1 + remaining // frame1_size if frame1_size > 0 else 1

    def _read_one_frame(self) -> Optional[Tuple[bytearray, np.ndarray]]:
        start = self._fh.read(64)
        if not start:
            return None
        if len(start) < 64 or start[:4] != FRAME_MAGIC:
            return None

        num_blocks = _ru32(start, BLOCK_COUNT_OFFSET)
        if num_blocks == 0 or num_blocks > 32:
            return None

        table_bytes = self._fh.read(num_blocks * BLOCK_ENTRY_SIZE)
        if len(table_bytes) < num_blocks * BLOCK_ENTRY_SIZE:
            return None

        partial_header = bytearray(start + table_bytes)
        blocks = _parse_block_table(partial_header)

        raw = next((b for b in blocks if b[0] == TAG_RAW_DATA), None)
        if raw is None:
            return None
        _, raw_off, raw_size = raw

        pixel_count = (raw_size - DATA_SUBHEADER_SIZE) // 2
        bytes_read_so_far = 64 + len(table_bytes)
        header_end = raw_off + DATA_SUBHEADER_SIZE
        remaining_header = self._fh.read(header_end - bytes_read_so_far)
        if len(remaining_header) < header_end - bytes_read_so_far:
            return None

        header_bytes = bytearray(partial_header + remaining_header)
        pixel_bytes = self._fh.read(pixel_count * 2)
        if len(pixel_bytes) < pixel_count * 2:
            return None

        pixel_data = np.frombuffer(pixel_bytes, dtype=np.uint16).reshape(
            self._height, self._width
        )
        return header_bytes, pixel_data

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield (frame_index, pixel_data) for each frame.
        pixel_data is a uint16 numpy array of shape (height, width).
        """
        self._fh.seek(0)
        frame_idx = 0
        while True:
            result = self._read_one_frame()
            if result is None:
                break
            _, pixel_data = result
            yield frame_idx, pixel_data
            frame_idx += 1

    def read_frame(self, frame_index: int) -> np.ndarray:
        """Read and return a specific frame by index (slow: scans sequentially)."""
        for idx, frame in self.frames():
            if idx == frame_index:
                return frame
        raise IndexError(f"Frame {frame_index} not found in {self.path.name}")

    def read_first_frame(self) -> np.ndarray:
        self._fh.seek(0)
        result = self._read_one_frame()
        if result is None:
            raise ValueError("Could not read first frame")
        return result[1]

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def adu_to_display(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a uint16 thermal frame to uint8 for display.
    Stretches the range to fill 0–255.
    """
    lo, hi = frame.min(), frame.max()
    if hi == lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = (frame.astype(np.float32) - lo) / (hi - lo) * 255.0
    return scaled.clip(0, 255).astype(np.uint8)


def read_planck_constants(seq_path: str) -> Optional[dict]:
    """
    Use exiftool to read FLIR Planck calibration constants from a .seq file.

    Returns a dict with keys R1, R2, B, O, F, or None if exiftool is not
    available or the constants cannot be found.

    Install exiftool on macOS: brew install exiftool
    """
    import subprocess, json as _json

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-PlanckR1", "-PlanckR2", "-PlanckB", "-PlanckO", "-PlanckF",
                "-json", str(seq_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        data = _json.loads(result.stdout)
        if not data:
            return None
        meta = data[0]
        constants = {
            "R1": float(meta["PlanckR1"]),
            "R2": float(meta["PlanckR2"]),
            "B":  float(meta["PlanckB"]),
            "O":  float(meta["PlanckO"]),
            "F":  float(meta["PlanckF"]),
        }
        return constants
    except (KeyError, ValueError, _json.JSONDecodeError):
        return None


def raw_to_celsius(frame: np.ndarray, planck: Optional[dict] = None) -> np.ndarray:
    """
    Convert a raw uint16 frame to degrees Celsius (float32).

    If Planck constants are provided (read via read_planck_constants), applies
    the FLIR Planck equation:
        T_celsius = B / ln(R1 / (R2 * (raw + O)) + F) - 273.15

    If no constants are given, falls back to the linear approximation
        T_celsius = raw * 0.04 - 273.15
    which is only correct for flirpy-exported radiometric TIFFs, NOT for raw
    .seq values. Always prefer providing Planck constants.
    """
    import numpy as _np

    f = frame.astype(_np.float64)

    if planck is not None:
        R1, R2, B, O, F = planck["R1"], planck["R2"], planck["B"], planck["O"], planck["F"]
        temp_k = B / _np.log(R1 / (R2 * (f + O)) + F)
        return (temp_k - 273.15).astype(_np.float32)

    # Fallback — warns at call site in tracking.py
    return (f * 0.04 - 273.15).astype(_np.float32)
