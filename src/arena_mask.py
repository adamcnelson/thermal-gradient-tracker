"""
Arena mask and tracking configuration.

The arena mask marks pixels where the floor ROI may be placed — inside the
thermal-gradient track, excluding walls, apparatus edges, and image borders.

The mask polygon is stored inside tracking_config.json so the same config
file serves as both the processing parameter set and the spatial mask.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    """All pipeline parameters, plus the arena polygon for this video geometry."""

    sampling_interval_frames: int = Field(10, description="Sample every N frames")
    mouse_roi_radius_px: int = Field(8, description="Radius of mouse surface ROI in pixels")
    floor_roi_radius_px: int = Field(8, description="Radius of adjacent floor ROI in pixels")
    max_floor_roi_shift_px: int = Field(80, description="Max vertical search distance for floor ROI")
    floor_roi_search_step_px: int = Field(2, description="Step size for vertical floor ROI search")
    min_mouse_area_px: int = Field(100, description="Minimum mouse blob area to accept")
    max_mouse_area_px: int = Field(10000, description="Maximum mouse blob area to accept")
    training_frame_count: int = Field(50, description="Frames to sample for manual training")
    random_seed: int = Field(123, description="Random seed for reproducible frame sampling")
    background_n_frames: int = Field(
        200, description="Number of frames used to build background model"
    )
    segmentation_threshold_sigma: float = Field(
        3.0,
        description=(
            "Foreground detection threshold: mean + N * std of the background-difference image. "
            "Increase to reduce false positives; decrease to detect faint mice."
        ),
    )
    # Mouse-entry detection
    auto_detect_tracking_start: bool = Field(
        True,
        description=(
            "Scan forward to find the first frame where the mouse is stably present. "
            "Frames before tracking_start_frame are recorded as NA in the output CSV."
        ),
    )
    min_stable_mouse_frames: int = Field(
        50,
        description=(
            "Number of consecutive sampled frames that must show a stable mouse "
            "before tracking formally begins."
        ),
    )
    tracking_start_confidence_threshold: float = Field(
        0.8,
        description="Minimum segmentation confidence required during the stable-entry window.",
    )
    max_entry_disturbance_score: float = Field(
        0.2,
        description=(
            "Maximum allowed disturbance score (0–1) during entry detection. "
            "Disturbance is high when the total foreground area greatly exceeds "
            "the expected mouse size, e.g. a researcher's hand is in frame."
        ),
    )
    manual_tracking_start_frame: Optional[int] = Field(
        None,
        description=(
            "Override auto-detection by manually setting the first frame to track. "
            "Set via --tracking-start-frame on the command line, or edit this value directly."
        ),
    )
    camera_fps: float = Field(
        10.0,
        description="Real camera acquisition frame rate. Used to convert frame_number to elapsed_time_sec.",
    )

    # Local-fallback recovery (Issue 1): when the global adaptive threshold finds no
    # qualifying blob, re-search a small window around the last known centroid at a
    # lower, locally-computed threshold. Recovers the mouse when its surface
    # temperature nearly matches the floor and the contrast is too faint to clear
    # the global threshold, but is still detectable within a small local window.
    enable_local_fallback: bool = Field(
        True,
        description="Enable local-fallback recovery when the global threshold loses the mouse.",
    )
    local_fallback_search_radius_px: int = Field(
        25,
        description="Half-width/height (symmetric) of the local search window around the last centroid.",
    )
    local_fallback_threshold_percentile: float = Field(
        80.0,
        description="Percentile of the local window's foreground score used as the fallback threshold.",
    )
    local_fallback_min_area_px: int = Field(
        30,
        description="Minimum blob area to accept during local-fallback recovery (lower than min_mouse_area_px).",
    )

    # Set automatically when the arena mask is created; null means full image
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    arena_polygon: Optional[List[List[int]]] = Field(
        None,
        description=(
            "Polygon vertices [[x, y], ...] marking the valid floor area. "
            "Null means the full image (minus a 2-pixel border) is used. "
            "Set this with scripts/create_arena_mask.py before tracking."
        ),
    )

    @classmethod
    def load(cls, path: str) -> "TrackingConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Tracking config not found: {path}\n"
                "Copy tracking_config.example.json and edit it, or run "
                "scripts/create_arena_mask.py to create one."
            )
        with open(p) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2)
        print(f"Config saved: {path}")

    def update_polygon(self, path: str, polygon_xy: List[List[int]], h: int, w: int) -> None:
        """Update the arena polygon in an existing config file (or create it)."""
        self.arena_polygon = polygon_xy
        self.image_height = h
        self.image_width = w
        self.save(path)


class ArenaMask:
    """
    Boolean mask where True = valid floor pixel for ROI placement.

    Created from a user-drawn polygon saved in tracking_config.json.
    If no polygon is defined, the full image minus a 2-pixel border is used.
    """

    def __init__(self, mask: np.ndarray, polygon_xy: Optional[List[List[int]]] = None):
        self._mask = mask.astype(bool)
        self._polygon = polygon_xy

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def shape(self) -> Tuple[int, int]:
        return self._mask.shape

    @property
    def polygon(self) -> Optional[List[List[int]]]:
        return self._polygon

    @classmethod
    def from_polygon(
        cls,
        polygon_xy: List[List[int]],
        image_shape: Tuple[int, int],
    ) -> "ArenaMask":
        """Create a mask from a list of (x, y) polygon vertices."""
        h, w = image_shape
        pts = np.array(polygon_xy, dtype=np.int32)
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], 1)
        return cls(canvas.astype(bool), polygon_xy=polygon_xy)

    @classmethod
    def from_full_image(cls, image_shape: Tuple[int, int], border_px: int = 2) -> "ArenaMask":
        """Full-image mask with a small border excluded (safe default)."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=bool)
        mask[border_px : h - border_px, border_px : w - border_px] = True
        return cls(mask, polygon_xy=None)

    @classmethod
    def from_config(
        cls,
        config: "TrackingConfig",
        image_shape: Tuple[int, int],
    ) -> "ArenaMask":
        """Build the arena mask from a TrackingConfig."""
        if config.arena_polygon is None:
            return cls.from_full_image(image_shape)
        return cls.from_polygon(config.arena_polygon, image_shape)

    def validate_shape(self, image_shape: Tuple[int, int]) -> None:
        """Raise ValueError if this mask does not match the video dimensions."""
        if self._mask.shape != image_shape:
            raise ValueError(
                f"Arena mask shape {self._mask.shape} does not match "
                f"video frame shape {image_shape}. "
                "Recreate the mask with scripts/create_arena_mask.py."
            )
