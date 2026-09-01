"""Load and save analysis configuration for bouts + treatment-effect analysis."""

import json
from pathlib import Path
from typing import List, Optional  # List used by AnalysisParams

from pydantic import BaseModel, Field


class BoutsConfig(BaseModel):
    position_smoothing_window_samples: int = Field(
        5,
        description=(
            "Rolling-median smoothing window for centroid-x, applied before dispersion is "
            "computed from it. Denoises brief per-frame jitter/dropout so it doesn't "
            "spuriously fragment a real rest bout."
        ),
    )
    dispersion_window_samples: int = Field(6, description="Rolling window width (samples)")
    dispersion_metric: str = Field("mad", description="mad | iqr | sd")
    stationary_dispersion_threshold_px: Optional[float] = Field(
        5.0, description="Centroid-x rolling dispersion threshold (px); edit in analysis_config.json"
    )
    velocity_smoothing_window_samples: int = Field(5, description="Velocity smoothing window")
    stationary_velocity_threshold_px_per_s: Optional[float] = Field(
        5.0, description="Velocity guard threshold (px/s); edit in analysis_config.json"
    )
    qc_valid_only_for_dispersion: bool = Field(
        True, description="Use only qc_flag==ok frames for dispersion"
    )
    min_bout_duration_sec: float = Field(10.0, description="Minimum stationary bout length")
    max_gap_merge_sec: float = Field(3.0, description="Max gap to merge two bouts")
    max_unknown_gap_merge_sec: float = Field(
        10.0,
        description=(
            "Max gap to merge two bouts when EVERY sample in the gap has qc_flag != 'ok' "
            "(no real measurement -- e.g. a brief no_mouse_roi dropout), as opposed to "
            "max_gap_merge_sec which applies when the gap contains real evidence of movement. "
            "A brief no_mouse_roi gap fragmenting real, long stationary bouts was found and "
            "measured directly (2026-08-25, project_v7 bout-fragmentation investigation): "
            "no_mouse_roi hits 1.6% of Test_3's frames and 16% of Test_4's, in runs mostly "
            "<=7.5s with a small number of much longer (10s+) runs that are real, extended "
            "tracking failures, not brief dropouts, and should NOT be bridged. 10.0s was chosen "
            "as the boundary between these two populations in that real data, not guessed."
        ),
    )
    frame_rate_fps: float = Field(
        8.0,
        description=(
            "Nominal camera frame rate. Not currently consumed by the pipeline — bout timing "
            "is derived entirely from elapsed_time_sec in the tracking CSV, which is computed "
            "from TrackingConfig.camera_fps. Kept here for documentation/future use; keep it "
            "in sync with camera_fps in the tracking config actually used for a given session. "
            "The '8 is the confirmed true rate' claim (v7 Stage 0 audit, 2026-08-12, from the "
            ".seq EXIF FrameRate tag) is CONTRADICTED by real cross-modal timing evidence found "
            "2026-08-25 for Test_3/Test_4 (human-verified anchors spanning full sessions cleanly "
            "imply true fps=10.0 in both -- see project memory). tracking_config_test3.json / "
            "tracking_config_test4.json were corrected to camera_fps=10.0; this shared default "
            "and the rest of the session corpus have NOT been re-audited yet."
        ),
    )


class AnalysisParams(BaseModel):
    group_factors: List[str] = Field(default_factory=lambda: ["virus", "injection", "phase"])
    primary_contrast: dict = Field(
        default_factory=lambda: {"factor": "injection", "levels": ["DCZ", "Vehicle"]}
    )
    random_effect: str = Field("mouse_id")
    stationary_states: List[str] = Field(
        default_factory=lambda: ["stationary", "non_stationary"]
    )
    trial_length_reference_sec: int = Field(3000)
    min_animals_for_lmm: int = Field(3)
    min_obs_per_group_for_lmm: int = Field(2)


class AnalysisConfig(BaseModel):
    bouts: BoutsConfig = Field(default_factory=BoutsConfig)
    analysis: AnalysisParams = Field(default_factory=AnalysisParams)

    @classmethod
    def load(cls, path: str) -> "AnalysisConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Analysis config not found: {path}\n"
                "Copy analysis_config.json from the project root."
            )
        with open(p) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2)
