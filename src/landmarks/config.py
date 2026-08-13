"""Config flag gating the v7 landmarks feature. Defaults to off (brief §5)."""

from pydantic import BaseModel, Field


class LandmarksConfig(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Opt-in switch for landmark-specific thermal measurement (warm spot, "
            "tail-base ΔT). False leaves the existing tracking/bout pipeline "
            "untouched. Flip only once the Stage 0 audit and golden regression "
            "test (see project_brief_v7.md) both pass."
        ),
    )
