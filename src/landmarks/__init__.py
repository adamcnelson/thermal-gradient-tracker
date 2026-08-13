"""
Feature-specific thermal landmark extraction (v7).

Additive subpackage per project_brief_v7.md §5: geometry is derived from
RGB, radiometry is sampled from thermal, and this subpackage never reaches
back into the existing tracking/bout modules. Everything here is inert
until wired in behind LandmarksConfig.enabled (see config.py) — that hook
does not exist yet.
"""
