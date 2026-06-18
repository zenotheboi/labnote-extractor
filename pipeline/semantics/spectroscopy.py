"""
pipeline/semantics/spectroscopy.py
=====================================
Spectroscopy (UV-vis / absorbance) validator plugin — STUB.

When implemented, validates Beer-Lambert-law consistency:
    A = ε · c · l
    A   : absorbance (dimensionless)
    ε   : molar absorptivity (L mol⁻¹ cm⁻¹)
    c   : concentration (mol/L)
    l   : path length (cm)

Register with: @plugin("spectroscopy", schema="schema/spectroscopy.schema.json",
                        required=["absorbance", "concentration", "path_length"])
"""

from __future__ import annotations

# from pipeline.registry import plugin, value_of
