"""
pipeline/semantics/synthesis.py
=================================
Organic synthesis validator plugin — STUB.

When implemented, validates limiting-reagent / theoretical-yield consistency:
    n_limiting = mass / M_limiting
    theoretical_yield = n_limiting * M_product
    percent_yield = (actual_mass / theoretical_yield) * 100

Register with: @plugin("synthesis", schema="schema/synthesis.schema.json",
                        required=["limiting_reagent_mass", "M_limiting",
                                  "M_product", "actual_mass"])
"""

from __future__ import annotations

# from pipeline.registry import plugin, value_of
