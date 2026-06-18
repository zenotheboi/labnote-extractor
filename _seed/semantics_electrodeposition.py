"""
semantics/electrodeposition.py
==============================
The electrodeposition plugin: adapts the unified `fields` dict (from the
perception layer) into the pure-physics validate_deposition() and registers
itself with the registry. validate_deposition stays untouched and testable;
this thin layer just does the field mapping + registration.

In the repo, imports become:
    from pipeline.registry import plugin, value_of
    from pipeline.semantics.electrodeposition import validate_deposition
Here (flat /home/claude) we import locally for the standalone demo.
"""

from registry import plugin, value_of
from electrochem_validator import validate_deposition


@plugin("electrodeposition",
        schema="schema/electrodeposition.schema.json",
        required=["J_mA_cm2", "area_cm2", "t_min"])
def electrodeposition_validator(fields: dict):
    inputs = {
        "J_mA_cm2": value_of(fields, "J_mA_cm2"),
        "area_cm2": value_of(fields, "area_cm2"),
        "t_min":    value_of(fields, "t_min"),
        # z and molar mass SHOULD come from reference/chem_resolve (PubChem);
        # fall back to Li defaults only if the resolver didn't supply them.
        "z":        value_of(fields, "z", 1),
        "M_g_mol":  value_of(fields, "M_g_mol", 6.94),
    }
    extracted = {
        k: value_of(fields, k)
        for k in ("I_A", "t_s", "Q_C", "n_mol", "mass_g")
        if value_of(fields, k) is not None
    }
    return validate_deposition(inputs, extracted)
