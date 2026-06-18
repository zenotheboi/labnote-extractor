"""
reference — external-data lookups for chemistry constants and compound properties.

Provides:
    chem_resolve.resolve(name) → {smiles, formula, molar_mass, z, source}
    constants                  → physical constants (FARADAY, AVOGADRO, …)

Separation from pipeline keeps the physics clean: validators consume values
from this layer rather than hardcoding magic numbers.
"""
