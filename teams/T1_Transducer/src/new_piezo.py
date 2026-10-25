"""
new_piezo.py
Function to instantiate and register a new Piezo material.
"""

from piezo import Piezo
from registry import registry


def newPiezoMaterial(name_of_material: str) -> Piezo:
    """Creates a new Piezo object and registers it."""
    return registry.add_material(name=name_of_material, freq_range=(0.0, 0.0))