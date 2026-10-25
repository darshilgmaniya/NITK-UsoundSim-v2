"""
add_attributes.py
Function to add/update material attributes in the registry.
"""

from registry import registry
from exceptions import PiezoNotFoundError


def addNewAttributes(name_of_material: str, freq_range: tuple[float, float]) -> None:
    """Updates property values for a named material in the registry."""
    try:
        registry.update_attributes(name_of_material, freq_range)
    except PiezoNotFoundError:
        registry.add_material(name_of_material, freq_range)