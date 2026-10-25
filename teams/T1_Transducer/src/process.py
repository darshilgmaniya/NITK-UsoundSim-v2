"""
process.py
Function for executing the main operational block logic.
"""

import random
from registry import registry
from new_piezo import newPiezoMaterial
from add_attributes import addNewAttributes


def process_material(
    name_of_material: str,
    freq_range: tuple[float, float] = (0.0, 0.0),
    create_new: bool = True,
) -> float:
    """
    Main processing function to create/update material definitions
    and extract a random float frequency within specified boundaries.
    """
    if create_new:
        newPiezoMaterial(name_of_material)
        addNewAttributes(name_of_material, freq_range)

    piezo_obj = registry.get_material(name_of_material)
    min_freq, max_freq = piezo_obj.freq_range

    return random.uniform(min_freq, max_freq)