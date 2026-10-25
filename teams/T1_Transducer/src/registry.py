"""
registry.py
Registry class for managing multiple material objects and global registry instance.
"""

from piezo import Piezo
from exceptions import PiezoNotFoundError


class PiezoRegistry:
    """Dynamic registry to store, look up, and manage multiple Piezo materials."""

    def __init__(self):
        self._materials: dict[str, Piezo] = {}

    def add_material(self, name: str, freq_range: tuple[float, float] = (0.0, 0.0)) -> Piezo:
        """Creates a material and adds it to the registry."""
        material_obj = Piezo(name=name, freq_range=freq_range)
        self._materials[name.lower()] = material_obj
        return material_obj

    def get_material(self, name: str) -> Piezo:
        """Looks up a material by name (case-insensitive). Raises PiezoNotFoundError if missing."""
        material_obj = self._materials.get(name.lower())
        if not material_obj:
            raise PiezoNotFoundError(f"Material '{name}' was not found in the registry.")
        return material_obj

    def update_attributes(self, name: str, freq_range: tuple[float, float]) -> None:
        """Updates frequency range attributes of an existing material."""
        material_obj = self.get_material(name)
        material_obj.freq_range = freq_range

    def list_materials(self) -> list[str]:
        """Returns a list of all registered material names."""
        return [mat.name for mat in self._materials.values()]

    def clear(self) -> None:
        """Clears all stored materials from the registry."""
        self._materials.clear()


# Shared global instance used across modules
registry = PiezoRegistry()