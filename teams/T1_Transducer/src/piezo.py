"""
piezo.py
Piezo data class with property setter validations.
"""

from exceptions import PiezoValidationError


class Piezo:
    """Represents a piezoelectric material with validated frequency range properties."""

    def __init__(self, name: str, freq_range: tuple[float, float] = (0.0, 0.0)):
        self.name = name
        self.freq_range = freq_range

    @property
    def freq_range(self) -> tuple[float, float]:
        return self._freq_range

    @freq_range.setter
    def freq_range(self, value: tuple[float, float]) -> None:
        """Validates that frequency range consists of two non-negative numbers where min <= max."""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise PiezoValidationError(
                "freq_range must be a tuple or list of 2 numbers (min_freq, max_freq)."
            )

        try:
            min_freq, max_freq = float(value[0]), float(value[1])
        except (ValueError, TypeError) as e:
            raise PiezoValidationError(f"Frequency values must be numeric. Details: {e}")

        if min_freq < 0 or max_freq < 0:
            raise PiezoValidationError(
                f"Frequencies must be non-negative. Got: ({min_freq}, {max_freq})"
            )

        if min_freq > max_freq:
            raise PiezoValidationError(
                f"Minimum frequency ({min_freq}) cannot exceed maximum frequency ({max_freq})."
            )

        self._freq_range = (min_freq, max_freq)