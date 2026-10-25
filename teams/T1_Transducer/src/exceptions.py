"""
exceptions.py
Custom exception hierarchy for Piezo operations.
"""

class PiezoError(Exception):
    """Base exception for all Piezo-related errors."""
    pass


class PiezoValidationError(PiezoError, ValueError):
    """Raised when frequency ranges or attributes fail validation checks."""
    pass


class PiezoNotFoundError(PiezoError, KeyError):
    """Raised when a requested material is not found in the registry."""
    pass