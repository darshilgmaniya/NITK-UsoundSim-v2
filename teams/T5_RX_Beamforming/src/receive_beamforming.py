"""Compatibility shim: lets the Receiver BF team's file ``liver_echo_simulation.py`` run unchanged
(``from receive_beamforming import das_beamform``)."""
from receiver_beamforming.beamformer import das_beamform, apodization_window  # noqa: F401

__all__ = ["das_beamform", "apodization_window"]
