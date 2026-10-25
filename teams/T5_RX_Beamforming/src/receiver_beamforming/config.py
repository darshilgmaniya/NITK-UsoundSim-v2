"""
config.py - settings of the Receiver Beamforming module
=======================================================

Two small dataclasses:

* ``BeamformConfig``  - the *geometry/medium* object handed to ``das_beamform``.  The attribute names
  (SPEED_OF_SOUND, SAMPLING_FREQUENCY, MAX_DEPTH, ...) are the same ones the Receiver BF team's own
  ``LiverConfig`` uses, so ``das_beamform`` accepts either object.
* ``RxConfig``        - the *user-facing* options of the full receive chain (lines, FOV, apodisation, ...).

Everything that describes the array (c, fs, f0, pitch, element count) is normally read from the
hand-off file produced by the relay, so the beamformer follows whatever array the upstream teams used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

APODIZATIONS = ("hann", "hamming", "blackman", "tukey", "rect")
TX_MODES = ("vertical", "nearest")


@dataclass
class BeamformConfig:
    """Geometry / medium description used by ``das_beamform`` (same field names as the team's LiverConfig)."""
    SPEED_OF_SOUND: float = 1540.0        # m/s
    CENTER_FREQUENCY: float = 5.0e6       # Hz
    SAMPLING_FREQUENCY: float = 40.0e6    # Hz
    N_ELEMENTS: int = 128
    ELEMENT_PITCH: float = 0.154e-3       # m
    MAX_DEPTH: float = 0.040              # m, deepest sample of the beamformed image
    START_TIME: float = 0.0               # s, first sample time of the raw data
    RX_APODIZATION: str = "hann"
    RX_F_NUMBER: float = 1.0
    TX_MODE: str = "vertical"             # 'vertical': t_tx = z/c   'nearest': earliest element arrival
    TX_DELAYS: Optional[np.ndarray] = None  # per-element transmit delays [s] (used with 'nearest')

    def get_depth_axis(self) -> np.ndarray:
        """Depth samples, dz = c / (2 fs)  (identical to the team's definition)."""
        dz = self.SPEED_OF_SOUND / (2.0 * self.SAMPLING_FREQUENCY)
        return np.arange(0.0, self.MAX_DEPTH, dz)


@dataclass
class RxConfig:
    """User-facing options of the receive chain (all optional; defaults work for any input)."""
    apodization: str = "hann"             # one of APODIZATIONS
    f_number: Optional[float] = 1.0       # receive F-number (dynamic aperture); None/0 = full aperture
    n_lines: Optional[int] = None         # number of image lines; None -> from line_spacing_m
    line_spacing_m: Optional[float] = None  # None -> element pitch
    x_range_m: Optional[tuple] = None     # (x_min, x_max); None -> auto from scatterers / aperture
    z_range_m: Optional[tuple] = None     # (z_min, z_max); None -> auto from scatterers / data
    max_lines: int = 600                  # safety cap
    apply_focus: bool = True              # dynamic receive focusing on/off
    tx_mode: str = "vertical"             # see BeamformConfig.TX_MODE
    tx_delays: Optional[np.ndarray] = None
    dynamic_range_db: float = 50.0        # B-mode preview only
    normalize: bool = True                # divide by the sum of apodisation weights
    beamformer: Optional[Callable] = None  # plug-in: any function with the das_beamform signature

    def validate(self) -> None:
        if self.apodization not in APODIZATIONS:
            raise ValueError(f"apodization must be one of {APODIZATIONS}, got {self.apodization!r}")
        if self.tx_mode not in TX_MODES:
            raise ValueError(f"tx_mode must be one of {TX_MODES}, got {self.tx_mode!r}")
        if self.f_number is not None and self.f_number < 0:
            raise ValueError("f_number must be >= 0 (or None)")
        if self.line_spacing_m is not None and self.line_spacing_m <= 0:
            raise ValueError("line_spacing_m must be > 0")
        if self.n_lines is not None and self.n_lines < 1:
            raise ValueError("n_lines must be >= 1")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be > 0")
