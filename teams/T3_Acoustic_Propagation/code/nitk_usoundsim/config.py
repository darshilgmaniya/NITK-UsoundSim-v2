"""
config.py
=========

Global configuration for NITK-UsoundSim.

This is the SINGLE SOURCE OF TRUTH for medium, transducer, imaging-FOV,
phantom and display parameters. Every other module (acoustic propagation,
tissue interaction, beamforming, image formation, ...) MUST import these
values from here rather than redefining or hard-coding them.
"""

import numpy as np

# ----------------------------------------------------------------------
# Medium parameters
# ----------------------------------------------------------------------
C = 1540.0          # speed of sound in tissue [m/s]
ALPHA_0 = 0.5        # attenuation coefficient [dB / MHz / cm]

# ----------------------------------------------------------------------
# Transducer array
# ----------------------------------------------------------------------
F0 = 5.0e6            # transducer centre frequency [Hz]
FS = 40.0e6            # sampling frequency [Hz]
N_ELEMENTS = 128       # number of array elements (changed 64 -> 128 for the integrated pipeline)
LAMBDA = C / F0         # wavelength in tissue at F0 [m]
PITCH = LAMBDA / 2       # element pitch (lambda/2 spacing) [m]

# ----------------------------------------------------------------------
# Imaging field of view
# ----------------------------------------------------------------------
X_MIN, X_MAX = -0.010, 0.010   # lateral FOV [m]
Z_MIN, Z_MAX = 0.000, 0.040     # axial (depth) FOV [m]
N_SCANLINES = 64

# ----------------------------------------------------------------------
# Calibration targets (synthetic point phantom)
# ----------------------------------------------------------------------
PHANTOM_POINTS = [
    {"x": 0.000, "z": 0.010, "amp": 1.0},
    {"x": 0.000, "z": 0.020, "amp": 1.0},
    {"x": 0.000, "z": 0.030, "amp": 1.0},
]

DYNAMIC_RANGE_DB = 50.0
OUTPUT_SIZE = (512, 512)
