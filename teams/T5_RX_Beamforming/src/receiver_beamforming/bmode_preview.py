"""
bmode_preview.py - quick envelope + log compression so the output of the beamformer can be LOOKED at
===================================================================================================
This is a *preview* for the integration check only.  Final B-mode image formation (envelope detection
choices, time-gain compensation, speckle handling, scan conversion, post-processing) belongs to the
B-mode / Image-Reconstruction teams, who receive ``beamformed_rf`` unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def envelope(beamformed_rf: np.ndarray) -> np.ndarray:
    """|analytic signal| along depth (axis 0)."""
    return np.abs(signal.hilbert(np.asarray(beamformed_rf, dtype=np.float64), axis=0))


def log_compress(env: np.ndarray, dynamic_range_db: float = 50.0, ref: float | None = None) -> np.ndarray:
    """20 log10(env/ref) clipped to [-dynamic_range_db, 0].  ref defaults to the global maximum."""
    ref = float(np.max(env)) if ref is None else float(ref)
    if ref <= 0:
        return np.full(env.shape, -dynamic_range_db)
    db = 20.0 * np.log10(np.maximum(env / ref, 1e-12))
    return np.clip(db, -dynamic_range_db, 0.0)


def to_uint8(bmode_db: np.ndarray, dynamic_range_db: float = 50.0) -> np.ndarray:
    """Map [-DR, 0] dB to 0..255 grayscale."""
    return np.round(255.0 * (bmode_db + dynamic_range_db) / dynamic_range_db).astype(np.uint8)
