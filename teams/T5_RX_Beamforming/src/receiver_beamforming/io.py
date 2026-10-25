"""
io.py - reading the hand-off from the relay (or any channel-data file) and writing the beamformer output
=======================================================================================================
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class ReceiverInput:
    """Everything the beamformer needs, in one validated object."""
    rx: np.ndarray                      # (n_elements, n_samples) pressure per receive element
    t_axis: np.ndarray                  # (n_samples,) seconds, strictly increasing, uniform
    element_x: np.ndarray               # (n_elements,) metres
    fs_hz: float
    c0: float = 1540.0
    f0_hz: float = 5.0e6
    pitch_m: Optional[float] = None
    scatterer_xyz: Optional[np.ndarray] = None   # (N,3) truth, validation / auto-FOV only
    source: str = "memory"

    @property
    def n_elements(self) -> int:
        return self.rx.shape[0]


def _scalar(d, key, default=None):
    if key in d:
        return float(np.asarray(d[key]).reshape(-1)[0])
    return default


def make_receiver_input(rx, t_axis, element_x, *, fs_hz=None, c0=1540.0, f0_hz=5.0e6, pitch_m=None,
                        scatterer_xyz=None, source="memory") -> ReceiverInput:
    """Validate arrays and build a ReceiverInput.  Fixes a transposed (T, n_el) layout automatically."""
    rx = np.asarray(rx, dtype=np.float64)
    t = np.asarray(t_axis, dtype=np.float64).ravel()
    ex = np.asarray(element_x, dtype=np.float64).ravel()
    if rx.ndim != 2:
        raise ValueError(f"received_pressure must be 2-D (n_elements, n_samples); got shape {rx.shape}")
    if rx.shape[0] != ex.size and rx.shape[1] == ex.size:
        warnings.warn("received_pressure looks transposed (n_samples, n_elements); transposing it.")
        rx = rx.T
    if rx.shape[0] != ex.size:
        raise ValueError(f"{rx.shape[0]} channels in received_pressure but {ex.size} element positions")
    if rx.shape[1] != t.size:
        raise ValueError(f"{rx.shape[1]} samples in received_pressure but t_axis has {t.size}")
    if t.size < 8:
        raise ValueError("record is too short (fewer than 8 samples)")
    if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(t)) and np.all(np.isfinite(ex))):
        raise ValueError("received_pressure / t_axis / element_x contain NaN or Inf")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t_axis must be strictly increasing")
    if np.ptp(dt) > 1e-3 * dt.mean():
        raise ValueError("t_axis must be uniformly sampled")
    fs = float(fs_hz) if fs_hz else 1.0 / float(dt.mean())
    if abs(fs * dt.mean() - 1.0) > 1e-3:
        raise ValueError(f"sampling_rate_hz ({fs:.4g}) disagrees with t_axis spacing ({1/dt.mean():.4g} Hz)")
    if np.any(np.diff(ex) <= 0):                       # sort elements by position, keep channel pairing
        order = np.argsort(ex)
        ex, rx = ex[order], rx[order]
    if pitch_m is None and ex.size > 1:
        pitch_m = float(np.median(np.diff(ex)))
    sc = None if scatterer_xyz is None else np.asarray(scatterer_xyz, dtype=float).reshape(-1, 3)
    return ReceiverInput(rx, t, ex, fs, float(c0), float(f0_hz), pitch_m, sc, source)


def load_receiver_input(src: Union[str, os.PathLike, dict, "ReceiverInput"]) -> ReceiverInput:
    """Load ``receiver_bf_input.npz`` (or the full ``04_part2_...npz``, or a dict with the same keys)."""
    if isinstance(src, ReceiverInput):
        return src
    if isinstance(src, (str, os.PathLike)):
        if not os.path.exists(src):
            raise FileNotFoundError(f"hand-off file not found: {src}")
        d = np.load(src, allow_pickle=False)
        name = str(src)
    elif isinstance(src, dict):
        d, name = src, "dict"
    else:
        raise TypeError("src must be a path, a dict with the hand-off keys, or a ReceiverInput")
    missing = [k for k in ("received_pressure", "t_axis", "element_x") if k not in d]
    if missing:
        raise KeyError(f"hand-off is missing key(s) {missing}; see docs/06_DATA_DICTIONARY.md")
    return make_receiver_input(
        d["received_pressure"], d["t_axis"], d["element_x"],
        fs_hz=_scalar(d, "sampling_rate_hz"), c0=_scalar(d, "c0_nominal_m_s", 1540.0),
        f0_hz=_scalar(d, "f0_hz", 5.0e6), pitch_m=_scalar(d, "pitch_m"),
        scatterer_xyz=d["scatterer_xyz"] if "scatterer_xyz" in d else None, source=name)


def save_beamformed(path: str, result, extra_description: str = "") -> str:
    """Write the hand-off file for the next team (B-mode image formation)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        description=np.array("RECEIVER BEAMFORMING OUTPUT - hand-off to B-mode image formation. "
                             "beamformed_rf is signed RF (depth, line); envelope/bmode_db are a PREVIEW only. "
                             + extra_description),
        beamformed_rf=result.beamformed_rf.astype(np.float32),
        beamformed_rf_shape_meaning=np.array("(n_depth, n_lines)"),
        z_axis=result.z_axis, z_axis_units=np.array("m"),
        x_lines=result.x_lines, x_lines_units=np.array("m"),
        envelope_preview=result.envelope.astype(np.float32),
        bmode_db_preview=result.bmode_db.astype(np.float32),
        dynamic_range_db=np.array(result.cfg.dynamic_range_db),
        sampling_rate_hz=np.array(result.input.fs_hz), c0_nominal_m_s=np.array(result.input.c0),
        f0_hz=np.array(result.input.f0_hz),
        apodization=np.array(result.cfg.apodization),
        f_number=np.array(-1.0 if not result.cfg.f_number else result.cfg.f_number),
        n_elements=np.array(result.input.n_elements))
    return path
