"""
image_reconstruction.py - T6 Image Reconstruction (runner 6)
============================================================
Sits between Receive Beamforming (T5) and B-mode Formation (T7).

    beamformed_rf (depth, lines)  ->  clean RF image-line data (depth, lines)  [+ envelope for inspection]

Steps (from the team's sample notebook, ``UsoundIMGRECsample``):
    1. DC-offset removal   : subtract the mean of every scan line (along depth).
    2. Band-pass filtering : zero-phase Butterworth around the centre frequency f0
                             (default band f0*(1 - bw/2) .. f0*(1 + bw/2); bw = 1.2 -> 2..8 MHz for f0 = 5 MHz).
    3. Envelope            : |Hilbert| of the filtered RF along depth (returned for plotting; T7 recomputes it).
    4. Depth axis          : z = c t / 2 is already fixed by T5, so z_axis is passed through unchanged.

The sampling rate along depth is fs_depth = c / (2 dz), dz = z[1] - z[0].
Scan conversion is *not* done here: it stays in T8 together with despeckling (it needs the final display grid).

Assumptions: the depth axis is uniform; RF is oscillating around zero at about f0.
Limitations: a linear-array (rectilinear) geometry only; no time-gain compensation (the simulation already has none).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


@dataclass
class ReconConfig:
    enabled: bool = True
    remove_dc: bool = True
    bandpass: bool = True
    bandwidth_frac: float = 1.2      # total band = f0 * bandwidth_frac, centred on f0
    order: int = 2                   # Butterworth order (applied forward + backward = zero phase)

    def validate(self) -> None:
        if not 0 < self.bandwidth_frac < 2:
            raise ValueError("bandwidth_frac must be in (0, 2)")
        if self.order < 1:
            raise ValueError("order must be >= 1")


@dataclass
class ReconResult:
    rf: np.ndarray                   # (depth, lines) conditioned RF
    envelope: np.ndarray             # (depth, lines) |Hilbert(rf)|
    z_axis: np.ndarray               # (depth,) [m]
    fs_depth_hz: float
    band_hz: Optional[tuple]
    notes: list = field(default_factory=list)


def image_reconstruction(beamformed_rf, z_axis, c: float = 1540.0, f0_hz: float = 5.0e6,
                         config: Optional[ReconConfig] = None) -> ReconResult:
    cfg = config or ReconConfig()
    cfg.validate()
    rf = np.asarray(beamformed_rf, dtype=float)
    z = np.asarray(z_axis, dtype=float)
    if rf.ndim != 2:
        raise ValueError("beamformed_rf must be 2-D (n_depths, n_lines)")
    if z.size != rf.shape[0]:
        raise ValueError(f"z_axis has {z.size} samples but beamformed_rf has {rf.shape[0]} depth rows")
    dz = float(np.mean(np.diff(z)))
    if dz <= 0:
        raise ValueError("z_axis must be increasing")
    fs = c / (2 * dz)
    notes, band = [], None
    out = rf.copy()
    if cfg.enabled and cfg.remove_dc:
        out = out - out.mean(axis=0, keepdims=True)
        notes.append("DC removed per line")
    if cfg.enabled and cfg.bandpass:
        nyq = fs / 2
        lo = max(f0_hz * (1 - cfg.bandwidth_frac / 2), 0.02 * nyq)
        hi = min(f0_hz * (1 + cfg.bandwidth_frac / 2), 0.95 * nyq)
        if lo < hi and out.shape[0] > 3 * (2 * cfg.order + 1) * 3:
            b, a = butter(cfg.order, [lo / nyq, hi / nyq], btype="bandpass")
            out = filtfilt(b, a, out, axis=0)
            band = (lo, hi)
            notes.append(f"band-pass {lo/1e6:.2f}-{hi/1e6:.2f} MHz (zero phase)")
        else:
            notes.append("band-pass skipped (band or signal too short)")
    env = np.abs(hilbert(out, axis=0))
    return ReconResult(out, env, z, fs, band, notes)
