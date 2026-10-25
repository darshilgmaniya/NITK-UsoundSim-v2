"""
bmode_formation.py
==================

T6 B-mode formation for the NITK-UsoundSim integration: envelope detection +
log compression of beamformed RF.

    bmode_formation(beamformed_rf, dynamic_range_db=None) -> bmode

Inputs
------
beamformed_rf    : (n_depths, n_lines) beamformed RF, depth along axis 0 --
                   the first output of T5's das_beamform(). Any real 2-D array
                   in that layout works (e.g. T6's beamformed carotid RF).
dynamic_range_db : displayed dynamic range DR [dB], > 0. Defaults to
                   DEFAULT_DYNAMIC_RANGE_DB (50 dB).

Outputs
-------
bmode : (n_depths, n_lines) float array in [0, 1], same shape as the input.
        1 = brightest pixel of the frame (0 dB), 0 = DR dB below it or less.
        Depth/lateral axes are unchanged: use das_beamform()'s z_axis and
        the scan-line positions.

Equations / algorithm
---------------------
1. Envelope, per scan line along depth (Hilbert transform, analytic signal):
       E(z, x) = | s(z, x) + j * H{s}(z, x) |       (scipy.signal.hilbert, axis=0)
2. Normalise to the frame peak and log-compress:
       L(z, x) = 20 * log10( max(E / max(E), 1e-12) )    [dB, <= 0]
3. Clip to the dynamic range and map to brightness:
       B(z, x) = ( clip(L, -DR, 0) + DR ) / DR           in [0, 1]
An all-zero input returns all zeros (no peak to normalise to).

Why the TIME-AXIS RULE (README Section 6) does not apply here
--------------------------------------------------------------
The rule governs delay-to-sample lookups: any stage that turns a delay tau
into an RF sample must use np.interp(tau, t_axis, rf) on the real t_axis
(which starts at -1.175 us), never tau * FS as an index. This module does no
such lookup. Its input is already beamformed and already on das_beamform()'s
depth axis -- sample i of every line IS depth z_axis[i], because
das_beamform() did the np.interp lookups on the real t_axis. The Hilbert
transform and log compression are sample-wise / along-line operations that
never convert a time into an index, so there is no time origin here to get
wrong. Downstream stages must keep using das_beamform()'s z_axis for depth
labels (not i * dz from 0 unless z_axis starts at 0, which it does: Z_MIN = 0).

Assumptions / limitations
-------------------------
- Normalisation is to the peak of THIS frame, so brightness is relative, not
  absolute: frames with different peaks are not comparable level-for-level.
- The Hilbert transform is applied along the whole line with no windowing;
  expect small edge effects in the first/last few samples of each line.
- No band-pass filtering, TGC (time-gain compensation) or attenuation
  correction -- T3's engine models attenuation, so deep echoes are weaker.
- The 1e-12 floor (-240 dB) only avoids log10(0); it is far below any DR.
- No scan conversion or despeckling -- those are T7's scan_convert() and
  postprocess().
"""

import numpy as np
from scipy.signal import hilbert

DEFAULT_DYNAMIC_RANGE_DB = 50.0   # was config.DYNAMIC_RANGE_DB in the first project


def bmode_formation(beamformed_rf, dynamic_range_db=None):
    """Envelope detection + log compression; see module docstring."""
    dr = DEFAULT_DYNAMIC_RANGE_DB if dynamic_range_db is None else float(dynamic_range_db)
    if dr <= 0:
        raise ValueError("dynamic_range_db must be positive")
    rf = np.asarray(beamformed_rf, dtype=float)
    if rf.ndim != 2:
        raise ValueError("beamformed_rf must be 2-D (n_depths, n_lines)")
    envelope = np.abs(hilbert(rf, axis=0))
    peak = envelope.max()
    if peak == 0:
        return np.zeros_like(envelope)
    db = 20 * np.log10(np.maximum(envelope / peak, 1e-12))
    return (np.clip(db, -dr, 0) + dr) / dr
