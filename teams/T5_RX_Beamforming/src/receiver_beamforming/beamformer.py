"""
beamformer.py - receive delay-and-sum (DAS) beamforming
=======================================================

``das_beamform`` keeps the call signature used by the Receiver Beamforming team's
``liver_echo_simulation.py``::

    beamformed_rf, z_axis = das_beamform(raw_rf, x_elements, x_lines, config,
                                         apply_focus=True, t_axis=t_axis,
                                         apodization="hann", f_number=1.0)

Input   raw_rf      (n_elements, n_samples, n_lines)  one channel record per scan line   (team layout)
                    or (n_elements, n_samples)        ONE record shared by all lines     (relay layout,
                                                      unfocused / plane-wave transmit)
Output  beamformed_rf (n_depth, n_lines)   signed RF, NOT a B-mode image
        z_axis        (n_depth,)           depth [m], dz = c / (2 fs)

Method (per scan line x_l and depth z)::

    t(z)      = t_tx(x_l, z) + sqrt((x_l - x_e)^2 + z^2) / c        time of flight to element e
    s_e       = raw_rf[e](t(z))                                     linear interpolation, 0 outside record
    w_e       = window( (x_e - x_l) / (z / (2 F)) )                 dynamic aperture + apodisation
    out(z,l)  = sum_e w_e s_e / sum_e w_e

``t_tx`` is z/c ("vertical", exact for a plane wave and for a line transmit near the axis) or the
earliest element arrival ("nearest", also valid with per-element transmit delays).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import APODIZATIONS

_EPS = 1e-12


# ---------------------------------------------------------------------------------------------
def apodization_window(name: str, u: np.ndarray) -> np.ndarray:
    """Window value for normalised position u in [-1, 1] (u = 0 is the aperture centre); 0 outside."""
    name = name.lower()
    if name in ("rect", "rectangular", "none", "boxcar"):
        return (np.abs(u) <= 1.0).astype(float)
    if name not in APODIZATIONS:
        raise ValueError(f"unknown apodization {name!r}; choose from {APODIZATIONS}")
    inside = np.abs(u) <= 1.0
    uc = np.clip(u, -1.0, 1.0)
    if name == "hann":
        w = 0.5 * (1.0 + np.cos(np.pi * uc))
    elif name == "hamming":
        w = 0.54 + 0.46 * np.cos(np.pi * uc)
    elif name == "blackman":
        w = 0.42 + 0.5 * np.cos(np.pi * uc) + 0.08 * np.cos(2.0 * np.pi * uc)
    else:  # tukey, alpha = 0.5: flat in the middle half, cosine tapers outside
        a = 0.5
        au = np.abs(uc)
        w = np.where(au <= 1.0 - a, 1.0, 0.5 * (1.0 + np.cos(np.pi * (au - (1.0 - a)) / a)))
    return np.where(inside, w, 0.0)


def _get(cfg, name, default=None):
    return getattr(cfg, name, default)


def resolve_depth_axis(config, t_axis: np.ndarray) -> np.ndarray:
    """Depth axis from the config (team convention); falls back to the record length if absent."""
    c = float(_get(config, "SPEED_OF_SOUND", 1540.0))
    fs = float(_get(config, "SAMPLING_FREQUENCY"))
    getter = _get(config, "get_depth_axis")
    if callable(getter):
        z = np.asarray(getter(), dtype=float)
    else:
        zmax = _get(config, "MAX_DEPTH", None)
        if zmax is None:
            zmax = 0.5 * c * float(t_axis[-1])
        z = np.arange(0.0, float(zmax), c / (2.0 * fs))
    if z.size < 2:
        raise ValueError("depth axis has fewer than 2 samples; check MAX_DEPTH / sampling frequency")
    return z


def _validate(raw_rf, x_elements, x_lines, t_axis):
    raw = np.asarray(raw_rf)
    if raw.ndim not in (2, 3):
        raise ValueError(f"raw_rf must be 2-D (n_el, n_samples) or 3-D (n_el, n_samples, n_lines); got {raw.shape}")
    xe = np.asarray(x_elements, dtype=float).ravel()
    xl = np.atleast_1d(np.asarray(x_lines, dtype=float)).ravel()
    if raw.shape[0] != xe.size:
        raise ValueError(f"raw_rf has {raw.shape[0]} channels but x_elements has {xe.size} entries "
                         f"(expected layout: (n_elements, n_samples[, n_lines]))")
    if raw.ndim == 3 and raw.shape[2] != xl.size:
        raise ValueError(f"raw_rf has {raw.shape[2]} lines but x_lines has {xl.size} entries")
    if t_axis is None:
        raise ValueError("t_axis is required (time of every sample, seconds)")
    t = np.asarray(t_axis, dtype=float).ravel()
    if t.size != raw.shape[1]:
        raise ValueError(f"t_axis has {t.size} samples but raw_rf has {raw.shape[1]}")
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(t)) and np.all(np.isfinite(xe)) and np.all(np.isfinite(xl))):
        raise ValueError("inputs contain NaN or Inf")
    return raw, xe, xl, t


def das_beamform(raw_rf, x_elements, x_lines, config, apply_focus: bool = True,
                 t_axis: Optional[np.ndarray] = None, apodization: str = "hann",
                 f_number: Optional[float] = 1.0, *, tx_mode: Optional[str] = None,
                 tx_delays: Optional[np.ndarray] = None, normalize: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Receive delay-and-sum with dynamic focus and dynamic aperture.  See the module docstring."""
    raw, xe, xl, t = _validate(raw_rf, x_elements, x_lines, t_axis)
    n_el, n_s = raw.shape[0], raw.shape[1]
    shared = raw.ndim == 2
    c = float(_get(config, "SPEED_OF_SOUND", 1540.0))
    fs = float(_get(config, "SAMPLING_FREQUENCY", 1.0 / (t[1] - t[0]) if t.size > 1 else 1.0))
    if t.size > 1:
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("t_axis must be strictly increasing")
        if abs(dt.mean() * fs - 1.0) > 1e-3:
            raise ValueError(f"t_axis spacing ({dt.mean():.3e} s) disagrees with SAMPLING_FREQUENCY ({fs:.3e} Hz)")
    if str(apodization).lower() not in tuple(APODIZATIONS) + ("rectangular", "none", "boxcar"):
        raise ValueError(f"unknown apodization {apodization!r}; choose from {APODIZATIONS}")
    z = resolve_depth_axis(config, t)
    mode = tx_mode or _get(config, "TX_MODE", "vertical")
    if mode not in ("vertical", "nearest"):
        raise ValueError("tx_mode must be 'vertical' or 'nearest'")
    tau = tx_delays if tx_delays is not None else _get(config, "TX_DELAYS")
    tau = np.zeros(n_el) if tau is None else np.asarray(tau, dtype=float).ravel()
    if tau.size != n_el:
        raise ValueError(f"tx_delays has {tau.size} entries, expected {n_el}")
    if tx_delays is not None and tx_mode is None:
        mode = "nearest"                                   # delays only make sense with this model
    ap_half_full = float(np.max(np.abs(xe - xe.mean()))) + 1e-9

    out = np.zeros((z.size, xl.size), dtype=np.float64)
    rows = np.arange(n_el)[None, :]                         # (1, n_el)
    z_col = z[:, None]                                      # (nz, 1)
    t0 = t[0]

    for li, x_line in enumerate(xl):
        lat = xe[None, :] - x_line                          # (1, n_el)
        r = np.hypot(lat, z_col) if apply_focus else np.broadcast_to(z_col, (z.size, n_el))
        if mode == "vertical":
            t_tx = z_col / c
        else:
            t_tx = (tau[None, :] + np.hypot(xe[None, :] - x_line, z_col) / c).min(axis=1, keepdims=True)
        tof = t_tx + r / c                                  # (nz, n_el)

        # dynamic aperture: half-width z / (2F); F = None/0 -> every element, static half-width
        if f_number:
            half = np.maximum(z_col / (2.0 * float(f_number)), 0.5 * np.min(np.diff(np.sort(xe))) if n_el > 1 else 1e-6)
        else:
            half = np.full_like(z_col, max(float(np.max(np.abs(xe - x_line))), ap_half_full) + 1e-9)
        w = apodization_window(apodization, lat / half)     # (nz, n_el)

        pos = (tof - t0) * fs
        i0 = np.floor(pos).astype(np.int64)
        fr = pos - i0
        ok = (i0 >= 0) & (i0 < n_s - 1)
        i0c = np.clip(i0, 0, n_s - 2)
        rec = raw if shared else raw[:, :, li]
        s = rec[rows, i0c] * (1.0 - fr) + rec[rows, i0c + 1] * fr
        w = w * ok
        num = (w * s).sum(axis=1)
        if normalize:
            den = w.sum(axis=1)
            out[:, li] = np.where(den > _EPS, num / np.maximum(den, _EPS), 0.0)
        else:
            out[:, li] = num
    return out, z
