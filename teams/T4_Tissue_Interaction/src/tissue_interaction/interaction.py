"""Stage 2: incident pressure at each scatterer -> reflected / scattered pressure.

    reflected_i(t) = amp_i * Tx_i * IFFT{ FFT{incident_i}(f) * H_i(f) }

    H_i(f) = (f/f0)^m_i                       frequency dependence of scattering
             * exp(-L_i * (f/f0)^y_i)         two-way attenuation, frequency dependent
                                              (this is what down-shifts the spectrum with depth)

``amp_i`` is signed: a negative interface reflectivity (soft tissue -> gas) flips
the polarity of the echo, exactly as in a real RF line.
The output has the shape the Acoustic Propagation module already expects:
``reflected_pressure_input`` (n_scatterers, n_samples).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import TissueInteractionConfig
from .scatterers import ScattererSet


def compute_reflected_pressure(scat: ScattererSet, incident: np.ndarray, t_axis: np.ndarray,
                               cfg: TissueInteractionConfig, chunk: int = 2048,
                               out: Optional[np.ndarray] = None) -> np.ndarray:
    """incident: (n_scatterers, n_samples) pressure arriving at each scatterer
    (same row order as ``scat``).  Returns float32 (n_scatterers, n_samples).
    Pass ``out`` (e.g. a np.lib.format.open_memmap) to keep RAM low."""
    incident = np.asarray(incident)
    if incident.ndim != 2 or incident.shape[0] != scat.n:
        raise ValueError(f"incident must have shape ({scat.n}, n_samples); got {incident.shape}. "
                         "Row i must be the pressure at scatterer i of the scatterer file.")
    n_samp = incident.shape[1]
    if len(t_axis) != n_samp:
        raise ValueError("t_axis length must equal the number of time samples")
    dt = float(t_axis[1] - t_axis[0])
    nfft = 1 << int(np.ceil(np.log2(2 * n_samp)))
    f = np.fft.rfftfreq(nfft, dt)
    fr = f / cfg.f0_hz
    full = cfg.path_model == "full"
    if out is None:
        out = np.empty((scat.n, n_samp), np.float32)
    for s in range(0, scat.n, chunk):
        sl = slice(s, min(s + chunk, scat.n))
        X = np.fft.rfft(incident[sl].astype(np.float64), n=nfft, axis=1)
        m = scat.scatter_exp[sl].astype(np.float64)[:, None]
        H = np.power(fr[None, :], m)                      # 0**0 = 1, 0**m = 0
        gain = scat.amp[sl].astype(np.float64)
        if full:
            L = scat.path_atten_np[sl].astype(np.float64)[:, None]
            y = scat.path_y_eff[sl].astype(np.float64)[:, None]
            H = H * np.exp(-L * np.power(fr[None, :], y))
            gain = gain * scat.path_tx_loss[sl]
        y_t = np.fft.irfft(X * H, n=nfft, axis=1)[:, :n_samp]
        out[sl] = (y_t * gain[:, None]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
def gaussian_pulse(t, t0, f0, fractional_bandwidth):
    """Gaussian-modulated sinusoid (-6 dB fractional bandwidth)."""
    sigma_f = fractional_bandwidth * f0 / (2 * np.sqrt(2 * np.log(2)))
    sigma_t = 1.0 / (2 * np.pi * sigma_f)
    return np.exp(-0.5 * ((t - t0) / sigma_t) ** 2) * np.cos(2 * np.pi * f0 * (t - t0))


def make_mock_incident(scat: ScattererSet, cfg: TissueInteractionConfig, p0: float = 1.0,
                       n_samples: Optional[int] = None):
    """Stand-in for Acoustic Propagation's forward_propagation(): a unit pulse that
    arrives at every scatterer after its tissue-aware time of flight (plane-wave-like
    insonification with the configured spreading law).  Attenuation / shadowing are NOT in it
    (they are applied by path_model='full').  Returns (incident, t_axis)."""
    dt = 1.0 / cfg.fs_hz
    sigma_f = cfg.fractional_bandwidth * cfg.f0_hz / (2 * np.sqrt(2 * np.log(2)))
    margin = 4.0 / (2 * np.pi * sigma_f)
    if n_samples is None:
        n_samples = int(np.ceil((scat.tof.max() + margin) / dt)) + 8
    t = np.arange(n_samples) * dt
    inc = np.empty((scat.n, n_samples), np.float32)
    ref = cfg.spreading_ref_m
    spread = (ref / np.maximum(scat.path_len, ref)) ** cfg.spreading_exponent
    for s in range(0, scat.n, 2048):
        sl = slice(s, min(s + 2048, scat.n))
        inc[sl] = (p0 * spread[sl, None] *
                   gaussian_pulse(t[None, :], scat.tof[sl, None], cfg.f0_hz, cfg.fractional_bandwidth)
                   ).astype(np.float32)
    return inc, t


def estimate_memory_mb(n_scatterers: int, n_samples: int) -> float:
    """Size of ONE (n_scatterers x n_samples) float32 array in MB."""
    return n_scatterers * n_samples * 4 / 1e6
