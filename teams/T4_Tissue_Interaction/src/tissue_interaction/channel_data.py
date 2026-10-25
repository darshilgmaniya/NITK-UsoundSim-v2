"""Stage 3: reflected pressure per scatterer -> echoes at the receive elements.

This is what the Receiver Beamforming team consumes ("element echoes"):

    channel_data[e, t] = sum_i  G_ie * reflected_i( t - tau_ie )

    tau_ie = r_ie / c_i          travel time scatterer i -> element e, with the mean speed of
                                 sound c_i of that scatterer's path (tissue-aware, so a phantom
                                 whose speed differs from the beamformer's assumed c0 produces
                                 realistic focusing errors)
    G_ie   = (r_ref / r_ie)^p    spreading (p = cfg.spreading_exponent)
           * cos(theta_ie)       element directivity (angle to the element normal)
           * exp(-a_i (r_ie - L_i))   attenuation difference between the real return path and the
                                 straight transmit path that stage 1 already integrated

Fractional delays use 4-tap cubic (Keys) interpolation; use fs >= 10 * f0.
Time zero is the time reference of the incident pressure (start of transmission).
"""
from __future__ import annotations

import numpy as np

from .config import TissueInteractionConfig
from .scatterers import ScattererSet


def _keys(x):
    x = np.abs(x)
    return np.where(x <= 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1,
                    np.where(x < 2, -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, 0.0))


def return_path_terms(scat: ScattererSet, cfg: TissueInteractionConfig, element_xz=None, normals=None):
    """tau (n_scat, n_el) [s], G (n_scat, n_el), element_xz (n_el, 2)."""
    if element_xz is None:
        element_xz, normals = cfg.probe.element_positions()
    ex, ez = element_xz[:, 0], element_xz[:, 1]
    px, pz = scat.xyz[:, 0][:, None], scat.xyz[:, 2][:, None]
    dx, dz = px - ex[None], pz - ez[None]
    r = np.maximum(np.hypot(dx, dz), 1e-6)
    cbar = np.where(scat.tof > 0, scat.path_len / np.maximum(scat.tof, 1e-12), cfg.c0)
    tau = r / cbar[:, None]
    G = (cfg.spreading_ref_m / np.maximum(r, cfg.spreading_ref_m)) ** cfg.spreading_exponent
    if cfg.element_directivity == "cosine":
        cos = (dx * normals[None, :, 0] + dz * normals[None, :, 1]) / r
        G = G * np.clip(cos, 0.0, 1.0)
    if cfg.path_model == "full":
        a_bar = scat.path_atten_np / (2.0 * np.maximum(scat.path_len, 1e-9))
        G = G * np.exp(-a_bar[:, None] * (r - scat.path_len[:, None]))
    return tau, G, element_xz


def compute_channel_data(scat: ScattererSet, reflected: np.ndarray, t_axis: np.ndarray,
                         cfg: TissueInteractionConfig, element_xz=None, normals=None,
                         chunk: int = 512, rel_threshold: float = 1e-4):
    """Returns (rf[n_el, n_out] float32, t_out[n_out], element_xz[n_el, 2])."""
    if element_xz is not None and normals is None:
        normals = np.tile([0.0, 1.0], (len(element_xz), 1))
    tau, G, element_xz = return_path_terms(scat, cfg, element_xz, normals)
    dt = float(t_axis[1] - t_axis[0])
    fs = 1.0 / dt
    n_samp = reflected.shape[1]
    n_el = element_xz.shape[0]
    n0max = int(np.ceil(tau.max() * fs))
    n_out = n_samp + n0max + 6
    out = np.zeros((n_el, n_out), np.float64)
    peak = np.abs(reflected).max(axis=1)
    active = np.where(peak > 1e-9 * max(peak.max(), 1e-30))[0]
    for s in range(0, active.size, chunk):
        idx = active[s:s + chunk]
        Y = reflected[idx].astype(np.float64)
        above = np.abs(Y) > (rel_threshold * peak[idx])[:, None]
        first = above.argmax(1)
        last = n_samp - 1 - above[:, ::-1].argmax(1)
        W = int((last - first).max()) + 1
        cols = first[:, None] + np.arange(W)[None, :]
        valid = (cols <= last[:, None]) & (cols < n_samp)
        seg = np.where(valid, np.take_along_axis(Y, np.clip(cols, 0, n_samp - 1), 1), 0.0)
        for e in range(n_el):
            pos = tau[idx, e] * fs
            n0 = np.floor(pos).astype(np.int64)
            fr = pos - n0
            wts = seg * G[idx, e][:, None]
            base = cols + n0[:, None]
            for off in (-1, 0, 1, 2):
                w = _keys(off - fr)[:, None]                # h(j - pos) with j = n0 + k + off
                j = np.clip(base + off, 0, n_out - 1)
                out[e] += np.bincount(j.ravel(), weights=(wts * w).ravel(), minlength=n_out)[:n_out]
    t_out = float(t_axis[0]) + np.arange(n_out) * dt
    return out.astype(np.float32), t_out, element_xz
