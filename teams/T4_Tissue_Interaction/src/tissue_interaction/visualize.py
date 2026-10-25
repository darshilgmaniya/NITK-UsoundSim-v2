"""2-D grayscale maps (the deliverable images) + a QA B-mode preview."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage, signal

from .config import TissueInteractionConfig
from .interaction import gaussian_pulse
from .phantom import Phantom
from .scatterers import ScattererSet, build_scatterers


def _norm(a, lo=None, hi=None):
    a = np.asarray(a, np.float64)
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)


def label_map_gray(ph: Phantom) -> np.ndarray:
    """Label map as grey levels ordered by acoustic impedance (dark = low Z)."""
    z = ph.lut("impedance")
    rank = np.argsort(np.argsort(z))
    g = rank / max(len(z) - 1, 1)
    return g[ph.index]


def reflectivity_map(ph: Phantom) -> np.ndarray:
    """|R| (normal incidence) at every pixel boundary, log scaled 0..1."""
    Z = ph.property_map("impedance")
    r = np.zeros_like(Z)
    rh = np.abs((Z[:, 1:] - Z[:, :-1]) / (Z[:, 1:] + Z[:, :-1]))
    rv = np.abs((Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1]))
    r[:, :-1] = np.maximum(r[:, :-1], rh)
    r[:-1, :] = np.maximum(r[:-1, :], rv)
    r = ndimage.maximum_filter(r, 2)
    db = 20 * np.log10(np.maximum(r, 1e-4))
    return _norm(db, -60, 0)


def property_panels(ph: Phantom, cfg: TissueInteractionConfig) -> Dict[str, Tuple[np.ndarray, str]]:
    return {
        "tissue_labels": (label_map_gray(ph), "tissue label (grey = impedance rank)"),
        "density": (_norm(ph.property_map("density"), 0, None), "density [kg/m^3]"),
        "speed_of_sound": (_norm(ph.property_map("speed")), "speed of sound [m/s]"),
        "impedance": (_norm(ph.property_map("impedance")), "impedance Z [MRayl]"),
        "attenuation": (_norm(ph.property_map("atten_db_cm", cfg.f0_hz)), "attenuation @ f0 [dB/cm]"),
        "backscatter": (_norm(ph.property_map("backscatter_db"), -60, 10), "backscatter level [dB re liver]"),
        "interface_reflectivity": (reflectivity_map(ph), "|R| at boundaries (log)"),
    }


def _splat(ph: Phantom, x, z, w, step):
    nx = int(np.ceil(ph.width / step)); nz = int(np.ceil(ph.depth / step))
    gx = (x + ph.width / 2) / step - 0.5
    gz = z / step - 0.5
    x0 = np.floor(gx).astype(int); z0 = np.floor(gz).astype(int)
    fx = gx - x0; fz = gz - z0
    img = np.zeros(nz * nx)
    for dz, dx, wt in ((0, 0, (1 - fz) * (1 - fx)), (0, 1, (1 - fz) * fx),
                       (1, 0, fz * (1 - fx)), (1, 1, fz * fx)):
        zi, xi = z0 + dz, x0 + dx
        ok = (zi >= 0) & (zi < nz) & (xi >= 0) & (xi < nx)
        img += np.bincount(zi[ok] * nx + xi[ok], weights=(w * wt)[ok], minlength=nz * nx)
    return img.reshape(nz, nx)


def scatterer_map(scat: ScattererSet, ph: Phantom, cfg: TissueInteractionConfig, dyn_db=50):
    """Where the echoes come from: |effective reflectivity| splatted + smoothed (dB)."""
    step = cfg.wavelength / 4
    a = np.abs(scat.effective_amp(cfg.path_model)) ** 2
    img = _splat(ph, scat.xyz[:, 0], scat.xyz[:, 2], a, step)
    img = ndimage.gaussian_filter(img, 2.0)
    db = 10 * np.log10(np.maximum(img, 1e-30) / max(img.max(), 1e-30))
    return _norm(db, -dyn_db, 0)


def echo_energy_map(scat: ScattererSet, reflected: np.ndarray, ph: Phantom,
                    cfg: TissueInteractionConfig, dyn_db=50):
    """Energy of the *actual module output* (sum_t reflected^2) mapped back to space."""
    e = (reflected.astype(np.float64) ** 2).sum(1)
    step = cfg.wavelength / 3
    img = ndimage.gaussian_filter(_splat(ph, scat.xyz[:, 0], scat.xyz[:, 2], e, step), 2.0)
    db = 10 * np.log10(np.maximum(img, 1e-30) / max(img.max(), 1e-30))
    return _norm(db, -dyn_db, 0)


def qa_bmode_preview(ph: Phantom, cfg: TissueInteractionConfig, dyn_db=60, dense_per_cell=10.0,
                     max_scatterers=400_000, max_pixels=4_000_000):
    """Convolutional B-mode preview (PSF * reflectivity -> envelope -> log).

    ONLY a quality-assurance picture so you can check the module without the other
    six teams' code.  It is NOT the B-mode team's deliverable.  A denser speckle
    realisation is generated for it (cheap: no time-domain waveforms needed)."""
    qcfg = replace(cfg, scatterers_per_cell=dense_per_cell, max_scatterers=max_scatterers)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = build_scatterers(ph, qcfg)
    lam = cfg.wavelength
    step = lam / 8
    while (ph.width / step) * (ph.depth / step) > max_pixels:
        step *= 1.2
    a = sc.effective_amp(cfg.path_model).astype(np.float64)
    img = _splat(ph, sc.xyz[:, 0], sc.xyz[:, 2], a, step)
    # axial pulse-echo kernel (round trip) and lateral beam profile
    sigma_f = cfg.fractional_bandwidth * cfg.f0_hz / (2 * np.sqrt(2 * np.log(2)))
    sigma_z = (1 / (2 * np.pi * sigma_f)) * cfg.c0 / 2
    zk = np.arange(-4 * sigma_z, 4 * sigma_z + step, step)
    kz = gaussian_pulse(2 * zk / cfg.c0, 0.0, cfg.f0_hz, cfg.fractional_bandwidth)
    sx = lam * cfg.f_number / 2.355
    xk = np.arange(-3 * sx, 3 * sx + step, step)
    kx = np.exp(-0.5 * (xk / sx) ** 2); kx /= kx.sum()
    rf = ndimage.convolve1d(ndimage.convolve1d(img, kz, axis=0, mode="constant"), kx, axis=1, mode="constant")
    env = np.abs(signal.hilbert(rf, axis=0))
    env = ndimage.gaussian_filter(env, 0.7)
    db = 20 * np.log10(np.maximum(env, 1e-12) / max(env.max(), 1e-12))
    return _norm(db, -dyn_db, 0)
