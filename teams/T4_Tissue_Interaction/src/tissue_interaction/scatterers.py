"""Phantom -> scatterer list (Stage 1 of the Tissue Interaction module).

Two kinds of scatterer are produced, exactly the two mechanisms of
"reflection / scattering" in the pipeline table:

* **interface** scatterers (kind=1): specular reflection at every boundary
  between two tissues.  Reflectivity uses the oblique-incidence Fresnel
  coefficient (Snell's law, total reflection beyond the critical angle) and a
  specular directivity term, so boundaries are bright when the beam hits them
  head-on and fade when it grazes them.
* **diffuse** scatterers (kind=0): sub-wavelength random scatterers inside each
  tissue.  Their strength comes from the tissue backscatter level, they create
  the speckle texture.

For every scatterer the straight path from the transducer is integrated through
the phantom to give two-way attenuation, attenuation exponent, interface
transmission loss (shadowing behind bone / air / lung), and one-way time of
flight.  These are applied later in :mod:`interaction` (path_model='full').
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage

from .config import TissueInteractionConfig
from .phantom import Phantom

KIND_DIFFUSE, KIND_INTERFACE = 0, 1


@dataclass
class ScattererSet:
    xyz: np.ndarray                  # (n,3) metres: x lateral, y = 0 (2-D), z depth
    amp: np.ndarray                  # (n,) signed intrinsic reflectivity (pressure ratio)
    kind: np.ndarray                 # (n,) 0 diffuse / 1 interface
    tissue_idx: np.ndarray           # (n,) tissue at (interface: behind) the scatterer
    tissue_idx_pre: np.ndarray       # (n,) tissue in front of the scatterer
    scatter_exp: np.ndarray          # (n,) backscatter ~ (f/f0)^m
    path_atten_np: np.ndarray        # (n,) two-way attenuation at f0 [Np]
    path_y_eff: np.ndarray           # (n,) effective attenuation exponent along the path
    path_tx_loss: np.ndarray         # (n,) two-way interface transmission factor (0..1)
    tof: np.ndarray                  # (n,) one-way time of flight [s]
    path_len: np.ndarray             # (n,) distance origin -> scatterer [m]
    incid_angle_deg: np.ndarray      # (n,) incidence angle on interface (nan for diffuse)
    tissue_names: List[str]
    f0_hz: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.xyz.shape[0]

    def row_labels(self) -> np.ndarray:
        if self.meta.get("row_labels") is not None:           # e.g. labels supplied by Acoustic Propagation
            return np.asarray(self.meta["row_labels"])
        out = []
        for i in range(self.n):
            if self.kind[i] == KIND_INTERFACE:
                lab = f"s{i:05d}:I:{self.tissue_names[self.tissue_idx_pre[i]]}>{self.tissue_names[self.tissue_idx[i]]}"
            else:
                lab = f"s{i:05d}:D:{self.tissue_names[self.tissue_idx[i]]}"
            out.append(lab)
        return np.array(out)

    def effective_amp(self, path_model: str = "full") -> np.ndarray:
        """Amplitude at f0 after path effects (used for QA maps)."""
        if path_model == "none":
            return self.amp
        return self.amp * self.path_tx_loss * np.exp(-self.path_atten_np)

    def summary(self) -> str:
        ni = int((self.kind == KIND_INTERFACE).sum())
        return f"{self.n} scatterers ({ni} interface, {self.n - ni} diffuse)"


# -----------------------------------------------------------------------------
def _interface_points(ph: Phantom, cfg, lam):
    idx = ph.index
    xc, zc = ph.x_coords(), ph.z_coords()
    jz, jx = np.nonzero(idx[:, :-1] != idx[:, 1:])
    Xh, Zh = xc[jx] + ph.dx / 2, zc[jz]
    Ah, Bh = idx[jz, jx], idx[jz, jx + 1]
    jz2, jx2 = np.nonzero(idx[:-1, :] != idx[1:, :])
    Xv, Zv = xc[jx2], zc[jz2] + ph.dz / 2
    Av, Bv = idx[jz2, jx2], idx[jz2 + 1, jx2]
    X = np.concatenate([Xh, Xv]); Z = np.concatenate([Zh, Zv])
    A = np.concatenate([Ah, Av]).astype(np.int64); B = np.concatenate([Bh, Bv]).astype(np.int64)
    if X.size == 0:
        return np.zeros(0), np.zeros(0)
    cs = lam * cfg.interface_spacing_over_lambda
    K = len(ph.tissue_list)
    ci = np.floor((X + ph.width / 2) / cs).astype(np.int64)
    cj = np.floor(Z / cs).astype(np.int64)
    ncj = int(np.ceil(ph.depth / cs)) + 2
    key = ((ci * ncj + cj) * K + np.minimum(A, B)) * K + np.maximum(A, B)
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    return np.bincount(inv, X) / cnt, np.bincount(inv, Z) / cnt


def _interface_scatterers(ph: Phantom, cfg, lam):
    X, Z = _interface_points(ph, cfg, lam)
    if X.size == 0:
        return None
    probe = cfg.probe
    _, d = probe.origins_and_dirs(X, Z)
    eps = 1.5 * max(ph.dx, ph.dz)
    pre = ph.label_at(X - eps * d[:, 0], Z - eps * d[:, 1])
    post = ph.label_at(X + eps * d[:, 0], Z + eps * d[:, 1])
    keep = pre != post
    X, Z, d, pre, post = X[keep], Z[keep], d[keep], pre[keep], post[keep]

    lutZ, lutC = ph.lut("impedance"), ph.lut("speed")
    Zm = ph.property_map("impedance")
    Ls = ndimage.gaussian_filter(np.log(Zm), 1.5)
    gz, gx = np.gradient(Ls, ph.dz, ph.dx)
    iz, ix = ph.pixel_indices(X, Z)
    iz, ix = np.clip(iz, 0, ph.nz - 1), np.clip(ix, 0, ph.nx - 1)
    g = np.stack([gx[iz, ix], gz[iz, ix]], 1)
    gn = np.linalg.norm(g, axis=1)
    nvec = g / np.maximum(gn, 1e-30)[:, None]
    cos1 = np.where(gn > 1e-9, np.abs((nvec * d).sum(1)), 1.0)
    cos1 = np.clip(cos1, 0.0, 1.0)
    sin1 = np.sqrt(1 - cos1 ** 2)
    Z1, Z2, c1, c2 = lutZ[pre], lutZ[post], lutC[pre], lutC[post]
    sin2 = c2 / c1 * sin1
    tir = sin2 >= 1.0
    cos2 = np.sqrt(np.clip(1 - sin2 ** 2, 0, 1))
    R = (Z2 * cos1 - Z1 * cos2) / (Z2 * cos1 + Z1 * cos2)
    R = np.where(tir, np.sign(Z2 - Z1), R)
    th = np.arccos(cos1)
    th0 = np.deg2rad(cfg.specular_theta0_deg)
    D = cfg.specular_floor + (1 - cfg.specular_floor) * np.exp(-(th / th0) ** 2)
    cs = lam * cfg.interface_spacing_over_lambda
    # n points inside one beam width (lam*F#) add coherently to ~ R*D
    amp = R * D * cs / (lam * cfg.f_number) * cfg.interface_gain
    ok = np.abs(amp) > 1e-7
    return dict(X=X[ok], Z=Z[ok], amp=amp[ok], pre=pre[ok], post=post[ok],
                theta=np.rad2deg(th[ok]))


def _smooth_noise(ph: Phantom, cfg, rng):
    cell = 0.5e-3
    nzg = int(np.ceil(ph.depth / cell)) + 3
    nxg = int(np.ceil(ph.width / cell)) + 3
    n = ndimage.gaussian_filter(rng.standard_normal((nzg, nxg)), cfg.texture_corr_mm * 1e-3 / cell)
    n /= max(n.std(), 1e-12)
    return n, cell


def _diffuse_scatterers(ph: Phantom, cfg, lam, rng, n_budget):
    area_mm2 = ph.width * ph.depth * 1e6
    cell_mm2 = (lam * cfg.pulse_cycles / 2) * (lam * cfg.f_number) * 1e6
    density = cfg.density_per_mm2 or cfg.scatterers_per_cell / cell_mm2
    db = ph.lut("backscatter_db")
    echo = db > cfg.prune_db
    frac = float(np.mean(echo[ph.index]))
    if frac <= 0:
        return None
    dens_eff = min(density, max(n_budget, 1) / (area_mm2 * frac))
    if dens_eff < density:
        warnings.warn(f"max_scatterers={cfg.max_scatterers} limits speckle density "
                      f"({dens_eff * cell_mm2:.1f} scatterers/cell instead of {density * cell_mm2:.1f}); "
                      "raise max_scatterers for finer speckle statistics.")
    n_cell = dens_eff * cell_mm2
    N = int(dens_eff * area_mm2)
    X = (rng.random(N) - 0.5) * ph.width
    Z = rng.random(N) * ph.depth
    lab = ph.label_at(X, Z)
    keep = echo[lab]
    X, Z, lab = X[keep], Z[keep], lab[keep]
    g = 10 ** (db[lab] / 20)
    if cfg.texture_strength > 0:
        noise, cell = _smooth_noise(ph, cfg, rng)
        tex = ndimage.map_coordinates(noise, [Z / cell + 1, (X + ph.width / 2) / cell + 1], order=1)
        g = g * np.clip(1 + cfg.texture_strength * tex, 0.2, 3.0)
    amp = cfg.diffuse_ref_reflectivity * g * rng.standard_normal(X.size) / np.sqrt(max(n_cell, 1e-6))
    return dict(X=X, Z=Z, amp=amp, lab=lab, n_cell=n_cell)


def _path_effects(ph: Phantom, X, Z, origin, cfg, chunk=1500):
    n = X.size
    atten = np.zeros(n); yeff = np.ones(n); tx = np.ones(n); tof = np.zeros(n); plen = np.zeros(n)
    f0 = cfg.f0_hz
    lutA, lutY = ph.lut("atten_np_m", f0), ph.lut("atten_exp")
    lutC, lutZ = ph.lut("speed"), ph.lut("impedance")
    ds_nom = 2.0 * max(ph.dx, ph.dz)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        vx, vz = X[sl] - origin[sl, 0], Z[sl] - origin[sl, 1]
        L = np.hypot(vx, vz)
        S = max(2, int(np.ceil(L.max() / ds_nom)))
        fr = (np.arange(S) + 0.5) / S
        px = origin[sl, 0][:, None] + vx[:, None] * fr
        pz = origin[sl, 1][:, None] + vz[:, None] * fr
        valid = ph.inside(px, pz)
        lab = ph.label_at(px, pz)
        seg = (L / S)[:, None]
        a = lutA[lab] * valid * seg
        tot = a.sum(1)
        atten[sl] = 2.0 * tot
        yeff[sl] = np.where(tot > 1e-12, (a * lutY[lab]).sum(1) / np.maximum(tot, 1e-12), 1.0)
        c = np.where(valid, lutC[lab], cfg.c0)
        tof[sl] = (seg / c).sum(1)
        Zs = lutZ[lab]
        R = (Zs[:, 1:] - Zs[:, :-1]) / (Zs[:, 1:] + Zs[:, :-1])
        tx[sl] = np.exp(np.log(np.clip(1 - R ** 2, 1e-6, 1.0)).sum(1))
        plen[sl] = L
    return atten, yeff, tx, tof, plen


def build_scatterers(phantom: Phantom, cfg: TissueInteractionConfig) -> ScattererSet:
    rng = np.random.default_rng(cfg.seed)
    lam = cfg.wavelength
    step = lam * cfg.refine_step_over_lambda
    factor = max(1, int(np.ceil(max(phantom.dx, phantom.dz) / step)))
    while factor > 1 and phantom.nx * phantom.nz * factor ** 2 > cfg.max_refined_pixels:
        factor -= 1
    ph = phantom.refined(factor, cfg.smooth_refine)

    itf = _interface_scatterers(ph, cfg, lam)
    n_if = 0 if itf is None else itf["X"].size
    dif = _diffuse_scatterers(ph, cfg, lam, rng, cfg.max_scatterers)   # budget = diffuse only

    parts = []
    if itf is not None:
        parts.append((itf["X"], itf["Z"], itf["amp"], np.full(n_if, KIND_INTERFACE),
                      itf["post"], itf["pre"], itf["theta"]))
    if dif is not None:
        m = dif["X"].size
        parts.append((dif["X"], dif["Z"], dif["amp"], np.full(m, KIND_DIFFUSE),
                      dif["lab"], dif["lab"], np.full(m, np.nan)))
    if not parts:
        raise ValueError("phantom produced no scatterers (everything anechoic and homogeneous?)")
    X = np.concatenate([p[0] for p in parts]); Z = np.concatenate([p[1] for p in parts])
    amp = np.concatenate([p[2] for p in parts]); kind = np.concatenate([p[3] for p in parts])
    post = np.concatenate([p[4] for p in parts]).astype(np.int16)
    pre = np.concatenate([p[5] for p in parts]).astype(np.int16)
    theta = np.concatenate([p[6] for p in parts])

    origin, _ = cfg.probe.origins_and_dirs(X, Z)
    atten, yeff, tx, tof, plen = _path_effects(ph, X, Z, origin, cfg)
    m_exp = ph.lut("scatter_exp")[post]
    xyz = np.stack([X, np.zeros_like(X), Z], 1)
    return ScattererSet(
        xyz=xyz, amp=amp.astype(np.float32), kind=kind.astype(np.int8), tissue_idx=post,
        tissue_idx_pre=pre, scatter_exp=m_exp.astype(np.float32),
        path_atten_np=atten.astype(np.float32), path_y_eff=yeff.astype(np.float32),
        path_tx_loss=tx.astype(np.float32), tof=tof.astype(np.float32),
        path_len=plen.astype(np.float32), incid_angle_deg=theta.astype(np.float32),
        tissue_names=ph.tissue_names(), f0_hz=cfg.f0_hz,
        meta=dict(refine_factor=factor, n_interface=int(n_if),
                  speckle_scatterers_per_cell=None if dif is None else float(dif["n_cell"])))
