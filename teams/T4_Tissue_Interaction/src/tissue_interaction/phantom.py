"""Phantom container: a 2-D label map + tissue per label + pixel spacing.

Coordinates (SI metres)
    x : lateral, 0 at the phantom centre, +x to the right
    z : depth,   0 at the skin / probe face, +z downward
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from .tissue_library import Tissue


@dataclass
class Phantom:
    index: np.ndarray                  # (nz, nx) int16, compact 0..K-1
    tissue_list: List[Tissue]          # length K
    dx: float                          # m per pixel (lateral)
    dz: float                          # m per pixel (depth)
    label_ids: Optional[List[int]] = None   # original label ids (for export)
    name: str = "phantom"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = np.ascontiguousarray(self.index.astype(np.int16))
        if self.label_ids is None:
            self.label_ids = list(range(len(self.tissue_list)))
        if self.index.ndim != 2:
            raise ValueError("phantom must be 2-D")
        if self.index.max() >= len(self.tissue_list) or self.index.min() < 0:
            raise ValueError("index map refers to a tissue that is not in tissue_list")

    # ---- geometry ---------------------------------------------------------
    @property
    def nz(self): return self.index.shape[0]

    @property
    def nx(self): return self.index.shape[1]

    @property
    def width(self): return self.nx * self.dx

    @property
    def depth(self): return self.nz * self.dz

    def x_coords(self): return (np.arange(self.nx) + 0.5 - self.nx / 2) * self.dx

    def z_coords(self): return (np.arange(self.nz) + 0.5) * self.dz

    def extent_m(self):
        """(xmin, xmax, zmax, zmin) for matplotlib imshow(extent=...)."""
        return (-self.width / 2, self.width / 2, self.depth, 0.0)

    # ---- properties -------------------------------------------------------
    def lut(self, attr: str, f0_hz: float = 3.5e6) -> np.ndarray:
        if attr == "impedance":
            vals = [t.impedance for t in self.tissue_list]
        elif attr == "atten_np_m":
            vals = [t.attenuation_np_m(f0_hz) for t in self.tissue_list]
        elif attr == "atten_db_cm":
            vals = [t.attenuation_db_cm(f0_hz) for t in self.tissue_list]
        else:
            vals = [getattr(t, attr) for t in self.tissue_list]
        return np.asarray(vals, dtype=np.float64)

    def property_map(self, attr: str, f0_hz: float = 3.5e6) -> np.ndarray:
        return self.lut(attr, f0_hz)[self.index]

    def tissue_names(self) -> List[str]:
        return [t.name for t in self.tissue_list]

    # ---- sampling ---------------------------------------------------------
    def pixel_indices(self, x, z):
        ix = np.floor(np.asarray(x) / self.dx + self.nx / 2).astype(np.int64)
        iz = np.floor(np.asarray(z) / self.dz).astype(np.int64)
        return iz, ix

    def inside(self, x, z):
        iz, ix = self.pixel_indices(x, z)
        return (iz >= 0) & (iz < self.nz) & (ix >= 0) & (ix < self.nx)

    def label_at(self, x, z):
        """Nearest-pixel compact label (edge pixels are extended outside)."""
        iz, ix = self.pixel_indices(x, z)
        iz = np.clip(iz, 0, self.nz - 1)
        ix = np.clip(ix, 0, self.nx - 1)
        return self.index[iz, ix]

    # ---- resampling -------------------------------------------------------
    def refined(self, factor: int, smooth: bool = True) -> "Phantom":
        """Up-sample the label map by an integer factor.  smooth=True rounds the
        staircase edges of low-resolution phantoms (one-hot linear zoom + argmax)."""
        factor = int(factor)
        if factor <= 1:
            return self
        K = len(self.tissue_list)
        shape = (self.nz * factor, self.nx * factor)
        if not smooth or K == 1:
            out = np.kron(self.index, np.ones((factor, factor), np.int16))
        else:
            out = np.zeros(shape, np.int16)
            best = np.full(shape, -1.0, np.float32)
            for k in range(K):
                m = (self.index == k).astype(np.float32)
                if not m.any():
                    continue
                z = ndimage.zoom(m, factor, order=1, mode="nearest", grid_mode=True)
                upd = z > best
                out[upd] = k
                best[upd] = z[upd]
        return Phantom(out, self.tissue_list, self.dx / factor, self.dz / factor,
                       self.label_ids, self.name, dict(self.meta))

    def summary(self) -> str:
        n = self.index.size
        lines = [f"Phantom '{self.name}': {self.nx} x {self.nz} px, "
                 f"{self.width*1e3:.1f} x {self.depth*1e3:.1f} mm, "
                 f"pixel {self.dx*1e3:.3f} x {self.dz*1e3:.3f} mm"]
        for k, t in enumerate(self.tissue_list):
            frac = (self.index == k).sum() / n
            lines.append(f"  [{self.label_ids[k]:>3}] {t.name:<18} Z={t.impedance_mrayl:6.3f} MRayl  "
                         f"c={t.speed:6.0f} m/s  {100*frac:5.1f}% of area")
        return "\n".join(lines)
