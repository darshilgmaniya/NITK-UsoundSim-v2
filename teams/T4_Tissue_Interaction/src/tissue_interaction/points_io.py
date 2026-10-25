"""Phantom given as POINTS / COORDINATES instead of an image.

Two kinds of point input are understood.

1. tissue cloud   x, z (+ y)  and  tissue name  (or acoustic properties) per point.
                  "which tissue is at this coordinate" - the module rasterises it to a
                  grid and runs the full phantom physics (interfaces, speckle, shadows).
2. scatterer list x, z (+ y)  and  amp  [+ tissue] per point.
                  "a reflector of strength amp sits here" - the format the Acoustic
                  Propagation team uses.  Path attenuation uses one homogeneous background
                  tissue (give a tissue cloud for heterogeneous attenuation).

Accepted containers: CSV/TXT with header, .npz (keys x,z,amp,tissue or scatterer_xyz,
scatterer_amp), .npy (N x 2|3|4 -> x,z[,amp]), dict of arrays, structured array.
Column aliases: x|x_mm|x_m|lateral, z|z_mm|z_m|depth, amp|amplitude|reflectivity,
tissue|material|label, density, speed, atten_db_cm_mhz, atten_exp, backscatter_db, scatter_exp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import TissueInteractionConfig
from .phantom import Phantom
from .scatterers import KIND_DIFFUSE, KIND_INTERFACE, ScattererSet
from .tissue_library import Tissue, get_tissue, tissue_from_dict

_UNIT = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6}
_ALIAS = {"x": "x", "lateral": "x", "z": "z", "depth": "z", "y": "y", "elevation": "y",
          "amp": "amp", "amplitude": "amp", "reflectivity": "amp", "tissue": "tissue",
          "material": "tissue", "label": "tissue", "kind": "kind"}
_PROPS = ("density", "speed", "atten_db_cm_mhz", "atten_exp", "backscatter_db", "scatter_exp")


@dataclass
class PointCloud:
    x: np.ndarray                    # m
    z: np.ndarray                    # m
    y: np.ndarray                    # m
    amp: Optional[np.ndarray] = None
    tissue: Optional[np.ndarray] = None            # array of str
    props: Dict[str, np.ndarray] = field(default_factory=dict)
    kind: Optional[np.ndarray] = None
    source: str = "points"

    @property
    def n(self): return self.x.size

    @property
    def mode(self) -> str:
        return "scatterers" if self.amp is not None else "tissue_cloud"

    def summary(self) -> str:
        s = (f"PointCloud '{self.source}': {self.n} points, mode={self.mode}, "
             f"x[{self.x.min()*1e3:.1f},{self.x.max()*1e3:.1f}] z[{self.z.min()*1e3:.1f},{self.z.max()*1e3:.1f}] mm")
        if self.tissue is not None:
            u, c = np.unique(self.tissue, return_counts=True)
            s += "\n  tissues: " + ", ".join(f"{a} ({b})" for a, b in zip(u, c))
        return s


def _scale_from_name(col: str, default: float) -> float:
    c = col.lower()
    for suf, f in (("_mm", 1e-3), ("_cm", 1e-2), ("_um", 1e-6), ("_m", 1.0)):
        if c.endswith(suf):
            return f
    return default


def _read_table(path: str) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        if "scatterer_xyz" in d:
            xyz = d["scatterer_xyz"]
            d.update(x=xyz[:, 0], z=xyz[:, 2])
            if "scatterer_amp" in d:
                d["amp"] = d["scatterer_amp"]
        return d
    if ext == ".npy":
        a = np.load(path)
        if a.dtype.names:
            return {n: a[n] for n in a.dtype.names}
        if a.ndim != 2 or a.shape[1] not in (2, 3, 4):
            raise ValueError(f".npy points must be N x 2 (x,z), N x 3 (x,z,amp) or N x 4 (x,y,z,amp); got {a.shape}")
        if a.shape[1] == 4:
            return dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], amp=a[:, 3])
        return dict(zip(("x", "z", "amp"), a.T))
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    rows = [r for r in rows if r and not r[0].startswith("#")]
    head = [h.strip() for h in rows[0]]
    cols = {h: [r[i].strip() for r in rows[1:]] for i, h in enumerate(head)}
    out = {}
    for h, v in cols.items():
        try:
            out[h] = np.asarray(v, float)
        except ValueError:
            out[h] = np.asarray(v, str)
    return out


def load_points(src, *, units: str = "mm", name: Optional[str] = None, allow_bare: bool = False) -> PointCloud:
    """units: unit of x/z when the column name carries no suffix (_mm, _m, _cm)."""
    if isinstance(src, PointCloud):
        return src
    if isinstance(src, dict):
        table, label = dict(src), "dict"
    elif isinstance(src, np.ndarray):
        if src.dtype.names:
            table = {n: src[n] for n in src.dtype.names}
        elif src.ndim == 2 and src.shape[1] in (2, 3, 4):
            table = (dict(x=src[:, 0], y=src[:, 1], z=src[:, 2], amp=src[:, 3]) if src.shape[1] == 4
                     else dict(zip(("x", "z", "amp"), src.T)))
        else:
            raise ValueError(f"array of shape {src.shape} is not N x 2|3|4")
        label = "array"
    else:
        table, label = _read_table(str(src)), os.path.basename(str(src))
    if units not in _UNIT:
        raise ValueError(f"units must be one of {sorted(_UNIT)}")
    canon, scale = {}, {}
    props, other = {}, {}
    for col, v in table.items():
        base = col.lower()
        for suf in ("_mm", "_cm", "_um", "_m"):
            if base.endswith(suf) and base[: -len(suf)] in _ALIAS:
                base = base[: -len(suf)]
                break
        key = _ALIAS.get(base)
        if key in ("x", "y", "z"):
            canon[key] = np.asarray(v, float)
            scale[key] = _scale_from_name(col, _UNIT[units])
        elif key in ("amp", "tissue", "kind"):
            canon[key] = np.asarray(v)
        elif col.lower() in _PROPS:
            props[col.lower()] = np.asarray(v, float)
    if "x" not in canon or "z" not in canon:
        raise ValueError(f"need x and z columns (found {sorted(table)}). Example header: x_mm,z_mm,tissue")
    n = canon["x"].size
    y = canon["y"] * scale["y"] if "y" in canon else np.zeros(n)
    pc = PointCloud(canon["x"] * scale["x"], canon["z"] * scale["z"], y,
                    amp=canon["amp"].astype(float) if "amp" in canon else None,
                    tissue=np.asarray([str(t) for t in canon["tissue"]]) if "tissue" in canon else None,
                    props=props, kind=canon.get("kind"), source=name or label)
    for k in ("x", "z"):
        arr = getattr(pc, k)
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite values in {k}")
    if pc.z.min() < 0:
        raise ValueError("z (depth) must be >= 0: z = 0 is the transducer face / skin surface, depth grows downward")
    if pc.amp is None and pc.tissue is None and not props and not allow_bare:
        raise ValueError("points carry neither 'amp' (scatterer list) nor 'tissue'/property columns (tissue cloud)")
    for k, arr in list(props.items()):
        if arr.size != n:
            raise ValueError(f"column {k} has {arr.size} values, expected {n}")
    return pc


def _tissues_of(pc: PointCloud):
    """Return (tissue_list, per-point index)."""
    if pc.tissue is not None:
        names = list(dict.fromkeys(pc.tissue.tolist()))
        lst = [get_tissue(n) for n in names]
        lookup = {n: i for i, n in enumerate(names)}
        return lst, np.array([lookup[t] for t in pc.tissue.tolist()])
    keys = [k for k in _PROPS if k in pc.props]
    if not {"density", "speed", "atten_db_cm_mhz"} <= set(keys):
        raise ValueError("without a 'tissue' column give density, speed and atten_db_cm_mhz for every point")
    M = np.stack([pc.props[k] for k in keys], 1)
    uniq, inv = np.unique(M, axis=0, return_inverse=True)
    lst = [tissue_from_dict({"name": f"custom_{i}", **dict(zip(keys, map(float, row)))})
           for i, row in enumerate(uniq)]
    return lst, inv.ravel()


# ---------------------------------------------------------------------------
def tissue_cloud_to_phantom(pc: PointCloud, spacing_mm: Optional[float] = None,
                            background: str = "generic_soft", max_fill_mm: Optional[float] = None,
                            pad_mm: float = 0.0) -> Phantom:
    """Rasterise a tissue cloud: every grid cell takes the tissue of its nearest point
    (cells farther than max_fill from any point become background).  The grid spans
    x in [-max|x|, +max|x|] and z in [0, max z] so absolute coordinates are preserved."""
    tl, idx = _tissues_of(pc)
    pts = np.stack([pc.x, pc.z], 1)
    tree = cKDTree(pts)
    if spacing_mm is None:
        d, _ = tree.query(pts, k=2) if pc.n > 1 else (np.array([[0, 1e-3]]), None)
        spacing = float(np.median(d[:, 1]))
        spacing = max(spacing, 1e-5)
    else:
        spacing = spacing_mm * 1e-3
    fill = (max_fill_mm * 1e-3) if max_fill_mm else 2.5 * spacing
    xm = max(np.abs(pc.x).max(), spacing) + pad_mm * 1e-3 + spacing / 2
    zm = pc.z.max() + pad_mm * 1e-3 + spacing / 2
    nx = int(np.ceil(2 * xm / spacing)); nz = int(np.ceil(zm / spacing))
    if nx * nz > 40_000_000:
        raise ValueError(f"grid would be {nx}x{nz} pixels - pass a larger spacing_mm")
    x = (np.arange(nx) + 0.5 - nx / 2) * spacing
    z = (np.arange(nz) + 0.5) * spacing
    X, Z = np.meshgrid(x, z)
    d, j = tree.query(np.stack([X.ravel(), Z.ravel()], 1), k=1, distance_upper_bound=fill)
    names = [t.name for t in tl]
    bg = get_tissue(background)
    if bg.name not in names:
        tl = tl + [bg]; names.append(bg.name)
    bgi = names.index(bg.name)
    index = np.where(np.isfinite(d), idx[np.minimum(j, pc.n - 1)], bgi).reshape(nz, nx)
    return Phantom(index.astype(np.int16), tl, spacing, spacing, name=pc.source,
                   meta=dict(source=pc.source, mode="tissue_cloud", points=pc.n, spacing_mm=spacing * 1e3))


def scatterers_from_points(pc: PointCloud, cfg: TissueInteractionConfig, background: str = "generic_soft",
                           phantom: Optional[Phantom] = None, c_medium: Optional[float] = None,
                           amp_from_tissue: bool = False, default_amp: float = 1.0) -> ScattererSet:
    """Scatterer-list mode: use the given positions as the scatterers.

    amplitude : pc.amp if present, else default_amp (1.0 = unit point reflector), or - with
                amp_from_tissue and a phantom - the diffuse reflectivity of the tissue at the point.
    path      : homogeneous background tissue (c_medium overrides its speed), or - if a phantom
                is given - integrated through the phantom (heterogeneous attenuation, shadows, speed)."""
    n = pc.n
    amp = pc.amp.astype(float) if pc.amp is not None else np.full(n, float(default_amp))
    origin, _ = cfg.probe.origins_and_dirs(pc.x, pc.z)
    kind = np.full(n, KIND_DIFFUSE, np.int8)
    if pc.kind is not None:
        kind = np.array([KIND_INTERFACE if str(k).lower() in ("1", "interface", "specular") else KIND_DIFFUSE
                         for k in pc.kind], np.int8)
    if phantom is not None:
        from .scatterers import _path_effects
        lam = cfg.wavelength
        factor = max(1, int(np.ceil(max(phantom.dx, phantom.dz) / (lam * cfg.refine_step_over_lambda))))
        while factor > 1 and phantom.nx * phantom.nz * factor ** 2 > cfg.max_refined_pixels:
            factor -= 1
        ph = phantom.refined(factor, cfg.smooth_refine)
        tidx = ph.label_at(pc.x, pc.z).astype(np.int16)
        atten, yeff, tx, tof, plen = _path_effects(ph, pc.x, pc.z, origin, cfg)
        if amp_from_tissue and pc.amp is None:
            amp = cfg.diffuse_ref_reflectivity * 10 ** (ph.lut("backscatter_db")[tidx] / 20)
        return ScattererSet(
            xyz=np.stack([pc.x, pc.y, pc.z], 1), amp=amp.astype(np.float32), kind=kind,
            tissue_idx=tidx, tissue_idx_pre=tidx, scatter_exp=ph.lut("scatter_exp")[tidx].astype(np.float32),
            path_atten_np=atten.astype(np.float32), path_y_eff=yeff.astype(np.float32),
            path_tx_loss=tx.astype(np.float32), tof=tof.astype(np.float32), path_len=plen.astype(np.float32),
            incid_angle_deg=np.full(n, np.nan, np.float32), tissue_names=ph.tissue_names(), f0_hz=cfg.f0_hz,
            meta=dict(mode="scatterer_list+phantom"))
    bg = get_tissue(background)
    names = [bg.name]
    if pc.tissue is not None:
        for t in pc.tissue.tolist():
            nm = get_tissue(t).name
            if nm not in names:
                names.append(nm)
        tidx = np.array([names.index(get_tissue(t).name) for t in pc.tissue.tolist()], np.int16)
    else:
        tidx = np.zeros(n, np.int16)
    tl = [get_tissue(nm) for nm in names]
    L = np.hypot(pc.x - origin[:, 0], pc.z - origin[:, 1])
    a_np = bg.attenuation_np_m(cfg.f0_hz)
    c_bg = float(c_medium) if c_medium else bg.speed
    return ScattererSet(
        xyz=np.stack([pc.x, pc.y, pc.z], 1), amp=amp.astype(np.float32), kind=kind,
        tissue_idx=tidx, tissue_idx_pre=tidx,
        scatter_exp=np.array([tl[i].scatter_exp for i in tidx], np.float32),
        path_atten_np=(2 * a_np * L).astype(np.float32),
        path_y_eff=np.full(n, bg.atten_exp, np.float32), path_tx_loss=np.ones(n, np.float32),
        tof=(L / c_bg).astype(np.float32), path_len=L.astype(np.float32),
        incid_angle_deg=np.full(n, np.nan, np.float32), tissue_names=names, f0_hz=cfg.f0_hz,
        meta=dict(mode="scatterer_list", background=bg.name, c_medium=c_bg))


def save_points_csv(path: str, ph: Phantom, spacing_mm: Optional[float] = None):
    """Export a phantom as a tissue-cloud CSV (x_mm, z_mm, tissue)."""
    step = 1
    if spacing_mm:
        step = max(1, int(round(spacing_mm * 1e-3 / ph.dx)))
    xs, zs = ph.x_coords()[::step], ph.z_coords()[::step]
    idx = ph.index[::step, ::step]
    names = np.array(ph.tissue_names())
    with open(path, "w") as fh:
        fh.write("x_mm,z_mm,tissue\n")
        for i, zv in enumerate(zs):
            row = names[idx[i]]
            fh.write("\n".join(f"{x*1e3:.4f},{zv*1e3:.4f},{t}" for x, t in zip(xs, row)) + "\n")
