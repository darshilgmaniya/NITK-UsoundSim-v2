"""Load a USER-SUPPLIED phantom and turn it into a :class:`Phantom`.

Supported inputs
----------------
* image files (png / jpg / tif / bmp) and numpy files (.npy / .npz)
* a numpy array passed straight from a notebook

Interpretation modes (``mode=``)
--------------------------------
labels     pixel value == label id  (0,1,2,... segmentation mask)        [default for integer data]
color      RGB colour == tissue (colour-coded segmentation)
intensity  grey level -> tissue classes (anatomical / CT / MRI-like image)  [default for float data]
auto       RGB image -> color, integer grey -> labels, float -> intensity

The label -> tissue mapping comes from a dict / JSON / YAML file::

    {
      "spacing_mm": [0.2, 0.2],
      "labels": {"0": "water", "1": "liver",
                 "2": {"name": "my_lesion", "density": 1070, "speed": 1570,
                       "atten_db_cm_mhz": 0.9, "backscatter_db": -8}},
      "colors": {"#ff0000": "liver", "0,0,255": "blood"},
      "ranges": [[0, 60, "water"], [60, 160, "liver"], [160, 256, "bone"]],
      "classes": ["fat", "liver", "bone"]
    }

or from ``body_part=`` (uses that preset's default label legend).
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .body_presets import get_preset
from .phantom import Phantom
from .tissue_library import Tissue, get_tissue

ArrayLike = Union[str, os.PathLike, np.ndarray]


# ----------------------------------------------------------------------------
def _read_mapping(mapping) -> dict:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return dict(mapping)
    path = str(mapping)
    with open(path) as fh:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def _read_array(src: ArrayLike) -> Tuple[np.ndarray, str]:
    if isinstance(src, np.ndarray):
        return src, "array"
    path = str(src)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path), os.path.basename(path)
    if ext == ".npz":
        z = np.load(path)
        for key in ("labels", "phantom", "image", "index"):
            if key in z:
                return z[key], os.path.basename(path)
        return z[z.files[0]], os.path.basename(path)
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("P", "1", "LA", "PA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB") if im.mode != "1" else im.convert("L")
    if im.mode == "RGBA":
        im = im.convert("RGB")
    return np.asarray(im), os.path.basename(path)


def _parse_color(key) -> np.ndarray:
    s = str(key).strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], float)
    parts = [float(p) for p in s.replace("(", "").replace(")", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"cannot parse colour key '{key}' (use '#rrggbb' or 'r,g,b')")
    return np.array(parts, float)


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
    v = values.ravel().astype(np.float64)
    if v.size > 400_000:
        v = np.random.default_rng(0).choice(v, 400_000, replace=False)
    c = np.linspace(v.min(), v.max(), k)          # spread init: robust for skewed histograms
    for _ in range(iters):
        a = np.abs(v[:, None] - c[None, :]).argmin(1)
        new = np.array([v[a == j].mean() if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.sort(c)


def _spacing(spacing_mm, width_mm, mapping, shape) -> Tuple[float, float]:
    if spacing_mm is None and width_mm is None:
        spacing_mm = mapping.get("spacing_mm")
        width_mm = mapping.get("width_mm") if spacing_mm is None else None
    if spacing_mm is not None:
        if np.isscalar(spacing_mm):
            return float(spacing_mm) * 1e-3, float(spacing_mm) * 1e-3
        dx, dz = spacing_mm
        return float(dx) * 1e-3, float(dz) * 1e-3
    if width_mm is not None:
        d = float(width_mm) * 1e-3 / shape[1]
        return d, d
    raise ValueError("physical scale unknown: pass spacing_mm=<mm per pixel> (or (dx, dz)) "
                     "or width_mm=<phantom width>, or put it in the mapping file. "
                     "A wrong scale silently gives wrong physics, so there is no default.")


def _compact(raw_tissues: Dict[int, Tissue], raw_index: np.ndarray):
    """Merge ids that map to the same tissue name -> compact index + tissue list."""
    names: List[str] = []
    tissues: List[Tissue] = []
    first_id: List[int] = []
    lut = np.zeros(int(raw_index.max()) + 1, np.int16)
    for rid in sorted(raw_tissues):
        t = raw_tissues[rid]
        if t.name not in names:
            names.append(t.name)
            tissues.append(t)
            first_id.append(rid)
        lut[rid] = names.index(t.name)
    return lut[raw_index], tissues, first_id


# ----------------------------------------------------------------------------
def load_phantom(src: ArrayLike, *, mode: str = "auto", mapping=None,
                 spacing_mm=None, width_mm=None, body_part: Optional[str] = None,
                 flipud: bool = False, strict: bool = True, name: Optional[str] = None) -> Phantom:
    arr, label = _read_array(src)
    mp = _read_mapping(mapping)
    body_part = body_part or mp.get("body_part")
    arr = np.squeeze(arr)
    if flipud:
        arr = arr[::-1]
    is_rgb = arr.ndim == 3 and arr.shape[2] in (3, 4)
    if arr.ndim == 3 and not is_rgb:
        raise ValueError(f"3-D arrays are not supported yet (got shape {arr.shape}); "
                         "pass a 2-D slice. (3-D support is on the roadmap.)")
    if is_rgb:
        arr = arr[..., :3]
    if mode == "auto":
        if is_rgb:
            mode = "color" if "colors" in mp else "intensity"
        elif np.issubdtype(arr.dtype, np.integer) or np.allclose(arr, np.round(arr)):
            mode = "intensity" if ("ranges" in mp or "classes" in mp) and "labels" not in mp else "labels"
        else:
            mode = "intensity"
    shape2 = arr.shape[:2]
    dx, dz = _spacing(spacing_mm, width_mm, mp, shape2)

    if mode == "labels":
        if is_rgb:
            arr = arr.mean(2)
        raw = np.round(arr).astype(np.int64)
        if raw.min() < 0:
            raise ValueError("label ids must be >= 0")
        table = {int(k): v for k, v in (mp.get("labels") or {}).items()}
        if not table and body_part:
            table = dict(get_preset(body_part).legend)
        if not table:
            raise ValueError(f"no label->tissue mapping for ids {np.unique(raw).tolist()}. "
                             "Provide mapping={'labels': {...}} or body_part=...")
        rt: Dict[int, Tissue] = {}
        for rid in np.unique(raw):
            rid = int(rid)
            if rid in table:
                rt[rid] = get_tissue(table[rid])
            elif strict:
                raise KeyError(f"label id {rid} has no tissue in the mapping (known: {sorted(table)}). "
                               "Add it or pass strict=False to fall back to generic_soft.")
            else:
                warnings.warn(f"label id {rid} unmapped -> generic_soft")
                rt[rid] = get_tissue("generic_soft")
        index, tissues, ids = _compact(rt, raw)

    elif mode == "color":
        if not is_rgb:
            raise ValueError("mode='color' needs an RGB image")
        cols = mp.get("colors")
        if not cols:
            raise ValueError("mode='color' needs mapping['colors'] = {'#rrggbb': tissue, ...}")
        keys = list(cols)
        ref = np.stack([_parse_color(k) for k in keys])              # (K,3)
        flat = arr.reshape(-1, 3).astype(np.float64)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        d = ((uniq[:, None, :] - ref[None]) ** 2).sum(2)
        nearest = d.argmin(1)
        raw = nearest[inv.ravel()].reshape(shape2)
        rt = {i: get_tissue(cols[k]) for i, k in enumerate(keys) if (raw == i).any()}
        index, tissues, ids = _compact(rt, raw)

    elif mode == "intensity":
        g = arr.mean(2) if is_rgb else arr.astype(np.float64)
        if g.max() > 255 or g.min() < 0 or not np.issubdtype(arr.dtype, np.integer):
            lo, hi = float(g.min()), float(g.max())
            g = (g - lo) / max(hi - lo, 1e-12) * 255.0
        if "ranges" in mp:
            raw = np.zeros(shape2, np.int64)
            rt = {}
            for i, (lo, hi, tname) in enumerate(mp["ranges"]):
                sel = (g >= lo) & (g < hi) if hi < 256 else (g >= lo)
                raw[sel] = i
                rt[i] = get_tissue(tname)
        elif "classes" in mp:
            cl = mp["classes"]
            centres = _kmeans_1d(g, len(cl))
            raw = np.abs(g[..., None] - centres).argmin(-1)
            rt = {i: get_tissue(t) for i, t in enumerate(cl)}
        else:
            raise ValueError("mode='intensity' needs mapping['ranges'] or mapping['classes'] "
                             "(tissues listed dark -> bright)")
        used = np.unique(raw)
        rt = {int(i): rt[int(i)] for i in used}
        index, tissues, ids = _compact(rt, raw)
    else:
        raise ValueError(f"unknown mode '{mode}'")

    return Phantom(index, tissues, dx, dz, ids, name or label,
                   {"source": label, "mode": mode, "body_part": body_part})
