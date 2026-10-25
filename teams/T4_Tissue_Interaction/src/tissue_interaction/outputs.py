"""Writers / readers: npz hand-off files, grayscale PNGs, montage, metadata."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .config import TissueInteractionConfig
from .phantom import Phantom
from .scatterers import ScattererSet

_SC_FIELDS = ["xyz", "amp", "kind", "tissue_idx", "tissue_idx_pre", "scatter_exp", "path_atten_np",
              "path_y_eff", "path_tx_loss", "tof", "path_len", "incid_angle_deg"]


# ---------------------------------------------------------------- scatterers
def save_scatterers(path: str, scat: ScattererSet, cfg: TissueInteractionConfig):
    """STAGE-1 hand-off file: give this to the Acoustic Propagation team.
    Their forward_propagation() needs ``scatterer_xyz`` (+ ``scatterer_amp``)."""
    d = {f"sc_{k}": getattr(scat, k) for k in _SC_FIELDS}
    d.update(
        description=np.array("Tissue Interaction stage 1: scatterer list generated from the user phantom. "
                             "Row order is the contract: row i of every later array refers to scatterer i."),
        scatterer_xyz=scat.xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]),
        scatterer_xyz_units=np.array("m"),
        scatterer_amp=scat.amp, scatterer_kind=scat.kind,
        scatterer_kind_meaning=np.array("0 = diffuse (speckle), 1 = interface (specular)"),
        scatterer_row_labels=scat.row_labels(), scatterer_tof_s=scat.tof,
        tissue_names=np.array(scat.tissue_names), f0_hz=np.array(scat.f0_hz),
        config_json=np.array(json.dumps(cfg.to_dict())))
    np.savez_compressed(path, **d)


def load_scatterers(path: str) -> Tuple[ScattererSet, TissueInteractionConfig]:
    z = np.load(path, allow_pickle=False)
    kw = {k: z[f"sc_{k}"] for k in _SC_FIELDS}
    scat = ScattererSet(tissue_names=[str(s) for s in z["tissue_names"]], f0_hz=float(z["f0_hz"]), **kw)
    cfg = TissueInteractionConfig.from_dict(json.loads(str(z["config_json"])))
    return scat, cfg


def load_incident(path: str, key: Optional[str] = None):
    """Read Acoustic Propagation's incident pressure.  Returns (incident, t_axis|None)."""
    z = np.load(path, allow_pickle=False)
    if key is None:
        for k in ("incident_pressure", "incident_pressure_at_scatterers", "p_incident", "incident"):
            if k in z:
                key = k
                break
    if key is None:
        cands = [k for k in z.files if z[k].ndim == 2 and np.issubdtype(z[k].dtype, np.floating)]
        if len(cands) != 1:
            raise KeyError(f"cannot pick the incident array automatically from keys {z.files}; "
                           "pass key='...'")
        key = cands[0]
    t = None
    for tk in ("t_axis", "t", "time"):
        if tk in z:
            t = z[tk]
            break
    return z[key], t


# ---------------------------------------------------------------- reflected
def save_reflected(path: str, scat: ScattererSet, reflected: np.ndarray, t_axis: np.ndarray,
                   cfg: TissueInteractionConfig, pressure_units: str = "same units as the incident pressure"):
    """STAGE-2 hand-off file.  Drop-in replacement for the Acoustic Propagation
    team's mock ``reflected_pressure_input`` (same key names, same shape rule)."""
    np.savez_compressed(
        path,
        description=np.array("Tissue Interaction output: reflected/scattered pressure waveform of each "
                             "scatterer (one row per scatterer, NOT per receive element)."),
        t_axis=t_axis, t_axis_units=np.array("s"),
        reflected_pressure_input=reflected,
        reflected_pressure_input_shape_meaning=np.array("(n_scatterers, n_samples)"),
        reflected_pressure_input_units=np.array(pressure_units),
        reflected_pressure_input_row_labels=scat.row_labels(),
        scatterer_xyz=scat.xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]),
        scatterer_xyz_units=np.array("m"),
        scatterer_kind=scat.kind, tissue_names=np.array(scat.tissue_names),
        scatterer_tissue_idx=scat.tissue_idx,
        path_model=np.array(cfg.path_model), f0_hz=np.array(cfg.f0_hz))


def save_property_maps(path: str, ph: Phantom, cfg: TissueInteractionConfig):
    """Acoustic property maps for the Acoustic Propagation team (c, rho, alpha)."""
    np.savez_compressed(
        path,
        description=np.array("Acoustic property maps of the user phantom. Axis 0 = depth z, axis 1 = lateral x."),
        x_axis_m=ph.x_coords(), z_axis_m=ph.z_coords(), pixel_spacing_m=np.array([ph.dx, ph.dz]),
        tissue_index=ph.index, tissue_names=np.array(ph.tissue_names()),
        density_kg_m3=ph.property_map("density").astype(np.float32),
        speed_m_s=ph.property_map("speed").astype(np.float32),
        impedance_rayl=ph.property_map("impedance").astype(np.float32),
        attenuation_db_cm_at_f0=ph.property_map("atten_db_cm", cfg.f0_hz).astype(np.float32),
        attenuation_exponent=ph.property_map("atten_exp").astype(np.float32),
        backscatter_db=ph.property_map("backscatter_db").astype(np.float32), f0_hz=np.array(cfg.f0_hz))


def save_metadata(path: str, ph: Phantom, scat: ScattererSet, cfg: TissueInteractionConfig, extra=None):
    meta = dict(phantom=dict(name=ph.name, shape=list(ph.index.shape), dx_m=ph.dx, dz_m=ph.dz,
                             width_mm=ph.width * 1e3, depth_mm=ph.depth * 1e3, meta=ph.meta),
                tissues=[dict(name=t.name, kind=t.kind, density=t.density, speed=t.speed,
                              impedance_mrayl=t.impedance_mrayl, atten_db_cm_mhz=t.atten_db_cm_mhz,
                              atten_exp=t.atten_exp, backscatter_db=t.backscatter_db)
                         for t in ph.tissue_list],
                scatterers=dict(total=scat.n, interface=int((scat.kind == 1).sum()),
                                diffuse=int((scat.kind == 0).sum()),
                                **{k: v for k, v in scat.meta.items() if k != "row_labels"}),
                config=cfg.to_dict())
    if extra:
        meta.update(extra)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, default=float)


# ---------------------------------------------------------------- images
def save_png_gray(path: str, arr01: np.ndarray):
    from PIL import Image
    Image.fromarray((np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(path)


def save_montage(path: str, panels: Dict[str, Tuple[np.ndarray, str]], ph: Phantom, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    ext = tuple(v * 1e3 for v in ph.extent_m())
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (key, (img, lab)) in zip(axs.ravel(), panels.items()):
        ax.axis("on")
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, aspect="equal", interpolation="nearest")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8); ax.set_ylabel("depth z [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- receiver BF
def save_receiver_input(path: str, rf: np.ndarray, t_rx: np.ndarray, element_xz: np.ndarray,
                        scat: ScattererSet, cfg: TissueInteractionConfig,
                        pressure_units: str = "same units as the incident pressure", extra: Optional[dict] = None):
    """STAGE-3 hand-off file for the RECEIVER BEAMFORMING team (per-element echoes)."""
    dt = float(t_rx[1] - t_rx[0])
    extra = {k: np.asarray(v) for k, v in (extra or {}).items()}
    np.savez_compressed(
        path, **extra,
        description=np.array("Tissue Interaction output for Receiver Beamforming: echoes recorded at every "
                             "receive element. Time zero = start of transmission (reference of the incident pressure)."),
        t_axis=t_rx, t_axis_units=np.array("s"), sampling_rate_hz=np.array(1.0 / dt),
        received_pressure=rf,
        received_pressure_shape_meaning=np.array("(n_elements, n_samples)"),
        received_pressure_units=np.array(pressure_units),
        element_x=element_xz[:, 0], element_z=element_xz[:, 1], element_x_units=np.array("m"),
        element_xz=element_xz,
        f0_hz=np.array(cfg.f0_hz), c0_nominal_m_s=np.array(cfg.c0),
        probe_geometry=np.array(cfg.probe.geometry), probe_name=np.array(cfg.probe.name),
        scatterer_xyz=scat.xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]),
        scatterer_xyz_units=np.array("m"), path_model=np.array(cfg.path_model),
        config_json=np.array(json.dumps(cfg.to_dict())))


def load_elements(path: str):
    """Receive-element positions from the Transducer / AP team: .npz/.npy/.csv with
    element_x (m) [and element_z]."""
    if path.lower().endswith(".npz"):
        z = np.load(path)
        ex = z["element_x"]
        ez = z["element_z"] if "element_z" in z.files else np.zeros_like(ex)
    elif path.lower().endswith(".npy"):
        ex = np.load(path).ravel(); ez = np.zeros_like(ex)
    else:
        a = np.atleast_2d(np.genfromtxt(path, delimiter=",", names=True))
        names = a.dtype.names
        ex = np.asarray(a[names[0]], float)
        ez = np.asarray(a[names[1]], float) if len(names) > 1 else np.zeros_like(ex)
        if names[0].endswith("_mm"):
            ex, ez = ex * 1e-3, ez * 1e-3
    return np.asarray(ex, float).ravel(), np.asarray(ez, float).ravel()
