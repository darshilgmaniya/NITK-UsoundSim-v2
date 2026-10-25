"""High level API: points / phantom in -> scatterers, reflected pressure, channel data, images out."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .channel_data import compute_channel_data
from .config import TissueInteractionConfig
from .interaction import compute_reflected_pressure, make_mock_incident
from .ap_io import APForward, estimate_f0, load_ap_forward
from .outputs import (save_metadata, save_png_gray, save_montage, save_property_maps,
                      save_receiver_input, save_reflected, save_scatterers)
from .phantom import Phantom
from .points_io import (PointCloud, load_points, scatterers_from_points, tissue_cloud_to_phantom)
from .scatterers import ScattererSet, build_scatterers
from .visualize import (echo_energy_map, property_panels, qa_bmode_preview, scatterer_map)


@dataclass
class TissueInteractionResult:
    phantom: Phantom
    cfg: TissueInteractionConfig
    scatterers: ScattererSet
    reflected: Optional[np.ndarray] = None
    t_axis: Optional[np.ndarray] = None
    channel_rf: Optional[np.ndarray] = None       # (n_elements, n_samples) -> Receiver BF
    t_rx: Optional[np.ndarray] = None
    element_xz: Optional[np.ndarray] = None
    panels: Dict[str, tuple] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)


def run_stage1(phantom: Phantom, cfg: TissueInteractionConfig, outdir: Optional[str] = None):
    """Phantom -> scatterer list (+ property maps)."""
    scat = build_scatterers(phantom, cfg)
    return scat, _save_stage1(scat, phantom, cfg, outdir)


def _save_stage1(scat, phantom, cfg, outdir):
    files = {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        files["scatterers"] = os.path.join(outdir, "tissue_scatterers.npz")
        save_scatterers(files["scatterers"], scat, cfg)
        if phantom is not None and scat.meta.get("mode") != "scatterer_list":
            files["property_maps"] = os.path.join(outdir, "tissue_property_maps.npz")
            save_property_maps(files["property_maps"], phantom, cfg)
    return files


def _display_grid(scat: ScattererSet, cfg) -> Phantom:
    """Blank grid that spans a scatterer list (only used to draw the maps)."""
    xm = max(np.abs(scat.xyz[:, 0]).max(), 2e-3) + 3e-3
    zm = scat.xyz[:, 2].max() + 3e-3
    sp = max(0.25e-3, 2 * xm / 1500, zm / 1500)
    nx, nz = int(np.ceil(2 * xm / sp)), int(np.ceil(zm / sp))
    from .tissue_library import get_tissue
    return Phantom(np.zeros((nz, nx), np.int16), [get_tissue(scat.tissue_names[0])], sp, sp, name="scatterer_list")


def _finish(ph, cfg, scat, incident, t_axis, outdir, qa_preview, qa_dense, channel_data, verbose,
            full_panels, stage1_files, extra_out=None):
    log = print if verbose else (lambda *a, **k: None)
    files = dict(stage1_files)
    if incident is None:
        incident, t_axis = make_mock_incident(scat, cfg)
        log("[stage 2] using built-in MOCK incident pulse (no Acoustic Propagation input given)")
    elif t_axis is None:
        raise ValueError("t_axis is required together with a user supplied incident array")
    reflected = compute_reflected_pressure(scat, incident, t_axis, cfg)
    log(f"[stage 2] reflected_pressure_input {reflected.shape}, path_model={cfg.path_model}")
    res = TissueInteractionResult(ph, cfg, scat, reflected, t_axis, files=files)
    if channel_data:
        res.channel_rf, res.t_rx, res.element_xz = compute_channel_data(scat, reflected, t_axis, cfg)
        log(f"[stage 3] channel data (n_elements, n_samples) = {res.channel_rf.shape}  -> Receiver BF")

    panels = property_panels(ph, cfg) if full_panels else {}
    panels["scatterer_map"] = (scatterer_map(scat, ph, cfg), "effective scatterer strength (dB)")
    panels["echo_energy"] = (echo_energy_map(scat, reflected, ph, cfg), "echo energy of module output (dB)")
    if qa_preview and full_panels:
        panels["qa_bmode_preview"] = (qa_bmode_preview(ph, cfg, dense_per_cell=10.0 if qa_dense else cfg.scatterers_per_cell),
                                      "QA B-mode preview (NOT final B-mode)")
    res.panels = panels
    if outdir:
        img_dir = os.path.join(outdir, "images")
        os.makedirs(img_dir, exist_ok=True)
        for key, (img, _) in panels.items():
            save_png_gray(os.path.join(img_dir, f"{key}.png"), img)
        files["montage"] = os.path.join(outdir, "tissue_interaction_montage.png")
        save_montage(files["montage"], panels, ph,
                     f"Tissue Interaction - {ph.name}  (f0 = {cfg.f0_hz/1e6:.1f} MHz, {cfg.probe.name})")
        files["reflected"] = os.path.join(outdir, "tissue_interaction_output.npz")
        save_reflected(files["reflected"], scat, reflected, t_axis, cfg)
        if res.channel_rf is not None:
            files["receiver_bf_input"] = os.path.join(outdir, "receiver_bf_input.npz")
            save_receiver_input(files["receiver_bf_input"], res.channel_rf, res.t_rx, res.element_xz, scat, cfg,
                                extra=extra_out)
            save_png_gray(os.path.join(img_dir, "channel_data_envelope.png"), channel_image(res))
        files["metadata"] = os.path.join(outdir, "run_metadata.json")
        save_metadata(files["metadata"], ph, scat, cfg)
        files["images"] = img_dir
        log(f"[done] wrote results to {outdir}")
    return res


def channel_image(res: TissueInteractionResult, dyn_db: float = 50.0) -> np.ndarray:
    """Grayscale picture of the channel data: rows = time (down), columns = element."""
    from scipy import signal
    env = np.abs(signal.hilbert(res.channel_rf.astype(np.float64), axis=1)).T
    db = 20 * np.log10(np.maximum(env, 1e-30) / max(env.max(), 1e-30))
    return np.clip((db + dyn_db) / dyn_db, 0, 1)


def run(phantom: Phantom, cfg: TissueInteractionConfig, incident: Optional[np.ndarray] = None,
        t_axis: Optional[np.ndarray] = None, outdir: Optional[str] = None, qa_preview: bool = True,
        qa_dense: bool = True, channel_data: bool = True, verbose: bool = True) -> TissueInteractionResult:
    """Full module run from a Phantom (label grid).  If ``incident`` is None a mock incident pulse is
    used (stand-alone testing); otherwise pass the Acoustic Propagation output (n_scatterers, n_samples)
    + t_axis, rows in the order of the stage-1 file."""
    scat, f1 = run_stage1(phantom, cfg, outdir)
    if verbose:
        print(f"[stage 1] {scat.summary()}  | phantom {phantom.nx}x{phantom.nz}px")
    return _finish(phantom, cfg, scat, incident, t_axis, outdir, qa_preview, qa_dense, channel_data,
                   verbose, True, f1)


def run_points(points, cfg: TissueInteractionConfig, *, mode: str = "auto", units: str = "mm",
               background: str = "generic_soft", spacing_mm: Optional[float] = None,
               max_fill_mm: Optional[float] = None, incident=None, t_axis=None,
               outdir: Optional[str] = None, qa_preview: bool = True, qa_dense: bool = True,
               channel_data: bool = True, verbose: bool = True) -> TissueInteractionResult:
    """Full module run from POINTS / COORDINATES.

    mode='tissue_cloud' : (x, z, tissue|properties)  -> rasterised -> full phantom physics
    mode='scatterers'   : (x, z, amp)                -> used directly as scatterers
    mode='auto'         : 'scatterers' if an amp column exists, else 'tissue_cloud'."""
    pc = points if isinstance(points, PointCloud) else load_points(points, units=units)
    m = pc.mode if mode == "auto" else mode
    if verbose:
        print(pc.summary())
    if m == "tissue_cloud":
        ph = tissue_cloud_to_phantom(pc, spacing_mm, background, max_fill_mm)
        if verbose:
            print(ph.summary())
        return run(ph, cfg, incident, t_axis, outdir, qa_preview, qa_dense, channel_data, verbose)
    if m != "scatterers":
        raise ValueError("mode must be auto | tissue_cloud | scatterers")
    scat = scatterers_from_points(pc, cfg, background)
    ph = _display_grid(scat, cfg)
    f1 = _save_stage1(scat, None, cfg, outdir)
    if verbose:
        print(f"[stage 1] scatterer list used as given: {scat.summary()}")
    return _finish(ph, cfg, scat, incident, t_axis, outdir, False, False, channel_data, verbose, False, f1)


def run_from_ap(ap, cfg: Optional[TissueInteractionConfig] = None, *, phantom: Optional[Phantom] = None,
                background: str = "generic_soft", amp_from_tissue: bool = False, default_amp: float = 1.0,
                c_medium: Optional[float] = None, f0_from_pulse: bool = True, outdir: Optional[str] = None,
                channel_data: bool = True, verbose: bool = True) -> TissueInteractionResult:
    """Tissue Interaction driven by the Acoustic Propagation forward_propagation() file.

    ap       : path to the .npz or an APForward
    phantom  : optional medium (Phantom from a tissue cloud / image).  With it, attenuation, shadowing and
               time of flight are integrated through the real tissue layout; without it a homogeneous
               background tissue is used.
    The scatterers are EXACTLY the AP file's scatterer_xyz (same order); output row i <-> input row i."""
    ap = ap if isinstance(ap, APForward) else load_ap_forward(ap)
    cfg = TissueInteractionConfig.from_dict((cfg or TissueInteractionConfig()).to_dict())
    log = print if verbose else (lambda *a, **k: None)
    log(ap.summary())
    for n in ap.notes:
        log("  note:", n)
    if f0_from_pulse and ap.pulse_p is not None and ap.pulse_t is not None:
        cfg.f0_hz = estimate_f0(ap.pulse_t, ap.pulse_p)
    cfg.fs_hz = ap.fs_hz
    if ap.element_x is not None:
        cfg.probe.element_x_m, cfg.probe.element_z_m = ap.element_x.tolist(), ap.element_z.tolist()
    if c_medium is None and ap.c:
        c_medium = ap.c
    pc = PointCloud(ap.xyz[:, 0], ap.xyz[:, 2], ap.xyz[:, 1], amp=ap.amp, source="ap_forward")
    scat = scatterers_from_points(pc, cfg, background, phantom, c_medium, amp_from_tissue, default_amp)
    scat.meta["row_labels"] = ap.row_labels
    ph = phantom if phantom is not None else _display_grid(scat, cfg)
    f1 = _save_stage1(scat, phantom, cfg, outdir)
    extra = {}
    if ap.pulse_t is not None:
        extra.update(tx_pulse_t=ap.pulse_t, tx_pulse_p=ap.pulse_p)
    log(f"[stage 1] scatterers taken from the AP file: {scat.summary()}, f0 = {cfg.f0_hz/1e6:.2f} MHz, "
        f"path_model={cfg.path_model}")
    return _finish(ph, cfg, scat, ap.incident, ap.t_axis, outdir, False, False, channel_data, verbose,
                   phantom is not None, f1, extra_out=extra)
