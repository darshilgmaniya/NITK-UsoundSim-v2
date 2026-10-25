"""
relay.py - the RELAY RACE that joins Acoustic Propagation (Team 3) and Tissue Interaction (Subtask 04)
=====================================================================================================

    runner 1  TISSUE INTERACTION  stage 1   phantom / points      -> scatterer list (x,y,z, amp, labels)
        |  baton: scatterer_xyz (N,3), row order = contract
    runner 2  ACOUSTIC PROP.      part 1    128 TX elements       -> incident_pressure (N, T)
        |  baton: incident_pressure + t_axis
    runner 3  TISSUE INTERACTION  stage 2   incident pressure     -> reflected_pressure_input (N, T)
        |  baton: reflected_pressure_input (one waveform per scatterer, NOT per element)
    runner 4  ACOUSTIC PROP.      part 2    reflected pressure    -> received_pressure (128, T)
        |  baton: received_pressure  ==>  RECEIVER BEAMFORMING team

Every hand-over is kept in memory and (if ``outdir`` is given) written to its own .npz so each team
can inspect what it received.  ``nitk_usoundsim.config`` is the single source of truth for the array
(128 elements), f0, fs, c and alpha0: the Tissue Interaction config is derived from it.

Who owns attenuation?  (``attenuation_owner``)
    'tissue' (default)  Tissue Interaction applies tissue-specific two-way attenuation, interface
                        transmission loss (shadows) -> AP is run with alpha0 = 0 (no double counting).
    'ap'                AP applies its uniform alpha0 (dB/MHz/cm) -> TI path_model = 'none'.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from nitk_usoundsim import acoustic_propagation as ap
from nitk_usoundsim import config as apcfg
from tissue_interaction import (Phantom, ProbeConfig, TissueInteractionConfig, build_scatterers,
                                compute_channel_data, compute_reflected_pressure, load_phantom,
                                load_points, make_phantom, scatterers_from_points,
                                tissue_cloud_to_phantom)
from tissue_interaction.outputs import save_reflected, save_scatterers
from tissue_interaction.phantom_generators import GENERATORS
from tissue_interaction.points_io import PointCloud
from tissue_interaction.scatterers import ScattererSet


@dataclass
class RelayConfig:
    """Array / medium parameters.  Defaults come from nitk_usoundsim/config.py."""
    n_elements: int = apcfg.N_ELEMENTS      # 128
    pitch_m: float = apcfg.PITCH
    f0_hz: float = apcfg.F0
    fs_hz: float = apcfg.FS
    c0: float = apcfg.C
    alpha0: float = apcfg.ALPHA_0
    n_cycles: int = ap.N_CYCLES
    time_margin_fraction: float = 0.15
    attenuation_owner: str = "tissue"       # 'tissue' | 'ap'
    tau_tx: Optional[np.ndarray] = None     # per-element TX delays [s] (hook for Transmit BF team)
    max_scatterers: int = 8000              # TI diffuse-scatterer budget (memory/time)
    seed: int = 0
    also_ti_channel: bool = True            # also compute TI stage 3 (tissue-aware delays) for comparison


def make_ti_config(rc: RelayConfig, **overrides) -> TissueInteractionConfig:
    """Tissue Interaction config that matches the Acoustic Propagation array exactly."""
    if rc.attenuation_owner not in ("tissue", "ap"):
        raise ValueError("attenuation_owner must be 'tissue' or 'ap'")
    cfg = TissueInteractionConfig(f0_hz=rc.f0_hz, c0=rc.c0, fs_hz=rc.fs_hz,
                                  max_scatterers=rc.max_scatterers, seed=rc.seed,
                                  path_model="full" if rc.attenuation_owner == "tissue" else "none")
    cfg.probe = ProbeConfig(name=f"NITK-UsoundSim linear {rc.n_elements}-element", geometry="linear",
                            n_elements=rc.n_elements, pitch_m=rc.pitch_m)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@dataclass
class RelayResult:
    cfg: TissueInteractionConfig
    rc: RelayConfig
    phantom: Optional[Phantom]
    scatterers: ScattererSet
    element_x: np.ndarray
    t_axis: np.ndarray
    incident: np.ndarray                    # (N, T)       runner 2 output
    reflected: np.ndarray                   # (N, T)       runner 3 output
    received_pressure: np.ndarray           # (128, T)     runner 4 output -> Receiver BF
    pulse_t: np.ndarray
    pulse_p: np.ndarray
    ti_channel: Optional[np.ndarray] = None  # (128, T')    alternative: TI stage 3
    ti_channel_t: Optional[np.ndarray] = None
    log: List[dict] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)   # seconds per runner


# ----------------------------------------------------------------------------------------------
def _is_points(source, mapping) -> bool:
    if isinstance(source, (PointCloud, np.ndarray, dict)):
        return True
    return mapping is None and str(source).lower().endswith((".csv", ".txt", ".npz", ".npy"))


def _runner1_scatterers(source, cfg, *, mapping=None, spacing_mm=None, background="generic_soft",
                        units="mm"):
    """Accepts: Phantom | preset name | image path (+mapping) | points file / PointCloud."""
    ph = None
    if isinstance(source, Phantom):
        ph = source
    elif isinstance(source, str) and source.strip().lower() in GENERATORS:
        ph = make_phantom(source)
    elif _is_points(source, mapping):
        pc = source if isinstance(source, PointCloud) else load_points(source, units=units)
        if pc.mode == "scatterers":
            return scatterers_from_points(pc, cfg, background), None
        ph = tissue_cloud_to_phantom(pc, spacing_mm, background, None)
    else:
        ph = load_phantom(source, mapping=mapping, spacing_mm=spacing_mm)
    return build_scatterers(ph, cfg), ph


def _time_axis(xyz, ex, rc: RelayConfig, half_width: float) -> np.ndarray:
    d = np.sqrt((xyz[:, [0]] - ex[None]) ** 2 + xyz[:, [2]] ** 2)
    tau = np.zeros(ex.size) if rc.tau_tx is None else np.asarray(rc.tau_tx, float)
    t_tx = (d / rc.c0 + tau[None]).max(1)
    t_rx = (d / rc.c0).max(1)
    mx = float((t_tx + t_rx).max())
    return np.arange(-half_width, mx * (1 + rc.time_margin_fraction) + 2 * half_width, 1.0 / rc.fs_hz)


def _baton(log, name, owner, **arrays):
    shapes = {k: tuple(np.shape(v)) for k, v in arrays.items()}
    log.append(dict(runner=name, owner=owner, shapes=shapes))
    return shapes


def run_relay(source, rc: Optional[RelayConfig] = None, *, mapping=None, spacing_mm=None,
              background: str = "generic_soft", units: str = "mm", outdir: Optional[str] = None,
              verbose: bool = True, **ti_overrides) -> RelayResult:
    rc = rc or RelayConfig()
    cfg = make_ti_config(rc, **ti_overrides)
    say = print if verbose else (lambda *a, **k: None)
    log: List[dict] = []
    files: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    ex = ap.create_linear_array(rc.n_elements, rc.pitch_m)
    alpha_ap = 0.0 if rc.attenuation_owner == "tissue" else rc.alpha0
    say(f"[setup ] {rc.n_elements} elements, pitch {rc.pitch_m*1e3:.3f} mm, aperture "
        f"{(ex[-1]-ex[0])*1e3:.1f} mm, f0 {rc.f0_hz/1e6:.1f} MHz, fs {rc.fs_hz/1e6:.0f} MHz, "
        f"attenuation owner = {rc.attenuation_owner}")

    # ---- runner 1: TI stage 1 ---------------------------------------------------------------
    t0 = time.time()
    scat, ph = _runner1_scatterers(source, cfg, mapping=mapping, spacing_mm=spacing_mm,
                                   background=background, units=units)
    xyz = scat.xyz
    timings['runner1_ti_stage1'] = time.time() - t0
    _baton(log, "1 TI stage 1  (phantom -> scatterers)", "Tissue Interaction", scatterer_xyz=xyz)
    say(f"[run 1 ] TI stage 1      : {scat.summary()}   ({time.time()-t0:.1f}s)")
    if outdir:
        files["01_scatterers"] = os.path.join(outdir, "01_tissue_scatterers.npz")
        save_scatterers(files["01_scatterers"], scat, cfg)

    # ---- runner 2: AP part 1 ----------------------------------------------------------------
    t0 = time.time()
    pulse_t, pulse_p = ap.generate_pulse(fs=rc.fs_hz, f0=rc.f0_hz, n_cycles=rc.n_cycles)
    t_axis = _time_axis(xyz, ex, rc, pulse_t[-1])
    incident = ap.forward_propagation_many(xyz[:, 0], xyz[:, 2], ex, t_axis, f0=rc.f0_hz, c=rc.c0,
                                           alpha0=alpha_ap, n_cycles=rc.n_cycles, tau_tx=rc.tau_tx)
    timings['runner2_ap_part1'] = time.time() - t0
    _baton(log, "2 AP part 1   (TX array -> scatterers)", "Acoustic Propagation",
           incident_pressure=incident, t_axis=t_axis)
    say(f"[run 2 ] AP part 1       : incident_pressure {incident.shape}, t = "
        f"{t_axis[0]*1e6:.1f}..{t_axis[-1]*1e6:.1f} us   ({time.time()-t0:.1f}s)")
    if outdir:
        files["02_part1"] = os.path.join(outdir, "02_part1_forward_propagation_outputs.npz")
        np.savez_compressed(
            files["02_part1"],
            description=np.array(f"PART 1 - forward_propagation(): incident pressure AT EACH SCATTERER from the "
                                 f"{rc.n_elements}-element TX array vs time. Row i <-> scatterer_xyz[i]."),
            t_axis=t_axis, t_axis_units=np.array("s"), incident_pressure=incident,
            incident_pressure_shape_meaning=np.array("(n_scatterers, n_time_samples)"),
            incident_pressure_row_labels=scat.row_labels(),
            scatterer_xyz=xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]), scatterer_xyz_units=np.array("m"),
            pulse_t=pulse_t, pulse_p=pulse_p, element_x=ex, element_x_units=np.array("m"))

    # ---- runner 3: TI stage 2 ---------------------------------------------------------------
    t0 = time.time()
    reflected = compute_reflected_pressure(scat, incident, t_axis, cfg)
    timings['runner3_ti_stage2'] = time.time() - t0
    _baton(log, "3 TI stage 2  (incident -> reflected)", "Tissue Interaction",
           reflected_pressure_input=reflected)
    say(f"[run 3 ] TI stage 2      : reflected_pressure_input {reflected.shape}, "
        f"path_model={cfg.path_model}   ({time.time()-t0:.1f}s)")
    if outdir:
        files["03_reflected"] = os.path.join(outdir, "03_tissue_interaction_output.npz")
        save_reflected(files["03_reflected"], scat, reflected, t_axis, cfg)

    # ---- runner 4: AP part 2 ----------------------------------------------------------------
    t0 = time.time()
    rx = ap.return_propagation_many(reflected, t_axis, xyz[:, 0], xyz[:, 2], ex, f0=rc.f0_hz, c=rc.c0,
                                    alpha0=alpha_ap)
    timings['runner4_ap_part2'] = time.time() - t0
    _baton(log, "4 AP part 2   (scatterers -> RX array)", "Acoustic Propagation", received_pressure=rx)
    say(f"[run 4 ] AP part 2       : received_pressure {rx.shape}  -> Receiver BF   ({time.time()-t0:.1f}s)")
    if outdir:
        files["04_part2"] = os.path.join(outdir, "04_part2_return_propagation_outputs.npz")
        meta = dict(
            t_axis=t_axis, t_axis_units=np.array("s"), sampling_rate_hz=np.array(rc.fs_hz),
            received_pressure=rx.astype(np.float32),
            received_pressure_shape_meaning=np.array(f"({rc.n_elements} rx_elements, n_time_samples)"),
            received_pressure_units=np.array("arbitrary pressure units (a.u.)"),
            received_pressure_row_labels=np.array([f"rx_element_{m:03d}" for m in range(rc.n_elements)]),
            element_x=ex, element_z=np.zeros_like(ex), element_x_units=np.array("m"),
            f0_hz=np.array(rc.f0_hz), c0_nominal_m_s=np.array(rc.c0), pitch_m=np.array(rc.pitch_m),
            n_elements=np.array(rc.n_elements), attenuation_owner=np.array(rc.attenuation_owner),
            scatterer_xyz=xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]), scatterer_xyz_units=np.array("m"))
        np.savez_compressed(files["04_part2"],
                            description=np.array("PART 2 - return_propagation(): FINAL module output for the Receiver "
                                                 "Beamforming team. Time zero = start of transmission."),
                            reflected_pressure_input=reflected,
                            reflected_pressure_input_row_labels=scat.row_labels(), **meta)
        files["receiver_bf_input"] = os.path.join(outdir, "receiver_bf_input.npz")
        np.savez_compressed(files["receiver_bf_input"],
                            description=np.array("INPUT FOR RECEIVER BEAMFORMING: pressure at every RX element vs time "
                                                 "(from AP part 2, driven by the real Tissue Interaction output)."),
                            **meta)

    # ---- optional: TI stage 3 (tissue-aware delays, spreading, directivity) -------------------
    ch = ch_t = None
    if rc.also_ti_channel:
        t0 = time.time()
        ch, ch_t, _ = compute_channel_data(scat, reflected, t_axis, cfg, element_xz=np.stack([ex, np.zeros_like(ex)], 1))
        timings['extra_ti_stage3'] = time.time() - t0
        say(f"[extra ] TI stage 3      : channel data {ch.shape} (alternative path, tissue-aware c)   ({time.time()-t0:.1f}s)")
        if outdir:
            files["alt_ti_stage3"] = os.path.join(outdir, "alt_ti_stage3_channel_data.npz")
            np.savez_compressed(files["alt_ti_stage3"], received_pressure=ch, t_axis=ch_t, element_x=ex,
                                description=np.array("ALTERNATIVE echo data from TI stage 3 (not the main baton)."))
    return RelayResult(cfg, rc, ph, scat, ex, t_axis, incident, reflected, rx, pulse_t, pulse_p, ch, ch_t, log, files, timings)
