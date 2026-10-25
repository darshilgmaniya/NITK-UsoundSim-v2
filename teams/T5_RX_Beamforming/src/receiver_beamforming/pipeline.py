"""
pipeline.py - the Receiver Beamforming runner (runner 5 of the relay)
====================================================================
    received_pressure (n_el, T)  --->  beamformed_rf (n_depth, n_lines)  [+ envelope / dB preview]

``beamform(input, RxConfig())`` is the one call every user needs.  It
  1. validates the hand-off and reads array / medium parameters from it,
  2. chooses a field of view automatically (from scatterer truth if present, else from the aperture and
     the record length) unless the user gives x/z ranges,
  3. runs ``das_beamform`` (or the user's own function via ``RxConfig.beamformer``),
  4. makes the preview envelope and log-compressed image.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .beamformer import das_beamform
from .bmode_preview import envelope, log_compress, to_uint8
from .config import BeamformConfig, RxConfig
from .io import ReceiverInput, load_receiver_input


@dataclass
class BeamformResult:
    input: ReceiverInput
    cfg: RxConfig
    beamformed_rf: np.ndarray        # (n_depth, n_lines)
    z_axis: np.ndarray               # (n_depth,)
    x_lines: np.ndarray              # (n_lines,)
    envelope: np.ndarray
    bmode_db: np.ndarray
    image_u8: np.ndarray
    seconds: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def extent_mm(self):
        """imshow extent [left, right, bottom, top] in mm."""
        return [self.x_lines[0] * 1e3, self.x_lines[-1] * 1e3, self.z_axis[-1] * 1e3, self.z_axis[0] * 1e3]


def choose_field_of_view(inp: ReceiverInput, cfg: RxConfig):
    """Return (x_lines, z_min, z_max, notes) for any input."""
    notes = []
    ex = inp.element_x
    ap_lo, ap_hi = float(ex.min()), float(ex.max())
    pitch = inp.pitch_m or (inp.c0 / inp.f0_hz / 2)
    c, t = inp.c0, inp.t_axis
    z_data = 0.5 * c * float(t[-1])                        # the record cannot hold deeper echoes
    # --- lateral ---
    if cfg.x_range_m is not None:
        x_lo, x_hi = map(float, cfg.x_range_m)
    elif inp.scatterer_xyz is not None and len(inp.scatterer_xyz):
        sx = inp.scatterer_xyz[:, 0]
        pad = 0.5e-3
        x_lo, x_hi = max(float(sx.min()) - pad, ap_lo), min(float(sx.max()) + pad, ap_hi)
        if x_hi - x_lo < 4 * pitch:                         # tiny phantom -> keep a usable width
            xm = 0.5 * (x_lo + x_hi); x_lo, x_hi = xm - 2 * pitch, xm + 2 * pitch
    else:
        x_lo, x_hi = ap_lo, ap_hi
        notes.append("x range = full aperture (no scatterer truth in the file)")
    if x_hi <= x_lo:
        raise ValueError(f"x_range must have x_max > x_min, got ({x_lo}, {x_hi})")
    if cfg.x_range_m is None and inp.scatterer_xyz is not None and len(inp.scatterer_xyz):
        sx = inp.scatterer_xyz[:, 0]
        n_out = int(np.sum((sx < ap_lo - 1e-9) | (sx > ap_hi + 1e-9)))
        if n_out:
            notes.append(f"{n_out} of {len(sx)} scatterers lie outside the array's lateral extent "
                         f"({ap_lo*1e3:.1f}..{ap_hi*1e3:.1f} mm) and are not inside the image; "
                         f"set RxConfig(x_range_m=...) to image beyond the aperture edge (one-sided, lower quality)")
    # --- depth ---
    if cfg.z_range_m is not None:
        z_lo, z_hi = map(float, cfg.z_range_m)
    elif inp.scatterer_xyz is not None and len(inp.scatterer_xyz):
        sz = inp.scatterer_xyz[:, 2]
        z_lo, z_hi = 0.0, float(sz.max()) + 3e-3
    else:
        z_lo, z_hi = 0.0, 0.9 * z_data
        notes.append("z range = 90 % of what the record can hold (no scatterer truth in the file)")
    z_hi = min(z_hi, z_data)
    if z_hi <= z_lo:
        raise ValueError(f"z_range must have z_max > z_min and fit the record (max {z_data*1e3:.1f} mm)")
    # --- line grid ---
    if cfg.n_lines is not None:
        n = int(cfg.n_lines)
    else:
        sp = cfg.line_spacing_m or pitch
        n = int(np.floor((x_hi - x_lo) / sp)) + 1
    if n > cfg.max_lines:
        notes.append(f"lines capped at max_lines = {cfg.max_lines} (requested {n})")
        n = cfg.max_lines
    n = max(n, 1)
    x_lines = np.linspace(x_lo, x_hi, n) if n > 1 else np.array([0.5 * (x_lo + x_hi)])
    return x_lines, z_lo, z_hi, notes


def beamform(source, cfg: Optional[RxConfig] = None) -> BeamformResult:
    """High-level entry: ReceiverInput | path | dict  ->  BeamformResult."""
    cfg = cfg or RxConfig()
    cfg.validate()
    inp = load_receiver_input(source)
    x_lines, z_lo, z_hi, notes = choose_field_of_view(inp, cfg)
    bc = BeamformConfig(SPEED_OF_SOUND=inp.c0, CENTER_FREQUENCY=inp.f0_hz, SAMPLING_FREQUENCY=inp.fs_hz,
                        N_ELEMENTS=inp.n_elements, ELEMENT_PITCH=inp.pitch_m or 0.0, MAX_DEPTH=z_hi,
                        START_TIME=float(inp.t_axis[0]), RX_APODIZATION=cfg.apodization,
                        RX_F_NUMBER=cfg.f_number or 0.0, TX_MODE=cfg.tx_mode, TX_DELAYS=cfg.tx_delays)
    fn = cfg.beamformer or das_beamform
    t0 = time.time()
    kw = dict(apply_focus=cfg.apply_focus, t_axis=inp.t_axis, apodization=cfg.apodization,
              f_number=cfg.f_number)
    if cfg.beamformer is None:
        kw.update(tx_mode=cfg.tx_mode, tx_delays=cfg.tx_delays, normalize=cfg.normalize)
    bf, z = fn(inp.rx, inp.element_x, x_lines, bc, **kw)
    sec = time.time() - t0
    bf, z = np.asarray(bf, dtype=float), np.asarray(z, dtype=float)
    if bf.ndim != 2 or bf.shape != (z.size, x_lines.size):
        raise ValueError(f"beamformer returned {bf.shape}; expected ({z.size}, {x_lines.size}) = (depth, lines)")
    keep = z >= z_lo - 1e-12
    bf, z = bf[keep], z[keep]
    env = envelope(bf)
    db = log_compress(env, cfg.dynamic_range_db)
    return BeamformResult(inp, cfg, bf, z, x_lines, env, db, to_uint8(db, cfg.dynamic_range_db), sec, notes)


def locate_peak(result: BeamformResult, x0: float, z0: float, half_x=1.5e-3, half_z=1.2e-3):
    """Position of the brightest envelope sample near (x0, z0) - used to check focusing accuracy."""
    xs, zs = result.x_lines, result.z_axis
    mx = (xs >= x0 - half_x) & (xs <= x0 + half_x)
    mz = (zs >= z0 - half_z) & (zs <= z0 + half_z)
    if not mx.any() or not mz.any():
        raise ValueError("search window lies outside the image")
    sub = result.envelope[np.ix_(mz, mx)]
    iz, ix = np.unravel_index(sub.argmax(), sub.shape)
    return float(xs[mx][ix]), float(zs[mz][iz])
