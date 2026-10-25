"""
full_relay.py - the complete relay race (integration step 2)
============================================================

    runner 1  TISSUE INTERACTION  stage 1   phantom / points      -> scatterer_xyz (N,3)
    runner 2  ACOUSTIC PROP.      part 1    128 TX elements       -> incident_pressure (N,T)
    runner 3  TISSUE INTERACTION  stage 2   incident pressure     -> reflected_pressure_input (N,T)
    runner 4  ACOUSTIC PROP.      part 2    reflected pressure    -> received_pressure (128,T)
    runner 5  RECEIVER BEAMFORMING          received_pressure     -> beamformed_rf (depth, lines)  [new]

Runners 1-4 are *not* re-implemented here: they are ``integration/01_ap_ti_relay`` (``usoundsim_relay``).
This file only adds runner 5 and the bookkeeping that joins it, so nothing upstream can drift.

Three ways in
    run_full(source, RelayConfig(), RxConfig())           full chain from a phantom / points / image
    run_from_handoff("receiver_bf_input.npz", RxConfig()) runner 5 only, from a saved file
    receiver_beamforming.beamform(...)                    runner 5 only, from arrays in memory
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import _paths  # noqa: F401
from usoundsim_relay import RelayConfig, RelayResult, run_relay
import receiver_beamforming as rb


@dataclass
class FullResult:
    relay: Optional[RelayResult]          # None when started from a hand-off file
    bf: rb.BeamformResult
    files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def beamformed_rf(self):
        return self.bf.beamformed_rf

    @property
    def image_db(self):
        return self.bf.bmode_db


def demo_points() -> np.ndarray:
    """Nine wire targets, x_mm, z_mm, amp (the same demo as integration 01)."""
    return np.array([[x, z, 1.0] for z in (10, 20, 30) for x in (-6.0, 0.0, 6.0)])


def _relay_to_receiver_input(res: RelayResult) -> "rb.ReceiverInput":
    """The in-memory baton between runner 4 and runner 5 (same content as receiver_bf_input.npz)."""
    return rb.make_receiver_input(
        res.received_pressure, res.t_axis, res.element_x, fs_hz=res.rc.fs_hz, c0=res.rc.c0,
        f0_hz=res.rc.f0_hz, pitch_m=res.rc.pitch_m, scatterer_xyz=res.scatterers.xyz, source="relay (memory)")


def run_full(source, rc: Optional[RelayConfig] = None, rx: Optional[rb.RxConfig] = None, *,
             mapping=None, spacing_mm=None, background: str = "generic_soft", units: str = "mm",
             outdir: Optional[str] = None, verbose: bool = True, **ti_overrides) -> FullResult:
    """Phantom / points / image -> ... -> beamformed image.  ``source`` accepts everything run_relay does."""
    rc = rc or RelayConfig()
    rx = rx or rb.RxConfig()
    rx.validate()                                        # fail on a bad option BEFORE the slow part
    if isinstance(source, str) and source.strip().lower() == "points":
        source = demo_points()                           # convenience keyword, as in the CLI
    say = print if verbose else (lambda *a, **k: None)
    relay = run_relay(source, rc, mapping=mapping, spacing_mm=spacing_mm, background=background,
                      units=units, outdir=outdir, verbose=verbose, **ti_overrides)
    files = dict(relay.files)
    timings = dict(relay.timings)

    # tx delays set for the relay must also reach the beamformer, otherwise focusing would be wrong
    if rc.tau_tx is not None and rx.tx_delays is None:
        rx = _with(rx, tx_delays=np.asarray(rc.tau_tx, float), tx_mode="nearest")
        say("[run 5 ] relay used per-element TX delays -> beamformer uses tx_mode='nearest' with the same delays")

    t0 = time.time()
    inp = _relay_to_receiver_input(relay)
    bf = rb.beamform(inp, rx)
    timings["runner5_receiver_bf"] = time.time() - t0
    say(f"[run 5 ] Receiver BF     : beamformed_rf {bf.beamformed_rf.shape} (depth, lines), "
        f"apod={rx.apodization}, F#={rx.f_number}   ({timings['runner5_receiver_bf']:.1f}s)")
    for n in bf.notes:
        say(f"[run 5 ]   note: {n}")
    if outdir:
        files["05_beamformed"] = rb.save_beamformed(os.path.join(outdir, "05_beamformed_outputs.npz"), bf)
    return FullResult(relay, bf, files, timings)


def run_from_handoff(path_or_dict, rx: Optional[rb.RxConfig] = None, *, outdir: Optional[str] = None,
                     verbose: bool = True) -> FullResult:
    """Runner 5 alone, starting from ``receiver_bf_input.npz`` written by an earlier run."""
    rx = rx or rb.RxConfig()
    t0 = time.time()
    bf = rb.beamform(path_or_dict, rx)
    timings = {"runner5_receiver_bf": time.time() - t0}
    if verbose:
        print(f"[run 5 ] Receiver BF     : {bf.input.rx.shape} -> beamformed_rf {bf.beamformed_rf.shape}   "
              f"({timings['runner5_receiver_bf']:.1f}s)")
    files = {}
    if outdir:
        files["05_beamformed"] = rb.save_beamformed(os.path.join(outdir, "05_beamformed_outputs.npz"), bf)
    return FullResult(None, bf, files, timings)


def _with(cfg: rb.RxConfig, **kw) -> rb.RxConfig:
    from dataclasses import replace
    return replace(cfg, **kw)
