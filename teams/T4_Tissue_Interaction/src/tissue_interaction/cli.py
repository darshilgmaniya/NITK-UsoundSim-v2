"""Command line interface:  python -m tissue_interaction <command> ...

  list-body-parts | list-tissues | make-phantom | make-points | scatterers | interact | run | from-ap | demo | selftest

INPUT can be a phantom IMAGE/array (--phantom) or POINTS / COORDINATES (--points, CSV/NPZ/NPY).
OUTPUT for Receiver Beamforming: receiver_bf_input.npz  (received_pressure[n_elements, n_samples]).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

from . import (config_for_body_part, list_body_parts, list_tissues, load_phantom, make_phantom)
from .channel_data import compute_channel_data
from .config import TissueInteractionConfig
from .interaction import compute_reflected_pressure
from .outputs import (load_elements, load_incident, load_scatterers, save_receiver_input, save_reflected)
from .pipeline import run, run_from_ap, run_points, run_stage1


def _add_input_args(p):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--phantom", help="phantom image (.png/.jpg/.tif) or .npy/.npz label array")
    g.add_argument("--points", help="phantom as POINTS: CSV/NPZ/NPY (x,z + tissue | amp | properties)")
    p.add_argument("--mode", default="auto", choices=["auto", "labels", "color", "intensity"],
                   help="(image input) how pixel values are interpreted")
    p.add_argument("--mapping", help="(image input) JSON/YAML mapping file")
    p.add_argument("--spacing-mm", type=float, nargs="+", help="(image input) mm per pixel: one value or 'dx dz'")
    p.add_argument("--width-mm", type=float, help="(image input) alternative to spacing: physical width")
    p.add_argument("--flipud", action="store_true")
    p.add_argument("--loose", action="store_true", help="unknown label ids -> generic_soft")
    p.add_argument("--points-mode", default="auto", choices=["auto", "tissue_cloud", "scatterers"])
    p.add_argument("--units", default="mm", choices=["m", "mm", "cm", "um"], help="(points) unit of x/z columns")
    p.add_argument("--bg-tissue", default="generic_soft", help="(points) background / homogeneous path tissue")
    p.add_argument("--grid-mm", type=float, help="(tissue cloud) raster grid spacing, default = point spacing")
    p.add_argument("--fill-mm", type=float, help="(tissue cloud) max distance a point fills, default 2.5 x spacing")


def _add_cfg_args(p):
    p.add_argument("--body-part", choices=list_body_parts(), help="typical probe + frequency + default legend")
    p.add_argument("--config", help="YAML config (overrides body-part defaults)")
    p.add_argument("--f0-mhz", type=float)
    p.add_argument("--fs-mhz", type=float, help="sampling rate of the mock incident / output (use >= 10 x f0)")
    p.add_argument("--geometry", choices=["linear", "sector"])
    p.add_argument("--apex-z-mm", type=float, help="sector apex depth (negative = behind the probe face)")
    p.add_argument("--n-elements", type=int); p.add_argument("--pitch-mm", type=float)
    p.add_argument("--elements", help="receive element positions file (element_x [m], optional element_z)")
    p.add_argument("--max-scatterers", type=int)
    p.add_argument("--path-model", choices=["full", "none"])
    p.add_argument("--seed", type=int)
    p.add_argument("--no-channel-data", action="store_true")
    p.add_argument("--out", default="outputs")


def _cfg(a) -> TissueInteractionConfig:
    cfg = config_for_body_part(a.body_part) if getattr(a, "body_part", None) else TissueInteractionConfig()
    if getattr(a, "config", None):
        cfg = TissueInteractionConfig.from_yaml(a.config)
    if a.f0_mhz: cfg.f0_hz = a.f0_mhz * 1e6
    if getattr(a, "fs_mhz", None): cfg.fs_hz = a.fs_mhz * 1e6
    if a.geometry: cfg.probe.geometry = a.geometry
    if a.apex_z_mm is not None: cfg.probe.apex_z_m = a.apex_z_mm * 1e-3
    if a.n_elements: cfg.probe.n_elements = a.n_elements
    if a.pitch_mm: cfg.probe.pitch_m = a.pitch_mm * 1e-3
    if a.elements:
        ex, ez = load_elements(a.elements)
        cfg.probe.element_x_m, cfg.probe.element_z_m = ex.tolist(), ez.tolist()
    if a.max_scatterers: cfg.max_scatterers = a.max_scatterers
    if a.path_model: cfg.path_model = a.path_model
    if a.seed is not None: cfg.seed = a.seed
    return cfg


def _phantom(a):
    sp = a.spacing_mm
    if sp is not None and len(sp) == 1:
        sp = sp[0]
    return load_phantom(a.phantom, mode=a.mode, mapping=a.mapping, spacing_mm=sp, width_mm=a.width_mm,
                        body_part=a.body_part, flipud=a.flipud, strict=not a.loose)


def selftest(verbose=True) -> bool:
    """Quick physics + interface checks (about 10 s). Returns True when everything passes."""
    import tempfile
    from scipy import signal
    from . import (PhantomBuilder, build_scatterers, emulate_ap_forward,
                   get_tissue, make_point_targets, run_from_ap, run_points)
    from .scatterers import KIND_INTERFACE
    ok_all = True

    def check(name, ok, info=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")

    cfg = TissueInteractionConfig(f0_hz=5e6, scatterers_per_cell=4, specular_floor=1.0)
    b = PhantomBuilder(10, 20, 0.05, "water", "t"); b.layer(10, 20, "bone")
    sc = build_scatterers(b.build(), cfg)
    i = sc.kind == KIND_INTERFACE
    z1, z2 = get_tissue("water").impedance, get_tissue("bone").impedance
    exp = (z2 - z1) / (z2 + z1) * 0.5 / cfg.f_number
    check("Fresnel reflection water->bone", i.any() and np.allclose(sc.amp[i], exp, rtol=1e-3), f"(R={(z2-z1)/(z2+z1):.3f})")
    b = PhantomBuilder(10, 20, 0.05, "liver", "t"); b.layer(10, 20, "air")
    sc = build_scatterers(b.build(), cfg)
    check("soft tissue->gas inverts polarity", np.all(sc.amp[sc.kind == KIND_INTERFACE] < 0))
    b = PhantomBuilder(10, 25, 0.1, "liver", "t"); b.layer(8, 11, "bone")
    sc = build_scatterers(b.build(), cfg)
    check("bone casts an acoustic shadow", sc.path_tx_loss[sc.xyz[:, 2] > 13e-3].max() < 0.7)
    c2 = TissueInteractionConfig(f0_hz=5e6, fs_hz=60e6); c2.probe.geometry = "linear"
    pts = make_point_targets(depths_mm=(20, 40), lateral_mm=(-5.0, 4.0))
    res = run_points(pts * np.array([1e3, 1e3, 1]), c2, units="mm", verbose=False)
    an = signal.hilbert(res.channel_rf.astype(float), axis=1); fs = 1 / (res.t_rx[1] - res.t_rx[0])
    ex, ez = res.element_xz[:, 0], res.element_xz[:, 1]
    worst = 0.0
    for x0, z0, _ in pts:
        xs = np.arange(x0 - 2e-3, x0 + 2e-3, 0.1e-3); zs = np.arange(z0 - 1.5e-3, z0 + 1.5e-3, 0.05e-3)
        X, Z = np.meshgrid(xs, zs)
        t = Z[..., None] / 1540 + np.hypot(X[..., None] - ex, Z[..., None] - ez) / 1540
        pos = (t - res.t_rx[0]) * fs
        n0 = np.clip(np.floor(pos).astype(int), 0, an.shape[1] - 2); fr = pos - n0; e = np.arange(len(ex))
        img = np.abs((an[e, n0] * (1 - fr) + an[e, n0 + 1] * fr).sum(-1))
        iz, ix = np.unravel_index(img.argmax(), img.shape)
        worst = max(worst, abs(xs[ix] - x0), abs(zs[iz] - z0))
    check("delay-and-sum focuses 4 point targets at true position", worst < 0.15e-3, f"(worst error {worst*1e3:.2f} mm)")
    with tempfile.TemporaryDirectory() as d:
        xyz = np.array([[0, 0, 0.02], [0.005, 0, 0.035]])
        emulate_ap_forward(os.path.join(d, "f.npz"), xyz)
        r = run_from_ap(os.path.join(d, "f.npz"), verbose=False)
        from .ap_io import load_ap_forward
        ap = load_ap_forward(os.path.join(d, "f.npz"))
        t_in = ap.t_axis[np.abs(signal.hilbert(ap.incident[0])).argmax()]
        env = np.abs(signal.hilbert(r.channel_rf[32]))
        win = (r.t_rx > t_in) & (r.t_rx < t_in + 20e-6)           # first echo only
        t_rx = r.t_rx[np.where(win)[0][env[win].argmax()]]
        check("AP file: echo = incident arrival + return time", abs((t_rx - t_in) - 0.02 / 1540) < 0.1e-6,
              f"(delta {(t_rx - t_in)*1e6:.2f} us)")
        check("AP file: row order and 64 elements preserved", r.reflected.shape[0] == 2 and r.channel_rf.shape[0] == 64)
    if verbose:
        print("SELFTEST", "PASSED" if ok_all else "FAILED")
    return ok_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tissue_interaction", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-body-parts"); sub.add_parser("list-tissues")
    sub.add_parser("selftest", help="run quick physics / interface checks")

    p = sub.add_parser("make-phantom", help="write a synthetic IMAGE phantom (png + npy + mapping json)")
    p.add_argument("--body-part", required=True, choices=list_body_parts())
    p.add_argument("--spacing-mm", type=float); p.add_argument("--out", default="sample_data")

    p = sub.add_parser("make-points", help="write synthetic POINT inputs (tissue-cloud CSV, or wire targets)")
    p.add_argument("--body-part", choices=list_body_parts())
    p.add_argument("--targets", action="store_true", help="write a point-target scatterer list instead")
    p.add_argument("--spacing-mm", type=float, default=0.5); p.add_argument("--out", default="sample_data")

    p = sub.add_parser("scatterers", help="STAGE 1 only: points/phantom -> scatterer list")
    _add_input_args(p); _add_cfg_args(p)

    p = sub.add_parser("interact", help="STAGE 2+3: scatterers + incident pressure -> reflected + channel data")
    p.add_argument("--scatterers", required=True); p.add_argument("--incident", required=True)
    p.add_argument("--incident-key"); p.add_argument("--path-model", choices=["full", "none"])
    p.add_argument("--elements"); p.add_argument("--no-channel-data", action="store_true")
    p.add_argument("--out", default="outputs")

    p = sub.add_parser("run", help="everything: stage 1+2+3 + grayscale images (mock incident unless --incident)")
    _add_input_args(p); _add_cfg_args(p)
    p.add_argument("--incident"); p.add_argument("--incident-key"); p.add_argument("--no-qa-preview", action="store_true")

    p = sub.add_parser("from-ap", help="MAIN FLOW: Acoustic Propagation forward_propagation() file -> echo signals for Receiver BF")
    p.add_argument("--forward", required=True, help="npz from forward_propagation() (incident_pressure, scatterer_xyz, ...)")
    p.add_argument("--medium-points", help="optional tissue-cloud CSV describing the tissue around the scatterers")
    p.add_argument("--medium-phantom", help="optional phantom image (+ --mapping, --spacing-mm) instead of points")
    p.add_argument("--mapping"); p.add_argument("--spacing-mm", type=float)
    p.add_argument("--body-part", choices=list_body_parts(), help="probe / geometry defaults")
    p.add_argument("--background", default="generic_soft", help="homogeneous medium when no phantom is given")
    p.add_argument("--default-amp", type=float, default=1.0, help="scatterer amplitude if the file has none")
    p.add_argument("--amp-from-tissue", action="store_true", help="amplitude = tissue diffuse reflectivity (needs a phantom)")
    p.add_argument("--c-medium", type=float, help="sound speed used by the AP team (m/s), default 1540 / from file")
    p.add_argument("--f0-mhz", type=float, help="override centre frequency (default: estimated from pulse_p)")
    p.add_argument("--path-model", choices=["full", "none"], help="'none' if AP already applied attenuation/shadowing")
    p.add_argument("--incident-key"); p.add_argument("--t-key"); p.add_argument("--fs-mhz", type=float)
    p.add_argument("--no-channel-data", action="store_true"); p.add_argument("--out", default="outputs")

    p = sub.add_parser("demo", help="run on a built-in synthetic phantom")
    p.add_argument("--body-part", default="abdomen", choices=list_body_parts())
    p.add_argument("--max-scatterers", type=int, default=30000); p.add_argument("--out", default="outputs")
    p.add_argument("--no-qa-preview", action="store_true")

    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", UserWarning)

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "list-body-parts":
        from .body_presets import BODY_PRESETS, PROBE_CLASSES
        for k in list_body_parts():
            b = BODY_PRESETS[k]; pc = PROBE_CLASSES[b.probe]
            print(f"{k:10s} {pc.name:26s} f0={pc.f0_hz/1e6:4.1f} MHz  {b.width_mm:.0f}x{b.depth_mm:.0f} mm  {b.description}")
    elif a.cmd == "list-tissues":
        for t in list_tissues():
            print(f"{t.name:18s} {t.kind:8s} rho={t.density:7.1f} c={t.speed:6.0f} Z={t.impedance_mrayl:6.3f} "
                  f"alpha={t.atten_db_cm_mhz:g}*f^{t.atten_exp:g} bs={t.backscatter_db:g}dB")
    elif a.cmd == "make-phantom":
        import json
        from PIL import Image
        ph = make_phantom(a.body_part, a.spacing_mm)
        os.makedirs(a.out, exist_ok=True)
        base = os.path.join(a.out, f"phantom_{a.body_part}")
        np.save(base + "_labels.npy", ph.index)
        Image.fromarray(ph.index.astype(np.uint8), "L").save(base + "_labels.png")
        json.dump({"spacing_mm": [ph.dx * 1e3, ph.dz * 1e3],
                   "labels": {str(i): t.name for i, t in enumerate(ph.tissue_list)}},
                  open(base + "_mapping.json", "w"), indent=2)
        print("wrote", base + "_labels.png / _labels.npy / _mapping.json")
    elif a.cmd == "make-points":
        from . import make_point_targets, save_points_csv
        os.makedirs(a.out, exist_ok=True)
        if a.targets:
            pts = make_point_targets()
            path = os.path.join(a.out, "point_targets.csv")
            np.savetxt(path, np.c_[pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2]], delimiter=",",
                       header="x_mm,z_mm,amp", comments="", fmt="%.4f")
        else:
            if not a.body_part: raise SystemExit("--body-part or --targets required")
            path = os.path.join(a.out, f"points_{a.body_part}.csv")
            save_points_csv(path, make_phantom(a.body_part, min(a.spacing_mm, 0.25)), a.spacing_mm)
        print("wrote", path)
    elif a.cmd in ("scatterers", "run"):
        cfg = _cfg(a)
        qa = not getattr(a, "no_qa_preview", False)
        inc = t = None
        if getattr(a, "incident", None):
            inc, t = load_incident(a.incident, a.incident_key)
        if a.points:
            if a.cmd == "scatterers":
                from .points_io import load_points, scatterers_from_points, tissue_cloud_to_phantom
                from .pipeline import _save_stage1
                from .scatterers import build_scatterers
                pc = load_points(a.points, units=a.units); print(pc.summary())
                m = pc.mode if a.points_mode == "auto" else a.points_mode
                if m == "tissue_cloud":
                    ph = tissue_cloud_to_phantom(pc, a.grid_mm, a.bg_tissue, a.fill_mm)
                    scat = build_scatterers(ph, cfg)
                else:
                    ph, scat = None, scatterers_from_points(pc, cfg, a.bg_tissue)
                files = _save_stage1(scat, ph, cfg, a.out)
                print(scat.summary(), "->", files["scatterers"])
            else:
                run_points(a.points, cfg, mode=a.points_mode, units=a.units, background=a.bg_tissue,
                           spacing_mm=a.grid_mm, max_fill_mm=a.fill_mm, incident=inc, t_axis=t,
                           outdir=a.out, qa_preview=qa, channel_data=not a.no_channel_data)
        else:
            ph = _phantom(a); print(ph.summary())
            if a.cmd == "scatterers":
                scat, files = run_stage1(ph, cfg, a.out)
                print(scat.summary(), "->", files["scatterers"])
            else:
                run(ph, cfg, inc, t, a.out, qa_preview=qa, channel_data=not a.no_channel_data)
    elif a.cmd == "interact":
        scat, cfg = load_scatterers(a.scatterers)
        if a.path_model: cfg.path_model = a.path_model
        if a.elements:
            ex, ez = load_elements(a.elements)
            cfg.probe.element_x_m, cfg.probe.element_z_m = ex.tolist(), ez.tolist()
        inc, t = load_incident(a.incident, a.incident_key)
        if t is None:
            raise SystemExit("the incident file needs a 't_axis' array (seconds)")
        refl = compute_reflected_pressure(scat, inc, t, cfg)
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "tissue_interaction_output.npz")
        save_reflected(path, scat, refl, t, cfg)
        print("reflected_pressure_input", refl.shape, "->", path)
        if not a.no_channel_data:
            rf, t_rx, exz = compute_channel_data(scat, refl, t, cfg)
            p2 = os.path.join(a.out, "receiver_bf_input.npz")
            save_receiver_input(p2, rf, t_rx, exz, scat, cfg)
            print("received_pressure (n_elements, n_samples)", rf.shape, "-> ", p2)
    elif a.cmd == "from-ap":
        from .ap_io import load_ap_forward
        from .points_io import load_points, tissue_cloud_to_phantom
        ap = load_ap_forward(a.forward, incident_key=a.incident_key, t_key=a.t_key,
                             fs_hz=a.fs_mhz * 1e6 if a.fs_mhz else None)
        cfg = config_for_body_part(a.body_part) if a.body_part else TissueInteractionConfig()
        if a.path_model: cfg.path_model = a.path_model
        ph = None
        if a.medium_points:
            ph = tissue_cloud_to_phantom(load_points(a.medium_points), background=a.background)
        elif a.medium_phantom:
            ph = load_phantom(a.medium_phantom, mapping=a.mapping, spacing_mm=a.spacing_mm, body_part=a.body_part)
        if a.f0_mhz:
            cfg.f0_hz = a.f0_mhz * 1e6
        res = run_from_ap(ap, cfg, phantom=ph, background=a.background, amp_from_tissue=a.amp_from_tissue,
                          default_amp=a.default_amp, c_medium=a.c_medium, f0_from_pulse=a.f0_mhz is None,
                          outdir=a.out, channel_data=not a.no_channel_data)
    elif a.cmd == "demo":
        cfg = config_for_body_part(a.body_part, max_scatterers=a.max_scatterers)
        ph = make_phantom(a.body_part); print(ph.summary())
        run(ph, cfg, outdir=os.path.join(a.out, a.body_part), qa_preview=not a.no_qa_preview)


if __name__ == "__main__":
    sys.exit(main())
