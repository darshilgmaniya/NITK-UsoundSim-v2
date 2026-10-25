"""
Reproduces every number in docs/08_VERIFICATION_REPORT.md (section "Integration 02") and writes
docs/verification/results_integration02.json.

    python verification/run_verification.py                 # quick checks, about 1 minute
    python verification/run_verification.py --presets       # also every body-part preset (about 8 min on 1 CPU)
    python verification/run_verification.py --presets generic abdomen
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _paths  # noqa: F401

import numpy as np
from usoundsim_full import demo_points, run_full
from usoundsim_relay import RelayConfig
import receiver_beamforming as rb

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PRESETS = ["generic", "abdomen", "kidney", "thyroid", "breast", "cardiac", "carotid", "msk", "obstetric",
           "brain", "lung", "prostate"]


def fwhm(profile, axis_vals):
    """Full width at -6 dB of a normalised profile around its maximum (linear interpolation)."""
    p = profile / profile.max(); i = int(p.argmax()); half = 0.5
    lo = i
    while lo > 0 and p[lo] > half: lo -= 1
    hi = i
    while hi < len(p) - 1 and p[hi] > half: hi += 1
    if lo == 0 or hi == len(p) - 1:
        return float("nan")
    xl = np.interp(half, [p[lo], p[lo + 1]], [axis_vals[lo], axis_vals[lo + 1]])
    xr = np.interp(half, [p[hi], p[hi - 1]], [axis_vals[hi], axis_vals[hi - 1]])
    return float(xr - xl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="*", default=None, help="run body-part presets (default list if no names)")
    a = ap.parse_args()
    out = {}

    # ---- 1. nine wire targets ------------------------------------------------------------------
    r = run_full("points", verbose=False)
    errs = []
    for x, _, z in r.relay.scatterers.xyz:
        xb, zb = rb.locate_peak(r.bf, x, z)
        errs.append((xb - x, zb - z))
    errs = np.array(errs) * 1e3
    out["wire_targets"] = dict(n=len(errs), max_abs_err_x_mm=float(np.abs(errs[:, 0]).max()),
                               max_abs_err_z_mm=float(np.abs(errs[:, 1]).max()),
                               image_shape=list(r.bf.beamformed_rf.shape), runner5_seconds=r.timings["runner5_receiver_bf"])
    print("wire targets:", out["wire_targets"])

    # ---- 2. resolution vs apodisation (single wire at z = 20 mm, x = 0) ------------------------------
    inp = r.bf.input
    single = run_full(np.array([[0.0, 20.0, 1.0]]), RelayConfig(also_ti_channel=False), units="mm", verbose=False)
    res = {}
    for apod in rb.APODIZATIONS:
        for fnum in (1.0, 2.0):
            b = rb.beamform(single.bf.input, rb.RxConfig(apodization=apod, f_number=fnum, x_range_m=(-3e-3, 3e-3),
                                                         z_range_m=(17e-3, 23e-3), line_spacing_m=0.02e-3))
            iz, ix = np.unravel_index(b.envelope.argmax(), b.envelope.shape)
            lat = fwhm(b.envelope[iz, :], b.x_lines)
            ax_ = fwhm(b.envelope[:, ix], b.z_axis)
            side = b.envelope[iz, :]; side = side / side.max()
            # highest lateral side lobe: local maxima outside the main lobe's first null
            from scipy.signal import find_peaks
            pk, _ = find_peaks(side)
            lobes = sorted([side[k] for k in pk if abs(b.x_lines[k] - b.x_lines[ix]) > 1.2 * lat], reverse=True)
            res[f"{apod}_F{fnum:g}"] = dict(lateral_fwhm_mm=lat * 1e3, axial_fwhm_mm=ax_ * 1e3,
                                          peak_side_lobe_db=float(20 * np.log10(lobes[0])) if lobes else None)
    out["resolution_20mm"] = res
    for k, v in res.items():
        print(f"  {k:14s} lateral FWHM {v['lateral_fwhm_mm']:.3f} mm  axial {v['axial_fwhm_mm']:.3f} mm  PSL {v['peak_side_lobe_db']}")

    # ---- 3. linearity -----------------------------------------------------------------------------
    pts = np.array([[0.0, 20.0, 1.0], [5.0, 30.0, 1.0]])
    a1 = run_full(pts, RelayConfig(also_ti_channel=False), units="mm", verbose=False)
    a2 = run_full(pts * np.array([1, 1, 2]), RelayConfig(also_ti_channel=False), units="mm", verbose=False)
    out["linearity_ratio"] = float(a2.bf.beamformed_rf.max() / a1.bf.beamformed_rf.max())
    print("linearity ratio (expect 2):", out["linearity_ratio"])

    # ---- 4. element counts --------------------------------------------------------------------------
    out["element_counts"] = {}
    for n in (64, 96, 128):
        rr = run_full("points", RelayConfig(n_elements=n, also_ti_channel=False), verbose=False)
        half_ap = float(rr.relay.element_x.max())
        inside = [(x, z) for x, _, z in rr.relay.scatterers.xyz if abs(x) <= half_ap]
        e = np.array([[a - x, b - z] for x, z in inside for a, b in [rb.locate_peak(rr.bf, x, z)]]) * 1e3
        out["element_counts"][n] = dict(aperture_half_width_mm=half_ap * 1e3, targets_inside_aperture=len(inside),
                                        targets_total=int(rr.relay.scatterers.n),
                                        max_abs_err_x_mm=float(np.abs(e[:, 0]).max()), max_abs_err_z_mm=float(np.abs(e[:, 1]).max()),
                                        note=rr.bf.notes[0] if rr.bf.notes else "")
        print(f"  {n:3d} elements: {len(inside)}/9 targets inside the aperture, max |err| x {np.abs(e[:,0]).max():.3f} mm, "
              f"z {np.abs(e[:,1]).max():.3f} mm")

    # ---- 5. hand-off file equals in-memory path ------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rr = run_full("points", RelayConfig(also_ti_channel=False), outdir=d, verbose=False)
        again = rb.beamform(rr.files["receiver_bf_input"])
        out["file_vs_memory_max_abs_diff"] = float(np.abs(again.beamformed_rf - rr.beamformed_rf).max())
    print("file vs memory max |diff|:", out["file_vs_memory_max_abs_diff"])

    # ---- 6. body-part presets (optional, slow) ---------------------------------------------------------
    if a.presets is not None:
        names = a.presets or PRESETS
        out["presets"] = {}
        for p in names:
            t = time.time()
            rr = run_full(p, RelayConfig(max_scatterers=1500, also_ti_channel=False), verbose=False)
            b = rr.bf
            out["presets"][p] = dict(n_scatterers=int(rr.relay.scatterers.n), rx_shape=list(rr.relay.received_pressure.shape),
                                     image_shape=list(b.beamformed_rf.shape), finite=bool(np.isfinite(b.beamformed_rf).all()),
                                     nonblank=bool(b.beamformed_rf.std() > 0), seconds=round(time.time() - t, 1))
            print(f"  preset {p:10s}", out["presets"][p])

    dst = os.path.join(ROOT, "docs", "verification")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "results_integration02.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("written", os.path.join(dst, "results_integration02.json"))


if __name__ == "__main__":
    main()
