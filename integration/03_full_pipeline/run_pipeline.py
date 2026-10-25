"""
run_pipeline.py - ONE command: any phantom in, final despeckled B-mode image out

    python run_pipeline.py                                         # asks what to run (menu, incl. "upload my file")
    python run_pipeline.py --upload                                # file dialog: pick your phantom (+ mapping .json) now
    python run_pipeline.py --source points                         # 9 wire targets, a few seconds
    python run_pipeline.py --source abdomen --max-scatterers 12000 # a built-in phantom preset
    python run_pipeline.py --source my_points.csv                  # x_mm,z_mm,amp   or   x_mm,z_mm,tissue
    python run_pipeline.py --source phantom.png --mapping map.json --spacing-mm 0.2   # YOUR image phantom
    python run_pipeline.py --source phantom.png --mapping map.json --f0-mhz 8 --pitch-mm 0.30   # other probe

Probe (defaults come from teams/T3_Acoustic_Propagation config: 128 elements, 5 MHz, pitch = lambda/2):
    --n-elements 128   --f0-mhz 5   --pitch-mm 0.154

Receiver options : --apod hann|hamming|blackman|tukey|rect  --f-number 1.0  --line-spacing-mm  --n-lines
                   --xrange XMIN XMAX  --zrange ZMIN ZMAX  --no-focus
Image options    : --dr 50   --despeckle guided|none   --guided-r 4   --guided-eps 0.001   --gamma 1.0
Outputs go to --out (default outputs/<source>): the baton files of integrations 01 and 02, 06_recon_rf.npz, 07_bmode_outputs.npz,
08_final_image.npz, final_image.npy, final_image.png (just the picture) and pipeline_overview.png.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import numpy as np

from usoundsim_relay import RelayConfig
from receiver_beamforming import RxConfig, APODIZATIONS
from usoundsim_full import demo_points
from usoundsim_pipeline import PostConfig, ReconConfig, pipeline_figure, run_pipeline
from usoundsim_pipeline.upload import IMAGE_EXT, ask_interactively, pick_files_dialog, resolve_inputs, sample_phantom


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="points | sample_abdomen | sample_thyroid | any phantom preset | points file (.csv/.npy) | image (.png/.jpg). Leave out to be asked.")
    p.add_argument("--upload", action="store_true", help="choose the phantom (and its mapping .json) in a file dialog while the code runs")
    p.add_argument("--mapping", help="mapping .json for an image phantom")
    p.add_argument("--spacing-mm", type=float, help="scatterer grid spacing for image / CSV-with-tissue phantoms")
    p.add_argument("--background", default="generic_soft")
    p.add_argument("--max-scatterers", type=int, default=8000)
    p.add_argument("--attenuation-owner", choices=["tissue", "ap"], default="tissue")
    p.add_argument("--n-elements", type=int, default=None)
    p.add_argument("--f0-mhz", type=float, default=None, help="probe centre frequency [MHz]")
    p.add_argument("--pitch-mm", type=float, default=None, help="element pitch [mm]")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fit-aperture", action="store_true",
                   help="image phantoms only: choose the pixel size so the phantom is 95%% as wide as the array "
                        "(otherwise a phantom wider than the array is clipped to the aperture)")
    p.add_argument("--with-ti-channel", action="store_true",
                   help="also compute the alternative TI stage-3 channel data (comparison only, adds ~25%% time)")
    p.add_argument("--apod", choices=APODIZATIONS, default="hann")
    p.add_argument("--f-number", type=float, default=1.0, help="0 = full aperture")
    p.add_argument("--line-spacing-mm", type=float, default=None)
    p.add_argument("--n-lines", type=int, default=None)
    p.add_argument("--xrange", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="mm")
    p.add_argument("--zrange", type=float, nargs=2, metavar=("ZMIN", "ZMAX"), help="mm")
    p.add_argument("--no-focus", action="store_true")
    p.add_argument("--dr", type=float, default=50.0, help="dynamic range of the B-mode [dB]")
    p.add_argument("--despeckle", choices=["guided", "none"], default="guided")
    p.add_argument("--guided-r", type=int, default=4)
    p.add_argument("--guided-eps", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--no-recon", action="store_true", help="skip T6 Image Reconstruction (DC removal + band-pass)")
    p.add_argument("--out")
    return p


def choose_input(a):
    """Fill a.source / a.mapping from --upload, from the menu (no --source), or from a sample name."""
    chosen = None
    if a.upload:
        chosen = resolve_inputs(pick_files_dialog())
    elif a.source is None:
        chosen = ask_interactively()
    elif a.source in ("sample_abdomen", "sample_thyroid"):
        chosen = sample_phantom(a.source)
    elif a.source.lower().endswith(".npy"):             # (N, 3) array x_mm, z_mm, amp
        chosen = resolve_inputs([a.source])
    if chosen is None:
        return
    if isinstance(chosen, str):                       # preset name or "points"
        a.source = chosen
        return
    print(f"[input ] using {chosen}")
    a.source, a.mapping = chosen.source, chosen.mapping or a.mapping
    if chosen.is_image and a.spacing_mm is None:
        a.fit_aperture = True                          # fit the phantom to the probe aperture unless a pixel size is given


def main(argv=None):
    a = build_parser().parse_args(argv)
    choose_input(a)
    rx = RxConfig(apodization=a.apod, f_number=a.f_number or None, n_lines=a.n_lines,
                  line_spacing_m=a.line_spacing_mm * 1e-3 if a.line_spacing_mm else None,
                  x_range_m=tuple(v * 1e-3 for v in a.xrange) if a.xrange else None,
                  z_range_m=tuple(v * 1e-3 for v in a.zrange) if a.zrange else None,
                  dynamic_range_db=a.dr, apply_focus=not a.no_focus)
    kw = {}
    if a.n_elements: kw["n_elements"] = a.n_elements
    if a.f0_mhz: kw["f0_hz"] = a.f0_mhz * 1e6
    if a.pitch_mm: kw["pitch_m"] = a.pitch_mm * 1e-3
    if a.seed is not None: kw["seed"] = a.seed
    rc = RelayConfig(max_scatterers=a.max_scatterers, attenuation_owner=a.attenuation_owner,
                     also_ti_channel=a.with_ti_channel, **kw)
    spacing_mm = a.spacing_mm
    if a.fit_aperture:
        if os.path.splitext(a.source)[1].lower() not in IMAGE_EXT:
            sys.exit("--fit-aperture works for image phantoms only")
        from PIL import Image
        width_px = Image.open(a.source).size[0]
        aperture_mm = (rc.n_elements - 1) * rc.pitch_m * 1e3
        spacing_mm = 0.95 * aperture_mm / width_px
        print(f"[setup ] --fit-aperture: image {width_px} px wide, aperture {aperture_mm:.1f} mm -> pixel size {spacing_mm:.4f} mm")
    post = PostConfig(dynamic_range_db=a.dr, despeckle=a.despeckle, guided_r=a.guided_r,
                      guided_eps=a.guided_eps, gamma=a.gamma)
    src = demo_points() if a.source == "points" else a.source
    out = a.out or os.path.join(HERE, "outputs", os.path.splitext(os.path.basename(a.source))[0])
    res = run_pipeline(src, rc, rx, post, ReconConfig(enabled=not a.no_recon), mapping=a.mapping, spacing_mm=spacing_mm,
                       background=a.background, outdir=out)
    np.save(os.path.join(out, "final_image.npy"), res.final)
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(out, "final_image.png"), res.final, cmap="gray", vmin=0, vmax=1)
    png = os.path.join(out, "pipeline_overview.png")
    pipeline_figure(res, f"NITK-UsoundSim pipeline | source = {a.source}", png)
    print(f"\nDONE.\n  final image (picture) : {os.path.join(out, 'final_image.png')}"
          f"\n  final image (array)   : {os.path.join(out, 'final_image.npy')}"
          f"\n  overview picture      : {png}")
    return res


if __name__ == "__main__":
    main()
