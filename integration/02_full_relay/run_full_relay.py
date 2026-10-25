"""
run_full_relay.py - ONE command for the whole chain, phantom to beamformed image

    python run_full_relay.py --source points                       # 9 wire targets, ~3 s
    python run_full_relay.py --source generic --max-scatterers 12000
    python run_full_relay.py --source abdomen --max-scatterers 12000
    python run_full_relay.py --source my_points.csv                # x_mm,z_mm,tissue  or  x_mm,z_mm,amp
    python run_full_relay.py --source phantom.png --mapping map.json --spacing-mm 0.2
    python run_full_relay.py --handoff outputs/points/receiver_bf_input.npz     # runner 5 only

Receiver-BF options:  --apod hann|hamming|blackman|tukey|rect   --f-number 1.5   --line-spacing-mm 0.1
                      --xrange -8 8  --zrange 2 35  --dr 50  --no-focus
Outputs go to --out (default outputs/<source>): the four baton files from integration 01,
05_beamformed_outputs.npz (hand-off to B-mode image formation) and full_relay_overview.png.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")

from usoundsim_relay import RelayConfig
from receiver_beamforming import RxConfig, APODIZATIONS
from usoundsim_full import demo_points, overview_figure, run_from_handoff, run_full


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="points | any phantom preset | points file | image")
    p.add_argument("--handoff", default=None, help="receiver_bf_input.npz: run only the Receiver BF step")
    p.add_argument("--mapping"); p.add_argument("--spacing-mm", type=float)
    p.add_argument("--background", default="generic_soft")
    p.add_argument("--max-scatterers", type=int, default=8000)
    p.add_argument("--attenuation-owner", choices=["tissue", "ap"], default="tissue")
    p.add_argument("--n-elements", type=int, default=None, help="default = config.N_ELEMENTS (128)")
    p.add_argument("--apod", choices=APODIZATIONS, default="hann")
    p.add_argument("--f-number", type=float, default=1.0, help="0 = full aperture")
    p.add_argument("--line-spacing-mm", type=float, default=None)
    p.add_argument("--n-lines", type=int, default=None)
    p.add_argument("--xrange", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="mm")
    p.add_argument("--zrange", type=float, nargs=2, metavar=("ZMIN", "ZMAX"), help="mm")
    p.add_argument("--dr", type=float, default=50.0, help="dynamic range of the preview [dB]")
    p.add_argument("--no-focus", action="store_true", help="switch dynamic receive focusing off")
    p.add_argument("--out")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if bool(a.source) == bool(a.handoff):
        sys.exit("give exactly one of --source or --handoff")
    rx = RxConfig(apodization=a.apod, f_number=a.f_number or None, n_lines=a.n_lines,
                  line_spacing_m=a.line_spacing_mm * 1e-3 if a.line_spacing_mm else None,
                  x_range_m=tuple(v * 1e-3 for v in a.xrange) if a.xrange else None,
                  z_range_m=tuple(v * 1e-3 for v in a.zrange) if a.zrange else None,
                  dynamic_range_db=a.dr, apply_focus=not a.no_focus)
    if a.handoff:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.handoff)))
        res = run_from_handoff(a.handoff, rx, outdir=out)
        title = f"hand-off = {os.path.basename(a.handoff)}"
    else:
        kw = {"n_elements": a.n_elements} if a.n_elements else {}
        rc = RelayConfig(max_scatterers=a.max_scatterers, attenuation_owner=a.attenuation_owner, **kw)
        src = demo_points() if a.source == "points" else a.source
        out = a.out or os.path.join(HERE, "outputs", os.path.splitext(os.path.basename(a.source))[0])
        res = run_full(src, rc, rx, mapping=a.mapping, spacing_mm=a.spacing_mm, background=a.background, outdir=out)
        title = f"source = {a.source}"
    png = os.path.join(out, "full_relay_overview.png")
    overview_figure(res, f"NITK-UsoundSim full relay | {title}", png)
    print(f"\nDONE.\n  beamformed hand-off : {res.files.get('05_beamformed')}\n  overview picture    : {png}")
    return res


if __name__ == "__main__":
    main()
