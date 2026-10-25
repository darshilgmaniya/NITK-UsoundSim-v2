"""
run_relay.py - one command for the whole relay (AP forward -> TI -> AP return -> Receiver BF input)

    python run_relay.py --source generic                      # built-in phantom (any TI preset name)
    python run_relay.py --source abdomen --max-scatterers 12000
    python run_relay.py --source my_points.csv                # x_mm,z_mm,tissue  or  x_mm,z_mm,amp
    python run_relay.py --source phantom.png --mapping map.json --spacing-mm 0.2
    python run_relay.py --source points --out outputs/points  # built-in wire-target demo
Outputs go to --out (default outputs/<source>): the four baton files, receiver_bf_input.npz, relay_overview.png.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths  # noqa: F401  (adds teams/T3_* and teams/T4_* to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from usoundsim_relay import RelayConfig, run_relay
from usoundsim_relay.verify import das_image


def demo_points():
    return np.array([[x, z, 1.0] for z in (10, 20, 30) for x in (-6.0, 0.0, 6.0)])   # x_mm, z_mm, amp


def overview_png(res, path, title):
    t_us = res.t_axis * 1e6
    sc = res.scatterers
    env_i = np.abs(res.incident).sum(1)
    k = int(np.argsort(env_i)[len(env_i) // 2]) if sc.n > 1 else 0
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    a.scatter(sc.xyz[:, 0] * 1e3, sc.xyz[:, 2] * 1e3, s=2, c=np.abs(sc.amp), cmap="magma", vmax=np.percentile(np.abs(sc.amp), 99))
    a.plot(res.element_x * 1e3, np.zeros_like(res.element_x), "c|", ms=6)
    a.invert_yaxis(); a.set_aspect("equal"); a.set_title(f"1 TI stage 1: {sc.n} scatterers", fontsize=10)
    a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
    a = ax[0, 1]
    a.plot(t_us, res.incident[k], lw=.8); a.set_title(f"2 AP part 1: incident pressure, scatterer {k}", fontsize=10)
    a.set_xlabel("time [us]"); a.set_xlim(t_us[np.abs(res.incident[k]) > 1e-3 * np.abs(res.incident[k]).max()][[0, -1]] + [-2, 2])
    a = ax[0, 2]
    a.plot(t_us, res.reflected[k], lw=.8, c="tab:red"); a.set_title(f"3 TI stage 2: reflected pressure, scatterer {k}", fontsize=10)
    a.set_xlabel("time [us]"); a.set_xlim(ax[0, 1].get_xlim())
    a = ax[1, 0]
    rx = res.received_pressure; v = np.percentile(np.abs(rx), 99.5)
    a.imshow(rx.T, aspect="auto", cmap="seismic", vmin=-v, vmax=v, extent=[0, rx.shape[0], t_us[-1], t_us[0]])
    a.set_title("4 AP part 2: received_pressure [128 x time] -> Receiver BF", fontsize=10)
    a.set_xlabel("RX element"); a.set_ylabel("time [us]")
    a = ax[1, 1]
    zmax = min(sc.xyz[:, 2].max() + 3e-3, res.t_axis[-1] * 1540 / 2)
    xs = np.linspace(-max(np.abs(sc.xyz[:, 0]).max(), 8e-3) - 2e-3, max(np.abs(sc.xyz[:, 0]).max(), 8e-3) + 2e-3, 300)
    zs = np.linspace(1e-3, zmax, 300)
    img = das_image(rx, res.t_axis, res.element_x, xs, zs)
    db = 20 * np.log10(np.maximum(img / img.max(), 1e-6))
    a.imshow(db, extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3], cmap="gray", vmin=-50, vmax=0, aspect="equal")
    a.set_title("QA only: quick delay-and-sum (not the BF team's image)", fontsize=10); a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
    a = ax[1, 2]; a.axis("off")
    lines = [title, ""] + [f"{i['runner']}\n    {i['owner']}: " + ", ".join(f"{k} {v}" for k, v in i["shapes"].items()) for i in res.log]
    a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle(f"NITK-UsoundSim relay: AP (128 el) -> TI -> AP -> Receiver BF   |   {title}", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--source", default="generic")
    ap_.add_argument("--mapping"); ap_.add_argument("--spacing-mm", type=float)
    ap_.add_argument("--max-scatterers", type=int, default=8000)
    ap_.add_argument("--attenuation-owner", choices=["tissue", "ap"], default="tissue")
    ap_.add_argument("--background", default="generic_soft")
    ap_.add_argument("--n-elements", type=int, default=None, help="default = config.N_ELEMENTS (128)")
    ap_.add_argument("--out")
    a = ap_.parse_args()
    kw = {}
    if a.n_elements: kw["n_elements"] = a.n_elements
    rc = RelayConfig(max_scatterers=a.max_scatterers, attenuation_owner=a.attenuation_owner, **kw)
    src = demo_points() if a.source == "points" else a.source
    out = a.out or os.path.join(HERE, "outputs", os.path.splitext(os.path.basename(a.source))[0])
    res = run_relay(src, rc, mapping=a.mapping, spacing_mm=a.spacing_mm, background=a.background, outdir=out)
    overview_png(res, os.path.join(out, "relay_overview.png"), f"source = {a.source}")
    print(f"\nDONE. Baton for Receiver BF: {res.files['receiver_bf_input']}\n      overview: {os.path.join(out, 'relay_overview.png')}")


if __name__ == "__main__":
    main()
