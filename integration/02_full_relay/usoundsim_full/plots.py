"""plots.py - one-page picture of the whole relay (scatterers -> channel data -> beamformed image)."""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def overview_figure(res, title: str = "", path: str | None = None):
    bf = res.bf
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw=dict(width_ratios=[1.3, 1, 0.75]))
    # 1: where the scatterers are
    if res.relay is not None:
        xyz = res.relay.scatterers.xyz
        amp = np.abs(res.relay.scatterers.amp) if hasattr(res.relay.scatterers, "amp") else np.ones(len(xyz))
        order = np.argsort(amp)
        ax[0].scatter(xyz[order, 0] * 1e3, xyz[order, 2] * 1e3, c=amp[order], s=3, cmap="gray_r")
        ax[0].set_title(f"runner 1: {len(xyz)} scatterers")
    elif bf.input.scatterer_xyz is not None:
        xyz = bf.input.scatterer_xyz
        ax[0].scatter(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3, s=6, c="k")
        ax[0].set_title(f"scatterer truth ({len(xyz)})")
    else:
        ax[0].text(0.5, 0.5, "no scatterer truth\nin hand-off file", ha="center", va="center")
        ax[0].set_title("scatterers")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("z [mm]"); ax[0].invert_yaxis(); ax[0].set_aspect("equal")
    # 2: channel data handed from runner 4 to runner 5
    inp = bf.input
    disp = np.sign(inp.rx) * np.abs(inp.rx) ** 0.4          # compress dynamic range so deep echoes are visible
    lim = np.percentile(np.abs(disp), 99.5) + 1e-30
    ax[1].imshow(disp.T, aspect="auto", cmap="gray", vmin=-lim, vmax=lim,
                 extent=[0, inp.n_elements, inp.t_axis[-1] * 1e6, inp.t_axis[0] * 1e6])
    ax[1].set_title(f"runner 4 -> 5: received_pressure {inp.rx.shape}")
    ax[1].set_xlabel("receive element"); ax[1].set_ylabel("time [us]")
    # 3: result
    im = ax[2].imshow(bf.bmode_db, extent=bf.extent_mm, aspect="equal", cmap="gray",
                      vmin=-bf.cfg.dynamic_range_db, vmax=0)
    ax[2].set_title(f"runner 5: beamformed preview\n({bf.cfg.dynamic_range_db:.0f} dB, inside the array aperture)", fontsize=10)
    ax[2].set_xlabel("x [mm]"); ax[2].set_ylabel("z [mm]")
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="dB")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig
