"""plots.py - the figure shown at the end of a pipeline run."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pipeline_figure(res, title: str, path: str) -> str:
    """Scatterers | beamformed B-mode | scan-converted image | final despeckled image."""
    sc = res.full.relay.scatterers.xyz
    ext = res.extent_mm
    fig, ax = plt.subplots(1, 4, figsize=(17, 6.2))
    ax[0].scatter(sc[:, 0] * 1e3, sc[:, 2] * 1e3, s=2, c="k")   # xyz = (x lateral, 0, z depth)
    ax[0].invert_yaxis(); ax[0].set_aspect("equal"); ax[0].set_title(f"phantom scatterers ({len(sc)})")
    ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("depth z (mm)")
    for a, img, t in ((ax[1], res.bmode, "B-mode (beamformed grid)"),
                      (ax[2], res.image, "scan-converted"),
                      (ax[3], res.final, "final (despeckled)")):
        e = [res.full.bf.x_lines[0] * 1e3, res.full.bf.x_lines[-1] * 1e3, res.full.bf.z_axis[-1] * 1e3,
             res.full.bf.z_axis[0] * 1e3] if img is res.bmode else ext
        a.imshow(img, cmap="gray", vmin=0, vmax=1, extent=e, aspect="equal")
        a.set_title(t); a.set_xlabel("x (mm)"); a.set_ylabel("depth z (mm)")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path
