"""Sanity check of the channel data with a MINIMAL delay-and-sum (NOT the Receiver BF team's code).

If targets focus at their true coordinates, delays / geometry / time axis of the module are consistent.
    python examples/verify_with_das.py
"""
import numpy as np
from scipy import signal
import tissue_interaction as ti


def das_grid(res, xs, zs, c=1540.0):
    """Plane-wave transmit (mock incident) + dynamic receive focusing on a pixel grid (vectorised)."""
    an = signal.hilbert(res.channel_rf.astype(np.float64), axis=1)
    fs = 1.0 / (res.t_rx[1] - res.t_rx[0])
    ex, ez = res.element_xz[:, 0], res.element_xz[:, 1]
    X, Z = np.meshgrid(xs, zs)
    t = Z[..., None] / c + np.hypot(X[..., None] - ex, Z[..., None] - ez) / c      # (nz, nx, n_el)
    pos = (t - res.t_rx[0]) * fs
    n0 = np.clip(np.floor(pos).astype(int), 0, an.shape[1] - 2)
    fr = pos - n0
    e = np.arange(len(ex))
    v = an[e, n0] * (1 - fr) + an[e, n0 + 1] * fr
    return np.abs(v.sum(-1))


def locate(res, x0, z0, half_x=2.5e-3, half_z=1.5e-3, step=0.05e-3):
    xs = np.arange(x0 - half_x, x0 + half_x + step, step)
    zs = np.arange(z0 - half_z, z0 + half_z + step, step)
    img = das_grid(res, xs, zs)
    iz, ix = np.unravel_index(img.argmax(), img.shape)
    return xs[ix], zs[iz], img


if __name__ == "__main__":
    cfg = ti.config_for_body_part("generic")
    cfg.f0_hz, cfg.fs_hz = 5e6, 60e6
    pts = ti.make_point_targets(depths_mm=(20, 35, 50), lateral_mm=(-6.0, 0.0, 6.0))
    res = ti.run_points(pts * np.array([1e3, 1e3, 1]), cfg, units="mm", verbose=False)
    for x, z, _ in pts:
        xb, zb, _ = locate(res, x, z)
        print(f"target ({x*1e3:5.1f},{z*1e3:5.1f}) mm -> focus ({xb*1e3:5.2f},{zb*1e3:5.2f}) mm  "
              f"error ({(xb-x)*1e3:+.2f},{(zb-z)*1e3:+.2f}) mm")
