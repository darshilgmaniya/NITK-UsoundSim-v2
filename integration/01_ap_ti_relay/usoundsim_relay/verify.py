"""Minimal delay-and-sum used ONLY to check the relay (NOT the Receiver Beamforming team's code).
Plane-wave style transmit (all elements fire together, as in AP's default tau_tx = 0)."""
from __future__ import annotations
import numpy as np
from scipy import signal


def das_image(rx, t_axis, element_x, xs, zs, c=1540.0):
    an = signal.hilbert(rx.astype(np.float64), axis=1)
    fs = 1.0 / float(t_axis[1] - t_axis[0])
    X, Z = np.meshgrid(xs, zs)
    t = Z[..., None] / c + np.hypot(X[..., None] - element_x, Z[..., None]) / c
    pos = (t - t_axis[0]) * fs
    n0 = np.clip(np.floor(pos).astype(int), 0, an.shape[1] - 2)
    fr = pos - n0
    e = np.arange(element_x.size)
    v = an[e, n0] * (1 - fr) + an[e, n0 + 1] * fr
    return np.abs(v.sum(-1))


def locate(rx, t_axis, element_x, x0, z0, half_x=1.5e-3, half_z=1.2e-3, step=0.04e-3, c=1540.0):
    xs = np.arange(x0 - half_x, x0 + half_x + step, step)
    zs = np.arange(z0 - half_z, z0 + half_z + step, step)
    img = das_image(rx, t_axis, element_x, xs, zs, c)
    iz, ix = np.unravel_index(img.argmax(), img.shape)
    return xs[ix], zs[iz]
