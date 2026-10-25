"""Stand-alone use of module 03 - no other module needed.  Builds synthetic channel data for three point
targets (plane-wave transmit), beamforms it and saves a picture.

    python examples/run_standalone.py            # writes standalone_demo.png
    python examples/run_standalone.py my_handoff.npz   # beamform a real hand-off file instead
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import receiver_beamforming as rb

if len(sys.argv) > 1:
    res = rb.beamform(sys.argv[1])
else:
    c, fs, f0, n = 1540.0, 40e6, 5e6, 128
    ex = (np.arange(n) - (n - 1) / 2) * c / f0 / 2
    t = np.arange(-1.2e-6, 50e-6, 1 / fs)
    rx = np.zeros((n, t.size))
    for x, z in [(-3e-3, 12e-3), (0.0, 22e-3), (3e-3, 32e-3)]:
        tau = z / c + np.hypot(x - ex, z) / c
        s = 3 / (2.5 * f0)
        d = t[None, :] - tau[:, None]
        rx += np.sin(2 * np.pi * f0 * d) * np.exp(-0.5 * (d / s) ** 2)
    res = rb.beamform(rb.make_receiver_input(rx, t, ex, fs_hz=fs, c0=c, f0_hz=f0), rb.RxConfig())
plt.figure(figsize=(5, 7))
plt.imshow(res.bmode_db, extent=res.extent_mm, cmap="gray", vmin=-50, vmax=0, aspect="auto")
plt.xlabel("x [mm]"); plt.ylabel("z [mm]"); plt.title("beamformed preview"); plt.colorbar(label="dB")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standalone_demo.png")
plt.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)
