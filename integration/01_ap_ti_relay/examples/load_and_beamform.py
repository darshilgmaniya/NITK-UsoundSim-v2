"""Example for the Receiver Beamforming team: load the hand-off file and form an image.

    python examples/load_and_beamform.py outputs/points/receiver_bf_input.npz

Uses only numpy/scipy/matplotlib. Transmit is unfocused (all elements fire together), so the
transmit time to a pixel is taken as z / c (plane wave); receive is delay-and-sum with dynamic focus.
"""
import sys
import numpy as np
from scipy import signal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/points/receiver_bf_input.npz"
d = np.load(path)
rx = d["received_pressure"].astype(np.float64)            # (128, T)
t = d["t_axis"]; ex = d["element_x"]; fs = float(d["sampling_rate_hz"]); c = float(d["c0_nominal_m_s"])
print(f"{rx.shape[0]} elements, {rx.shape[1]} samples, fs {fs/1e6:.0f} MHz, aperture {(ex[-1]-ex[0])*1e3:.1f} mm")

# 1) analytic signal per channel (envelope detection after summation)
an = signal.hilbert(rx, axis=1)

# 2) pixel grid
xs = np.linspace(-10e-3, 10e-3, 400)
zs = np.linspace(2e-3, 34e-3, 500)
X, Z = np.meshgrid(xs, zs)

# 3) delay-and-sum: sample each channel at t_tx + t_rx (linear interpolation), optional Hann apodisation
apod = np.hanning(ex.size + 2)[1:-1]
t_tot = Z[..., None] / c + np.hypot(X[..., None] - ex, Z[..., None]) / c         # (nz, nx, M)
pos = (t_tot - t[0]) * fs
i0 = np.clip(np.floor(pos).astype(int), 0, rx.shape[1] - 2); fr = pos - i0
m = np.arange(ex.size)
val = an[m, i0] * (1 - fr) + an[m, i0 + 1] * fr
img = np.abs((val * apod).sum(-1))

# 4) log compression and display
db = 20 * np.log10(np.maximum(img / img.max(), 1e-6))
plt.figure(figsize=(5, 7))
plt.imshow(db, extent=[xs[0]*1e3, xs[-1]*1e3, zs[-1]*1e3, zs[0]*1e3], cmap="gray", vmin=-50, vmax=0)
plt.xlabel("x [mm]"); plt.ylabel("z [mm]"); plt.title("delay-and-sum, 128 elements"); plt.colorbar(label="dB")
out = path.replace(".npz", "_beamformed.png"); plt.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)
