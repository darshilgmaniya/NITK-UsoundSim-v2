"""
liver_echo_simulation.py

Simplified liver tissue-scattering simulator for NITK-UsoundSim.
Generates synthetic raw channel RF echoes and passes them to the existing
receive_beamforming.das_beamform() function.

This is a simplified single-scattering model, NOT a full-wave acoustic solver
and NOT clinical data.

Requirements:
    pip install numpy matplotlib

Keep this file in the same folder as receive_beamforming.py.
Run:
    python liver_echo_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

try:
    from receive_beamforming import das_beamform
except ImportError as exc:
    raise ImportError(
        "Could not import das_beamform. Put liver_echo_simulation.py and "
        "receive_beamforming.py in the same folder."
    ) from exc


@dataclass
class LiverConfig:
    SPEED_OF_SOUND: float = 1540.0
    CENTER_FREQUENCY: float = 5.0e6
    SAMPLING_FREQUENCY: float = 40.0e6
    N_ELEMENTS: int = 48
    N_LINES: int = 48
    ELEMENT_PITCH: float = 0.30e-3
    MAX_DEPTH: float = 0.060
    START_TIME: float = -1.175e-6
    RX_APODIZATION: str = "hann"
    RX_F_NUMBER: float = 1.0

    def get_t_axis(self):
        n = int(np.ceil((2 * self.MAX_DEPTH / self.SPEED_OF_SOUND
                         - self.START_TIME) * self.SAMPLING_FREQUENCY))
        return self.START_TIME + np.arange(n) / self.SAMPLING_FREQUENCY

    def get_depth_axis(self):
        # Standard pulse-echo depth mapping z = c*t/2, sampled at dz=c/(2*fs).
        dz = self.SPEED_OF_SOUND / (2 * self.SAMPLING_FREQUENCY)
        return np.arange(0.0, self.MAX_DEPTH, dz)


def make_liver_phantom(seed=7, max_depth=0.060):
    """Create scatterers: (x, z, amplitude, kind)."""
    rng = np.random.default_rng(seed)
    scatterers = []

    # Diffuse liver parenchyma scatterers, distributed throughout the field.
    n_diffuse = 2600
    xs = rng.uniform(-0.012, 0.012, n_diffuse)
    zs = rng.uniform(0.004, max_depth - 0.002, n_diffuse)
    amps = rng.normal(0.0, 0.65, n_diffuse)
    for x, z, a in zip(xs, zs, amps):
        scatterers.append((x, z, a, 0))

    # Stronger echoes from two simplified tissue interfaces.
    for z_wall, count, strength in [(0.010, 220, 0.85), (0.028, 180, 0.55)]:
        xs = rng.uniform(-0.011, 0.011, count)
        zs = z_wall + rng.normal(0, 0.00018, count)
        amps = rng.normal(0, strength, count)
        for x, z, a in zip(xs, zs, amps):
            scatterers.append((x, z, a, 1))

    # A few vessel-like anechoic regions. Reduce diffuse scatterer amplitudes
    # inside the ellipses, then add bright, thin vessel-wall boundaries.
    vessels = [
        (-0.0040, 0.023, 0.0015, 0.0030),
        ( 0.0045, 0.041, 0.0020, 0.0040),
    ]
    filtered = []
    for x, z, a, kind in scatterers:
        inside = any(((x - vx) / rx) ** 2 + ((z - vz) / rz) ** 2 < 1
                     for vx, vz, rx, rz in vessels)
        if not (inside and kind == 0):
            filtered.append((x, z, a, kind))
    scatterers = filtered

    # Vessel wall scatterers around each ellipse.
    for vx, vz, rx, rz in vessels:
        theta = np.linspace(0, 2 * np.pi, 150, endpoint=False)
        for t in theta:
            x = vx + rx * np.cos(t)
            z = vz + rz * np.sin(t)
            scatterers.append((x, z, rng.normal(0, 1.0), 2))

    return np.asarray([s[:3] for s in scatterers], dtype=float), vessels


def pulse_waveform(time_offsets, fc):
    """Short Gaussian-windowed sinusoidal RF pulse."""
    sigma = 0.22e-6
    return np.sin(2 * np.pi * fc * time_offsets) * np.exp(
        -0.5 * (time_offsets / sigma) ** 2
    )


def simulate_raw_rf(config, scatterers, x_elements, x_lines, seed=9):
    """
    Generate raw RF with shape (n_elements, n_samples, n_lines).

    The transmit travel time is approximated by the shortest path from the
    array to the scatterer. Receive travel time is computed per element.
    A Gaussian transmit/receive lateral sensitivity reduces off-axis echoes.
    """
    rng = np.random.default_rng(seed)
    c = config.SPEED_OF_SOUND
    fs = config.SAMPLING_FREQUENCY
    fc = config.CENTER_FREQUENCY
    t_axis = config.get_t_axis()
    n_samples = t_axis.size
    n_el = x_elements.size
    n_lines = x_lines.size
    rf = np.zeros((n_el, n_samples, n_lines), dtype=np.float32)

    # Use a finite pulse support to avoid unnecessary computation.
    half_support = int(np.ceil(0.55e-6 * fs))
    pulse_offsets = np.arange(-half_support, half_support + 1) / fs
    pulse = pulse_waveform(pulse_offsets, fc)
    pulse_norm = np.max(np.abs(pulse)) + 1e-12
    pulse = pulse / pulse_norm

    # Small noise floor mimics electronic noise; amplitude is kept modest.
    rf += rng.normal(0.0, 0.0015, rf.shape).astype(np.float32)

    print(f"[T3] Simulating {len(scatterers)} liver scatterers...")
    for line_idx, x_line in enumerate(x_lines):
        if line_idx % 8 == 0:
            print(f"[T3] Scan line {line_idx + 1}/{n_lines}")

        # Approximate a broad, line-centred transmit beam.
        for sx, sz, amp in scatterers:
            # Smooth lateral beam sensitivity, width grows with depth.
            beam_width = max(0.0007, sz / 18.0)
            tx_sensitivity = np.exp(-0.5 * ((sx - x_line) / beam_width) ** 2)
            if tx_sensitivity < 0.015:
                continue

            # Broadcast transmit approximation: closest element's travel time.
            tau_tx = np.min(np.hypot(sx - x_elements, sz)) / c
            lateral_rx = np.abs(x_elements - sx)
            rx_sensitivity = np.exp(-0.5 * (lateral_rx / max(0.001, sz / 2.5)) ** 2)

            # Approximate frequency-dependent attenuation:
            # 0.5 dB/(cm MHz) one-way; apply round-trip attenuation.
            depth_cm = sz * 100.0
            attenuation_db = 2.0 * 0.5 * depth_cm * (fc / 1e6)
            attenuation = 10.0 ** (-attenuation_db / 20.0)
            scatter_amp = amp * tx_sensitivity * attenuation

            for el_idx, x_el in enumerate(x_elements):
                sensitivity = rx_sensitivity[el_idx]
                if sensitivity < 0.01:
                    continue

                tau_rx = np.hypot(sx - x_el, sz) / c
                arrival = tau_tx + tau_rx
                center_idx = int(round((arrival - t_axis[0]) * fs))
                start = center_idx - half_support
                end = center_idx + half_support + 1

                src_start = 0
                src_end = pulse.size
                if start < 0:
                    src_start = -start
                    start = 0
                if end > n_samples:
                    src_end -= end - n_samples
                    end = n_samples
                if start >= end or src_start >= src_end:
                    continue

                # Sub-sample timing is approximated by the nearest sample.
                rf[el_idx, start:end, line_idx] += (
                    scatter_amp * sensitivity * pulse[src_start:src_end]
                ).astype(np.float32)

    return rf, t_axis


def main():
    config = LiverConfig()
    x_elements = (
        np.arange(config.N_ELEMENTS) - (config.N_ELEMENTS - 1) / 2
    ) * config.ELEMENT_PITCH
    x_lines = np.linspace(x_elements[0], x_elements[-1], config.N_LINES)

    scatterers, vessels = make_liver_phantom(max_depth=config.MAX_DEPTH)
    print("[T3] Liver phantom created.")
    print(f"[T3] Elements: {config.N_ELEMENTS}; scan lines: {config.N_LINES}")
    print(f"[T3] Sampling frequency: {config.SAMPLING_FREQUENCY / 1e6:.1f} MHz")
    print("[T3] Generating raw receive RF data...")

    raw_rf, t_axis = simulate_raw_rf(
        config, scatterers, x_elements, x_lines
    )
    print("[T3] Raw RF generation complete.")
    print("Raw RF shape (elements, samples, lines):", raw_rf.shape)
    print("Raw RF peak amplitude:", float(np.max(np.abs(raw_rf))))

    # Run the existing T5 receive beamformer.
    print("[T5] Running receive delay-and-sum beamforming...")
    beamformed_rf, z_axis = das_beamform(
        raw_rf,
        x_elements,
        x_lines,
        config,
        apply_focus=True,
        t_axis=t_axis,
        apodization=config.RX_APODIZATION,
        f_number=config.RX_F_NUMBER,
    )
    print("[T5] Receive beamforming complete.")
    print("Beamformed RF shape (depths, lines):", beamformed_rf.shape)
    print("Beamformed RF peak amplitude:",
          float(np.max(np.abs(beamformed_rf))))
    print("[DONE] Raw RF echoes and beamformed RF are ready.")
    print("[NOTE] B-mode envelope detection and log compression are not included.")

    # Visualize the phantom, one raw RF channel, and beamformed RF (not B-mode).
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Phantom scatterer map.
    ax = axes[0]
    sc = ax.scatter(
        scatterers[:, 0] * 1000,
        scatterers[:, 1] * 1000,
        c=scatterers[:, 2],
        s=2,
        cmap="gray",
        vmin=-1.5,
        vmax=1.5,
    )
    ax.set_title("Simulated liver scatterers")
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_ylim(config.MAX_DEPTH * 1000, 0)
    ax.set_xlim(x_elements[0] * 1000, x_elements[-1] * 1000)

    # Raw RF at the centre element and centre scan line.
    ax = axes[1]
    el_mid = config.N_ELEMENTS // 2
    line_mid = config.N_LINES // 2
    ax.plot(t_axis * 1e6, raw_rf[el_mid, :, line_mid], linewidth=0.6)
    ax.set_title("Raw received RF echo")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)

    # Display signed beamformed RF. This is not a B-mode image.
    ax = axes[2]
    max_val = np.percentile(np.abs(beamformed_rf), 99.5) + 1e-12
    ax.imshow(
        beamformed_rf,
        extent=[
            x_lines[0] * 1000, x_lines[-1] * 1000,
            z_axis[-1] * 1000, z_axis[0] * 1000
        ],
        aspect="auto",
        cmap="gray",
        vmin=-max_val,
        vmax=max_val,
        interpolation="nearest",
    )
    ax.set_title("T5 beamformed RF (not B-mode)")
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Depth (mm)")

    fig.suptitle("NITK-UsoundSim: Liver Echo Simulation and Receive Beamforming")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
