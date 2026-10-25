import numpy as np
import matplotlib.pyplot as plt

# Load transmit beamforming output
data = np.load("outputs/tx_beamforming_package.npz")

element_x = data["element_x_m"]
tx_delays = data["tx_delays_s"]
tx_weights = data["tx_weights"]

# Convert units for easy plotting
element_number = np.arange(1, len(element_x) + 1)
element_x_mm = element_x * 1000
delays_us = tx_delays * 1e6

# -------------------------------------------------
# Figure 1: Transmit delay for all 128 elements
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(element_number, delays_us, marker="o", markersize=3)

plt.xlabel("Transducer Element Number")
plt.ylabel("Transmit Delay (microseconds)")
plt.title("Transmit Delay for 128-Element Array")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Figure 2: Element positions and delay
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(element_x_mm, delays_us)

plt.xlabel("Element Position (mm)")
plt.ylabel("Transmit Delay (microseconds)")
plt.title("Element Position vs Transmit Delay")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Print important information
# -------------------------------------------------

print("\n--- TRANSMIT BEAMFORMING SUMMARY ---")
print(f"Number of elements : {len(element_x)}")
print(f"Frequency          : {data['tx_frequency_hz'] / 1e6:.2f} MHz")
print(f"Focus X            : {data['focus_x_m'] * 1000:.2f} mm")
print(f"Focus Z            : {data['focus_z_m'] * 1000:.2f} mm")
print(f"Sound speed        : {data['sound_speed_m_s']:.0f} m/s")
print(f"Steering angle     : {data['steering_angle_deg']:.1f} degrees")
print(f"Minimum delay      : {delays_us.min():.4f} us")
print(f"Maximum delay      : {delays_us.max():.4f} us")

print("\nTransmit beamforming visualization completed.")