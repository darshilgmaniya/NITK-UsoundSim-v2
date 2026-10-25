# NITK Ultrasound Simulation – Transmit Beamforming (128 Elements)

This module is the **Transmit Beamforming team** block for the NITK ultrasound simulation.
It is designed to sit between the Transducer block and the Acoustic Propagation block.

## Pipeline

Transducer parameters → 128-element linear array → TX delays + apodization → Acoustic Propagation

## Philips L12-4 reference
The Philips L12-4 page specifies a **128-element linear array** and a **4–12 MHz frequency range**.
This project therefore uses 128 elements and keeps the operating frequency editable.

Important: Philips does not publish every geometric parameter needed for a numerical simulation on the product page. Therefore values such as element pitch and the exact simulated centre frequency are exposed in `config/transducer_interface.json` and are explicitly editable assumptions.

## Default simulation assumptions
- Number of elements: 128
- Element pitch: 0.30 mm (editable simulation assumption)
- Centre frequency: 8.0 MHz (editable; inside Philips 4–12 MHz range)
- Sound speed: 1540 m/s (editable; should ultimately be taken from the Acoustic Propagation team's model)
- Focus: x = 0 mm, z = 40 mm
- Steering angle: 0°
- Apodization: Hanning
- Pulse cycles: 2

## What your team sends to Acoustic Propagation
The main hand-off file is:

`outputs/tx_beamforming_package.npz`

It contains:
- `element_x_m`: shape `(128,)`
- `tx_delays_s`: shape `(128,)`
- `tx_weights`: shape `(128,)`
- `tx_frequency_hz`: scalar
- `sound_speed_m_s`: scalar
- `focus_x_m`, `focus_z_m`, `steering_angle_deg`: scalars

A JSON version is also written for easy inspection.

## If the Transducer team changes something
Edit only:

`config/transducer_interface.json`

Typical fields they may provide/change:
- `n_elements`
- `pitch_m`
- `frequency_min_hz`
- `frequency_max_hz`
- `center_frequency_hz`
- `element_width_m` (optional metadata)
- `element_height_m` (optional metadata)
- `material`

The beamforming code validates the centre frequency against the supplied operating range.

## Run
From this folder:

```bash
python run_transmit_beamforming.py
```

Outputs are placed in `outputs/`.

## Acoustic Propagation integration
Use:

```python
from src.acoustic_handoff import load_tx_package

tx = load_tx_package("outputs/tx_beamforming_package.npz")

# Pass these into the acoustic propagation model:
element_x = tx["element_x_m"]
tx_delays = tx["tx_delays_s"]
tx_weights = tx["tx_weights"]
```

The key interface is `tx_delays_s`: one delay per transmit element.

## Important integration rule
All teams must use the same:
- number of elements
- element ordering
- element positions/pitch
- sampling/frequency assumptions where applicable
- speed of sound
- coordinate convention

Do not silently change 128 to 64 in another module.
