# Hand-off to the Receiver Beamforming Team

## 1. What you receive
One file per run: `outputs/<run>/receiver_bf_input.npz` (field list in [06_DATA_DICTIONARY.md](06_DATA_DICTIONARY.md)).

| Key | Shape | Meaning |
|---|---|---|
| `received_pressure` | (128, T) | pressure at each receive element, row m = element m |
| `t_axis` | (T,) | seconds; time zero = centre of the transmit pulse |
| `element_x` | (128,) | element x [m]; `element_z` = 0 |
| `sampling_rate_hz`, `f0_hz`, `c0_nominal_m_s`, `pitch_m` | scalars | 40 MHz, 5 MHz, 1540 m/s, 0.154 mm |
| `scatterer_xyz` | (N,3) | true positions, **for validation only** |

## 2. What the data is and is not
| It is | It is not |
|---|---|
| noise-free RF-like pressure per element, 25 ns sampling | electronic noise, ADC quantisation, or probe bandwidth other than the 3-cycle pulse |
| the sum over all scatterers of delayed, attenuated echoes | corrected for element directivity or 1/r spreading (the AP model has neither; see limitations) |
| transmit unfocused, all elements firing with zero delay | focused or steered; use `RelayConfig.tau_tx` to change (tell us the delays you need) |
| in arbitrary pressure units | calibrated pascals |

## 3. Load and beamform (tested example)
`integration/01_ap_ti_relay/examples/load_and_beamform.py`:
```bash
python examples/load_and_beamform.py outputs/points/receiver_bf_input.npz
```
Core of the example (delay-and-sum with dynamic receive focus and Hann apodisation):
```python
d  = np.load("receiver_bf_input.npz")
rx = d["received_pressure"].astype(float); t = d["t_axis"]; ex = d["element_x"]
fs = float(d["sampling_rate_hz"]); c = float(d["c0_nominal_m_s"])
an = signal.hilbert(rx, axis=1)
t_tot = Z[..., None]/c + np.hypot(X[..., None] - ex, Z[..., None])/c     # transmit z/c + receive distance/c
pos = (t_tot - t[0]) * fs; i0 = np.floor(pos).astype(int); fr = pos - i0
val = an[m, i0]*(1-fr) + an[m, i0+1]*fr
img = np.abs((val * np.hanning(ex.size+2)[1:-1]).sum(-1))
```
Result for the nine wire targets:

![bf example](images/bf_example_points.png)

The faint arcs 25 to 35 dB below the targets are expected: with all elements firing together, the true transmit time to a pixel is that of the nearest element, not z/c. A beamformer that models the actual transmit (or a focused transmit via `tau_tx`) removes them.

## 4. Checks you can run on your beamformer
| Check | Expected |
|---|---|
| Nine wire targets (x = -6, 0, 6 mm; z = 10, 20, 30 mm) | each peak within 0.1 mm of its true position |
| Lateral FWHM at z = 20 mm, single wire, unapodised | about 0.40 mm (128 elements) |
| Axial FWHM | about 0.55 mm |
| Double the strength of all scatterers | image doubles exactly |

## 5. Choosing between the two echo files
| File | Use when |
|---|---|
| `receiver_bf_input.npz` (default) | you want the Team 3 model: uniform c, no spreading or directivity |
| `alt_ti_stage3_channel_data.npz` | you want tissue-aware time of flight, spreading loss, element directivity (its own time axis, T' differs) |
Agreement between the two on wire targets: image correlation 0.98, peak positions identical. Do not combine them.

## 6. Requesting changes
Open an issue with: array parameters you need, transmit scheme (focus, plane waves, angles), whether you need noise, and the exact file layout change. Row order, units and key names are a contract ([06_DATA_DICTIONARY.md](06_DATA_DICTIONARY.md), rule "Compatibility").


---

## Status update (2026-10-14): the hand-off has been taken up
`receiver_bf_input.npz` is now consumed by `teams/T5_RX_Beamforming` and the whole chain runs from
`integration/02_full_relay` (see `docs/16_RECEIVER_BEAMFORMING_AND_FULL_RELAY.md`). The example in this document
(`load_and_beamform.py`) remains valid as a minimal reference. The Receiver BF team's own `receive_beamforming.py` was not
available, so the module contains a re-implementation with the same call signature; the team's function can be plugged in with
`RxConfig(beamformer=...)`. The next hand-off is `05_beamformed_outputs.npz` (B-mode image formation).
