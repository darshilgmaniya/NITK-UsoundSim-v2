# Receiver Beamforming - explained

## 1. What the module does
Acoustic Propagation delivers `received_pressure`: what each of the 128 receive elements recorded over time. One scatterer
appears in every channel at a different time (the farther the element, the later the echo). **Beamforming undoes those
time differences.** For every image point it asks: "if a reflector were here, when would each element have heard it?", picks
that sample from each channel, and adds them. Echoes from that point add up in phase; echoes from elsewhere do not.

```
channel 1 ~~~~~/\/\~~~~~~~~~~            pick sample at t1 \
channel 2 ~~~~~~/\/\~~~~~~~~~     --->   pick sample at t2  >  weight, sum  ->  one image sample
channel 3 ~~~~~~~/\/\~~~~~~~~            pick sample at t3 /
```

## 2. Interface
| | Name | Shape | Unit | Notes |
|---|---|---|---|---|
| in | `received_pressure` | (n_el, T) | a.u. | row m = element m (relay layout). The team layout (n_el, T, n_lines) is also accepted |
| in | `t_axis` | (T,) | s | uniform, increasing; time zero = centre of the transmit pulse |
| in | `element_x` | (n_el,) | m | any array size; unsorted positions are sorted together with their channels |
| in | `sampling_rate_hz`, `c0_nominal_m_s`, `f0_hz`, `pitch_m` | scalar | | read from the file; derived from `t_axis` / `element_x` when absent |
| in | `scatterer_xyz` | (N,3) | m | optional truth, used only to choose the field of view automatically |
| out | `beamformed_rf` | (n_depth, n_lines) | a.u. | signed RF, **not** B-mode |
| out | `z_axis` | (n_depth,) | m | depth, spacing c / (2 fs) |
| out | `x_lines` | (n_lines,) | m | lateral position of each image line |
| out | `envelope_preview`, `bmode_db_preview` | (n_depth, n_lines) | a.u., dB | preview only |

File written by the relay layer: `05_beamformed_outputs.npz` (keys above plus `dynamic_range_db`, `apodization`, `f_number`,
`sampling_rate_hz`, `c0_nominal_m_s`, `f0_hz`, `n_elements`).

## 3. Algorithm (per image line at x_l, per depth z)
1. Time of flight to element e: `t_e = t_tx + sqrt((x_l - x_e)^2 + z^2) / c`.
2. `t_tx` (transmit time to the point): `z/c` in `"vertical"` mode (exact for a plane wave, the relay default, and a good model for a
   line transmit near the axis); the earliest element arrival `min_e (tau_e + d_e/c)` in `"nearest"` mode, which stays correct
   when the Transmit Beamforming team supplies per-element delays `tau_e`.
3. Sample channel e at `t_e` by linear interpolation; samples outside the record contribute zero.
4. Dynamic aperture: only elements with `|x_e - x_l| <= z / (2F)` count, weighted by the apodisation window across that
   aperture. The aperture therefore grows with depth and keeps the lateral resolution roughly constant.
5. Output `sum_e w_e s_e / sum_e w_e`. The division makes the brightness independent of how many elements are active.

The depth axis is `z = 0, dz, 2dz, ...` with `dz = c / (2 fs)` (19.25 um at 40 MHz), the same definition as the team's `LiverConfig`.

## 4. Why it works for any input
| Situation | Handling |
|---|---|
| Any number of elements (64, 96, 128, ...) | everything is derived from `element_x`; nothing is hard-coded |
| Transposed array (T, n_el) | detected and fixed with a warning |
| Elements stored in random order | sorted together with their channels |
| No scatterer truth in the file (real or foreign data) | field of view = aperture width and 90 % of the record depth, with a note |
| Tiny phantom (one point) | minimum width of four element pitches |
| Very wide phantom | line count capped by `max_lines`, with a note |
| NaN / Inf, wrong shapes, non-uniform or non-increasing time axis | clear `ValueError` naming the problem |
| Focused or steered transmit | `tx_mode="nearest"` with `tx_delays` (set automatically by the integration layer when the relay used `tau_tx`) |
| Team's own beamformer | `RxConfig(beamformer=fn)` |

## 5. Known limits (honest list)
* **Transmit model.** `t_tx` is a geometric approximation. A plane wave from a finite aperture gives a faint arc of artefacts
  25 to 35 dB below a strong target (documented in `docs/10_RECEIVER_BF_HANDOFF.md`); a focused transmit or compounding removes them.
* **No sound-speed correction.** The beamformer assumes one speed `c`; tissue with a different speed shifts structures slightly.
* **No time-gain compensation, no noise model, no coherence weighting.** Attenuation is left in the data on purpose.
* **Single-angle, linear array.** No steering angle, no curved arrays, 2-D only (y = 0).
* **Preview image is not B-mode.** Envelope and log compression here exist only so the chain can be checked.
* **The team's `receive_beamforming.py` was not available.** `das_beamform` follows the team's call signature; its numerical details
  (interpolation, apodisation formula, normalisation) are this module's, not necessarily the team's.

## 6. Run the tests
```bash
cd teams/T5_RX_Beamforming && python -m pytest -q tests      # 23 tests, ~5 s
```
What they cover: focusing accuracy, every apodisation, F-number `None`, exact linearity, 2-D equals 3-D layout, team-style config
object, transmit-delay handling, window properties, every error message, transposed and unsorted input, automatic field of view,
user field of view, plug-in beamformer, option validation, save/load round trip.
