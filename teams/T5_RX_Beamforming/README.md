# Module 03 - Receiver Beamforming

Turns the **128 channel recordings** produced by Acoustic Propagation into an **image**: delay-and-sum with dynamic
receive focus and dynamic aperture. Runs on its own (`pip install -r requirements.txt`, then `pytest -q tests`) and plugs
into the relay as **runner 5** (see [`integration/02_full_relay`](../../integration/02_full_relay)).

```
received_pressure (n_el, T)  --->  das_beamform  --->  beamformed_rf (n_depth, n_lines)  --->  B-mode team
        + t_axis, element_x                              + z_axis, x_lines
```

## What is in this folder
| Path | Content |
|---|---|
| `src/receiver_beamforming/beamformer.py` | `das_beamform()` - the DAS core, same call signature as the team's `liver_echo_simulation.py` |
| `src/receiver_beamforming/pipeline.py` | `beamform()` - one call for any input: validates, picks the field of view, beamforms, previews |
| `src/receiver_beamforming/io.py` | read the hand-off (`receiver_bf_input.npz`), validate, write `05_beamformed_outputs.npz` |
| `src/receiver_beamforming/config.py` | `RxConfig` (user options) and `BeamformConfig` (geometry, same field names as the team's `LiverConfig`) |
| `src/receiver_beamforming/bmode_preview.py` | envelope + log compression, **preview only** |
| `src/receive_beamforming.py` | shim so the team's file `from receive_beamforming import das_beamform` works unchanged |
| `demos/liver_echo_simulation.py` | the Receiver BF team's original script, **unmodified** |
| `demos/run_liver_demo.py` | runs that script against this module (`--save out.png` for headless) |
| `tests/` | 23 tests |
| `docs/RECEIVER_BEAMFORMING.md` | theory, interface, options, limits |

## Use it
```python
import sys; sys.path.insert(0, "src")
import receiver_beamforming as rb

res = rb.beamform("receiver_bf_input.npz", rb.RxConfig(apodization="hann", f_number=1.5))
res.beamformed_rf     # (n_depth, n_lines) signed RF  -> hand to the B-mode team
res.bmode_db          # preview image, 0 to -50 dB
res.z_axis, res.x_lines
```
Team-style call (per-line data, 3-D array):
```python
rf, z = rb.das_beamform(raw_rf, x_elements, x_lines, config, apply_focus=True,
                        t_axis=t_axis, apodization="hann", f_number=1.0)
```
`raw_rf` may be `(n_el, n_samples, n_lines)` (one record per scan line) **or** `(n_el, n_samples)` (one record for all
lines, which is what the relay produces).

## Options (`RxConfig`)
| Option | Default | Meaning |
|---|---|---|
| `apodization` | `"hann"` | `hann`, `hamming`, `blackman`, `tukey`, `rect` |
| `f_number` | `1.0` | dynamic aperture half-width = z / (2 F). `None` or `0`: all elements |
| `line_spacing_m` / `n_lines` | element pitch | image line density (lines capped at `max_lines`) |
| `x_range_m`, `z_range_m` | auto | field of view; auto uses the scatterer truth if present, else the aperture and record length |
| `apply_focus` | `True` | dynamic receive focusing |
| `tx_mode`, `tx_delays` | `"vertical"` | transmit time model: `z/c`, or earliest element arrival (needed with focused transmit) |
| `normalize` | `True` | divide by the sum of apodisation weights |
| `dynamic_range_db` | `50` | preview only |
| `beamformer` | `None` | plug in the team's own function with the `das_beamform` signature |

## Important notes
* **`receive_beamforming.py` was not part of the upload.** `das_beamform` here was written from the way the team's script calls
  it (arguments and return values). If the team has its own file, use it through `RxConfig(beamformer=...)`
  (see `integration/02_full_relay/examples/use_your_own_beamformer.py`); nothing else changes.
* Output is **RF, not B-mode**. Envelope detection and log compression are only a preview so the chain can be inspected; the B-mode
  team owns the real image formation.
* Verified: nine wire targets land within 0.04 mm of their true position; amplitude is exactly linear; see
  [`docs/08_VERIFICATION_REPORT.md`](../../docs/08_VERIFICATION_REPORT.md).
