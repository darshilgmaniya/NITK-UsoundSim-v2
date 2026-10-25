# 16 - Receiver Beamforming and the full relay

## 1. The relay race, now with five runners
```
 phantom / points / image
        |  runner 1  Tissue Interaction stage 1     01_tissue_scatterers.npz
        |  runner 2  Acoustic Propagation part 1    02_part1_forward_propagation_outputs.npz
        |  runner 3  Tissue Interaction stage 2     03_tissue_interaction_output.npz
        |  runner 4  Acoustic Propagation part 2    04_part2_return_propagation_outputs.npz  (+ receiver_bf_input.npz)
        v  runner 5  Receiver Beamforming (NEW)     05_beamformed_outputs.npz  ->  B-mode image formation (next team)
```
Runners 1 to 4 live in `integration/01_ap_ti_relay` and are called, not copied. `integration/02_full_relay` only adds runner 5 and
the glue, so a fix in an upstream module reaches the full chain automatically.

![full relay on the points demo](images/full_relay_points.png)
![full relay on the generic phantom](images/full_relay_generic.png)
*Right-hand panels show only the part of the phantom inside the array aperture (+-9.8 mm for 128 elements); the generic phantom is wider.*

## 2. The baton between runner 4 and runner 5
`received_pressure (n_el, T)`, `t_axis (T)`, `element_x (n_el)` plus scalars. In memory the arrays pass unchanged (a test checks they are
bit-identical); on disk they pass through `receiver_bf_input.npz` (float32).

## 3. Three ways to run it
| Goal | Command / call |
|---|---|
| Everything, one command | `python integration/02_full_relay/run_full_relay.py --source abdomen` |
| Everything, from Python | `from usoundsim_full import run_full; res = run_full("abdomen", RelayConfig(), RxConfig())` |
| Only the beamformer, from a saved file | `python run_full_relay.py --handoff outputs/points/receiver_bf_input.npz --apod hamming` |
| Only the beamformer, from arrays | `receiver_beamforming.beamform(make_receiver_input(rx, t, element_x, fs_hz=...))` |

`--source` accepts `points`, any phantom preset (generic, abdomen, kidney, thyroid, breast, cardiac, carotid, msk, obstetric, brain, lung,
prostate), a points CSV (`x_mm,z_mm,amp` or `x_mm,z_mm,tissue`), or a phantom image with `--mapping` and `--spacing-mm`.

## 4. Choices that make it work for any input
* Bad receiver options are rejected **before** the slow simulation starts.
* Per-element transmit delays given to the relay (`RelayConfig.tau_tx`) are passed on to the beamformer automatically (`tx_mode="nearest"`).
* The field of view is chosen from the data. Phantoms wider than the array are clipped to the aperture and a note says so.
* Array size, sampling rate, sound speed and centre frequency all come from the hand-off, nothing is hard-coded in module 03.
* The team's own beamformer plugs in through `RxConfig(beamformer=fn)`; see `examples/use_your_own_beamformer.py`.

## 5. Where things are
| Part | Path |
|---|---|
| Beamformer and tools | `teams/T5_RX_Beamforming/src/receiver_beamforming/` |
| Module explanation | `teams/T5_RX_Beamforming/docs/RECEIVER_BEAMFORMING.md` |
| Team's original demo, unmodified | `teams/T5_RX_Beamforming/demos/liver_echo_simulation.py` |
| Full-relay glue, CLI, tests, verification | `integration/02_full_relay/` |
| Numbers | `docs/08_VERIFICATION_REPORT.md` (section Integration 02) |

## 6. Limits
Single-angle plane-wave style transmit, linear array, 2-D, no sound-speed correction, no time-gain compensation, no noise model. The
beamformed preview is not a finished B-mode image. Full list: `teams/T5_RX_Beamforming/docs/RECEIVER_BEAMFORMING.md` section 5.
