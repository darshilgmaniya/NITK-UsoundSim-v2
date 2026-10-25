# Integration 02 - full relay (runner 5: Receiver Beamforming)

One call from a phantom to a beamformed image:

```
Tissue Interaction -> Acoustic Propagation -> Tissue Interaction -> Acoustic Propagation -> Receiver Beamforming
     runner 1              runner 2                 runner 3              runner 4              runner 5 (new)
```
Runners 1 to 4 are `integration/01_ap_ti_relay` (unchanged); runner 5 is `teams/T5_RX_Beamforming`. This folder owns no physics, only the glue.

## Run
```bash
pip install -r ../../requirements.txt
python run_full_relay.py --source points                     # 9 wires, about 6 s
python run_full_relay.py --source abdomen --max-scatterers 8000
python run_full_relay.py --source my_points.csv              # x_mm,z_mm,amp   or   x_mm,z_mm,tissue
python run_full_relay.py --source phantom.png --mapping map.json --spacing-mm 0.2
python run_full_relay.py --handoff outputs/points/receiver_bf_input.npz --apod hamming --f-number 2
```
Receiver options: `--apod hann|hamming|blackman|tukey|rect`, `--f-number`, `--line-spacing-mm`, `--n-lines`, `--xrange`, `--zrange`, `--dr`, `--no-focus`.
Outputs (folder `outputs/<source>/`, git-ignored): the four relay files, `receiver_bf_input.npz`, **`05_beamformed_outputs.npz`**, `full_relay_overview.png`.

From Python:
```python
import _paths                                   # makes modules importable (repo-relative)
from usoundsim_full import run_full
from usoundsim_relay import RelayConfig
from receiver_beamforming import RxConfig

res = run_full("abdomen", RelayConfig(max_scatterers=8000), RxConfig(apodization="hann", f_number=1.5), outdir="outputs/abdomen")
res.beamformed_rf, res.bf.z_axis, res.bf.x_lines     # hand these to the B-mode team
res.relay.received_pressure                          # the baton between runner 4 and 5
```

## Contents
| Path | Purpose |
|---|---|
| `usoundsim_full/full_relay.py` | `run_full`, `run_from_handoff` |
| `usoundsim_full/plots.py` | overview figure |
| `run_full_relay.py` | command line |
| `examples/use_your_own_beamformer.py` | plug in the team's own `receive_beamforming.py` |
| `full_relay_demo.ipynb` | Colab-ready walk-through |
| `tests/` | 14 end-to-end tests |
| `verification/run_verification.py` | reproduces the numbers in `docs/08_VERIFICATION_REPORT.md` |

## What was checked
Nine wire targets within 0.036 mm laterally and 0.010 mm in depth; exact amplitude linearity; 64, 96 and 128 elements; focused transmit delays;
CSV, image and preset inputs. Input matrix (single CPU core, all five runners):

| Source | Scatterers | Channel data | Image (depth x lines) | Time [s] |
|---|---|---|---|---|
| points (9 wires) | 9 | 128 x 2166 | 1715 x 85 | 4.9 |
| CSV x,z,amp | 3 | 128 x 2223 | 1819 x 52 | 3.4 |
| CSV x,z,tissue (0.5 mm grid) | 8213 | 128 x 2726 | 2208 x 114 | 29.1 |
| image phantom + mapping (0.2 mm) | 22835 | 128 x 5711 | 4312 x 127 | 89.3 |
| single scatterer | 1 | 128 x 977 | 676 x 7 | 0.1 |
| shallow target (2-3 mm) | 2 | 128 x 810 | 312 x 13 | 0.1 |
| 64 / 96 elements | 9 | 64 x 2047 / 96 x 2102 | 1715 x 64 / 1715 x 85 | 0.4 / 0.7 |
| attenuation owner = AP | 9 | 128 x 2166 | 1715 x 85 | 1.4 |
| generic | 3078 | 128 x 4864 | 3794 x 127 | 17.2 |
| abdomen | 9211 | 128 x 8387 | 6387 x 127 | 58.5 |
| kidney | 6453 | 128 x 8135 | 6387 x 127 | 34.5 |
| thyroid | 3834 | 128 x 3110 | 2233 x 127 | 18.2 |
| breast | 3801 | 128 x 3686 | 2753 x 127 | 35.6 |
| cardiac | 7950 | 128 x 9264 | 7431 x 127 | 88.4 |
| musculoskeletal | 3667 | 128 x 3318 | 2493 x 127 | 16.3 |
| obstetric | 9484 | 128 x 10013 | 7945 x 127 | 103.6 |
| brain | 6183 | 128 x 6106 | 4830 x 127 | 75.9 |
| lung | 3159 | 128 x 3645 | 2753 x 127 | 22.0 |

Details: `docs/08_VERIFICATION_REPORT.md`, section Integration 02.

## Good to know
* Phantoms wider than the array are clipped to the aperture; `BeamformResult.notes` says how many scatterers fell outside.
* The team's `receive_beamforming.py` was not available when this was built; module 03 contains a re-implementation with the same call signature.
* The beamformed preview is not a finished B-mode image.
