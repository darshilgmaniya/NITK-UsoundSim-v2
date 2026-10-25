# Integration 03 - full pipeline (runners 6 and 7: B-mode, scan conversion, despeckling)

One call from **any phantom** to the **final image**:

```
TI -> AP -> TI -> AP -> Receiver BF  ->  B-mode formation  ->  scan conversion + despeckle
 runners 1-4     runner 5                 runner 6                  runner 7
 (integration 01) (integration 02)        (module 04)               (module 05)
```
This folder owns no physics, only the glue. Runners 1-5 are `integration/02_full_relay`, unchanged.

## Run
```bash
pip install -r ../../requirements.txt
python run_pipeline.py --source points                                   # 9 wires, about 5 s
python run_pipeline.py --source abdomen                                  # built-in preset (also kidney, thyroid, breast, cardiac, ...)
python run_pipeline.py --source my_points.csv                            # x_mm,z_mm,amp   or   x_mm,z_mm,tissue
python run_pipeline.py --source phantom.png --mapping map.json --fit-aperture   # YOUR image phantom
```
**Give the file while the code runs** (no editing of any code):
```
python run_pipeline.py --upload     # file dialog: pick the phantom (+ its mapping .json)
python run_pipeline.py              # no arguments: a menu asks what to run
```
In `full_pipeline_demo.ipynb` run the *Upload your own phantom* cell (an Upload button appears). Code: `usoundsim_pipeline/upload.py`.

Outputs (`outputs/<source>/`, git-ignored): the baton files of integrations 01 and 02, `06_recon_rf.npz`, `07_bmode_outputs.npz`,
`08_final_image.npz`, **`final_image.png`** (the picture), **`final_image.npy`** (array, [0, 1]), `pipeline_overview.png`.

## What a phantom can be
| Input | How |
|---|---|
| Preset | `--source abdomen` (see the list in `teams/T4_Tissue_Interaction/src/tissue_interaction/body_presets.py`) |
| Points file | CSV with `x_mm,z_mm,amp` (point targets) or `x_mm,z_mm,tissue` (a tissue name from the tissue library) |
| Label-map image | PNG/JPG whose pixel values are labels + `--mapping map.json` (label -> tissue name, optional `spacing_mm`). Examples: `teams/T4_Tissue_Interaction/sample_data/` |
| Intensity or colour image | same, with the mapping file formats of `example_intensity_mapping.json` / `example_color_mapping.json` |
| Python array | `run_pipeline(np.array([[x_mm, z_mm, amp], ...]))` |

**Size matters.** The image is clipped to the array width. A 144 mm wide phantom on a 19.6 mm array gives a narrow strip.
`--fit-aperture` chooses the pixel size so an image phantom is 95 % as wide as the array (it shrinks the phantom physically).
Or set `--spacing-mm` yourself, or use a wider probe (`--pitch-mm`, `--n-elements`).

## Probe
Defaults come from `teams/T3_Acoustic_Propagation/code/nitk_usoundsim/config.py`: **128 elements, 5 MHz, pitch lambda/2 = 0.154 mm**
(aperture 19.6 mm). Change with `--n-elements`, `--f0-mhz`, `--pitch-mm`.
**Warning:** a pitch above lambda (for example 0.30 mm at 8 MHz = 1.56 lambda, the Philips L12-4 setting of the first project)
gives grating-lobe (X-shaped) artefacts in this beamformer. Keep the pitch near lambda/2 unless you want to study that effect.

## Options
Receiver: `--apod hann|hamming|blackman|tukey|rect  --f-number  --line-spacing-mm  --n-lines  --xrange  --zrange  --no-focus`.
Image: `--dr 50  --despeckle guided|none  --guided-r 4  --guided-eps 0.001  --gamma 1.0`.
Speed: `--max-scatterers` (default 8000; more = finer speckle, slower). The comparison channel of TI stage 3 is off (`--with-ti-channel`).

## From Python
```python
import _paths
from usoundsim_pipeline import run_pipeline, PostConfig
from usoundsim_relay import RelayConfig
from receiver_beamforming import RxConfig

res = run_pipeline("phantom.png", RelayConfig(also_ti_channel=False), RxConfig(), PostConfig(despeckle="guided"),
                   mapping="map.json", spacing_mm=0.04, outdir="outputs/mine")
res.final        # (H, W) final image in [0, 1]       res.x_img, res.z_img  axes [m]
res.bmode        # (depth, lines) before scan conversion
res.full.beamformed_rf, res.full.relay.received_pressure     # earlier batons
```

## Timing (one CPU core)
| Phantom | Scatterers | Time |
|---|---|---|
| 9 wires | 9 | 5 s |
| abdomen label map, `--fit-aperture`, 5 MHz | 11,735 | 30 s |
| same, 8 MHz / 0.30 mm | 11,821 | 41 s |
| abdomen label map, 0.3 mm pixels, clipped, TI stage 3 on (not recommended) | 17,132 | 4 min 47 s |

## Tests
`python -m pytest -q` (6 tests): shapes and ranges, nine wires within 0.5 mm of the truth, `despeckle="none"` equals the scan-converted image,
CSV phantom, image phantom with mapping and saved files, early failure on bad options.
