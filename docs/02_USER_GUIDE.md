# User Guide

## 1. What the relay does
You give it a **source** (a phantom or a list of points). It returns the echo signal that a 128-element array would record, plus every intermediate array. See [03_ARCHITECTURE.md](03_ARCHITECTURE.md) for the four runners.

## 2. Choosing a source
| Source | How to pass it | Use when |
|---|---|---|
| Built-in preset | `--source generic` (or abdomen, brain, breast, cardiac, carotid, kidney, lung, msk, obstetric, prostate, thyroid) | quick realistic tissue with organs, vessels, bone, gas |
| Reflector list | `--source my_points.csv` (columns `x_mm,z_mm,amp`) | wire/point targets, calibration, resolution tests |
| Tissue cloud | `--source cloud.csv` (columns `x_mm,z_mm,tissue`) | you know which tissue sits at each coordinate |
| Phantom image | `--source phantom.png --mapping map.json --spacing-mm 0.2` | a segmented picture: each colour or label is a tissue |
| Python object | `run_relay(Phantom(...))`, `run_relay(np.array([[x_mm,z_mm,amp],...]), units="mm")` | scripting |

Coordinates: x lateral (0 = array centre), z depth (0 = skin surface). Point files are read in millimetres by default; in Python pass `units="m"`, `"cm"` or `"um"` to change this (`run_relay(..., units="mm")`). Column names with a unit suffix such as `x_mm` are accepted.

### Reflector list example (`my_points.csv`)
```
x_mm,z_mm,amp
0,10,1.0
0,20,1.0
5,30,0.5
```
### Tissue cloud example
```
x_mm,z_mm,tissue
0,5,liver
0,25,bone
```
Available tissue names: abscess, air, amniotic_fluid, bile, blood, bone, bowel_gas, brain_gray, brain_white, breast_fat, breast_glandular, calcification, cartilage, contrast_agent, coupling_gel, cyst_fluid, fat, fetal_soft, generic_soft, kidney_cortex, kidney_medulla, kidney_sinus, liver, lung, muscle, myocardium, pancreas, prostate, skin, spleen, tendon, thyroid, tumor_hyperechoic, tumor_hypoechoic, urine, uterus, vessel_wall, water. Properties of each: `teams/T4_Tissue_Interaction/docs/TISSUE_TABLE.md`. To add your own tissue, register it in Python with `tissue_interaction.register_tissue(...)` before calling `run_relay` (example file: `teams/T4_Tissue_Interaction/configs/custom_tissues_example.json`).

### Phantom image and mapping
See `teams/T4_Tissue_Interaction/docs/TISSUE_INTERACTION_EXPLAINED.md` for the mapping file formats (colour to tissue, label to tissue).

## 3. Command line reference (`run_relay.py`)
| Option | Default | Meaning |
|---|---|---|
| `--source` | `generic` | preset name, `points` (built-in 9 wires), points file, or image |
| `--mapping` | none | colour/label to tissue map for an image source |
| `--spacing-mm` | none | pixel size of an image source or grid size for a tissue cloud |
| `--max-scatterers` | 8000 | budget of diffuse (speckle) scatterers; more = better speckle, slower |
| `--attenuation-owner` | `tissue` | `tissue` or `ap`, see section 5 |
| `--background` | `generic_soft` | tissue used for path attenuation of a reflector list |
| `--n-elements` | 128 | override the array size (use to compare with 64) |
| `--out` | `outputs/<source>` | output folder |

## 4. Python use
```python
import _paths                                  # when running outside integration/01_ap_ti_relay, add the folder to sys.path first
from usoundsim_relay import run_relay, RelayConfig

res = run_relay("generic", RelayConfig(max_scatterers=12000), outdir="outputs/my_run")
res.received_pressure      # (128, T)  baton for Receiver BF
res.incident               # (N, T)    after AP part 1
res.reflected              # (N, T)    after TI stage 2
res.t_axis, res.element_x  # seconds, metres
res.timings                # seconds per runner
res.scatterers.xyz         # (N, 3) metres
```
Full signatures: [05_API_REFERENCE.md](05_API_REFERENCE.md).

## 5. Choosing who applies attenuation
| Setting | AP part | TI part | Use when |
|---|---|---|---|
| `attenuation_owner="tissue"` (default) | alpha0 forced to 0 | `path_model="full"`: tissue-specific, frequency-dependent attenuation, interface loss, shadows | realistic phantoms |
| `attenuation_owner="ap"` | uniform 0.5 dB/MHz/cm (from `config.ALPHA_0`) | `path_model="none"` | simple targets, comparison with the original Team 3 model |
Never both: that would count attenuation twice (verified in [08_VERIFICATION_REPORT.md](08_VERIFICATION_REPORT.md), V5).

## 6. Speed and size
Cost grows linearly with the number of scatterers N and with the number of elements. Measured: `--max-scatterers 2000` (3,586 scatterers in total) about 13 s; `--max-scatterers 8000` (9,592 in total) about 33 s. Memory 1 to 1.6 GB. Table and advice in [09_PERFORMANCE.md](09_PERFORMANCE.md).

## 7. Recipes
**Compare 64 and 128 elements**
```bash
python run_relay.py --source points --n-elements 64  --out outputs/pts64
python run_relay.py --source points --n-elements 128 --out outputs/pts128
```
**Focused transmit (hook for the Transmit Beamforming team)**
```python
import numpy as np
from nitk_usoundsim import config
ex = (np.arange(config.N_ELEMENTS) - (config.N_ELEMENTS - 1) / 2) * config.PITCH
d = np.hypot(ex - 0.0, 0.020)                   # distance of each element to a focus at x = 0, z = 20 mm
tau = (d.max() - d) / config.C                  # outer elements fire first [s]
pts = np.array([[0, 20, 1.0], [4, 20, 1.0]])    # x_mm, z_mm, amp
res = run_relay(pts, RelayConfig(tau_tx=tau), units="mm")
```
The check beamformer in `verify.py` assumes an unfocused transmit; with `tau_tx` use your own receive beamformer.

**Use the tissue-aware alternative instead of AP part 2**
`res.ti_channel` and `res.ti_channel_t` (also `alt_ti_stage3_channel_data.npz`). Use it or `res.received_pressure`, never both.

**Sound-speed or density change in your phantom**
Edit the tissue in a JSON file and register it (see TI docs); nothing changes in the relay.

## 8. Reading the overview figure
Top row: scatterer map (runner 1), incident waveform at one scatterer (runner 2), reflected waveform (runner 3). Bottom row: the 128-element echo image (runner 4, the hand-off array), a quick delay-and-sum check image, and the shapes handed over at each step.
