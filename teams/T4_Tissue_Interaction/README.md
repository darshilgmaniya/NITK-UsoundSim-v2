# NITK-UsoundSim - Tissue Interaction (Subtask 04)

Physics-based **tissue interaction** stage of an end-to-end in-silico ultrasound pipeline
(Transducer -> Transmit BF -> Acoustic Propagation -> **Tissue Interaction** -> Receiver BF -> Reconstruction -> B-mode -> Post-processing).

You give it a **phantom of any body region** - as **points / coordinates** (CSV/NPZ/NPY: tissue clouds or scatterer lists) or as an image/array. It converts the phantom into
tissue-aware acoustic maps, places specular and diffuse scatterers, and turns the incident pressure at those scatterers into
the **reflected / scattered pressure waveforms** the next stage consumes. Everything is also written out as **2-D grayscale images**.

![abdomen montage](docs/images/montage_abdomen.png)

*Abdomen preset: gallbladder, vessels, kidney, tumour/cyst, rib and bowel gas. Note the dark gallbladder and cyst, the
bright interfaces, and the acoustic shadows behind the rib and the gas.*

## What it does

| Input | Output |
|---|---|
| user phantom as **points** (`x,z,tissue` or `x,z,amp`) or as label / colour / grey-level image + pixel size | **`receiver_bf_input.npz` - per-element echoes for Receiver Beamforming** |
| | `tissue_scatterers.npz` - scatterer list for Acoustic Propagation |
| (optional) incident pressure from Acoustic Propagation | `tissue_interaction_output.npz` - `reflected_pressure_input (n_scatterers, n_samples)` |
| body part / probe preset | `tissue_property_maps.npz` - c, rho, alpha, Z maps for Acoustic Propagation |
| custom tissues (JSON/YAML) | 2-D grayscale PNGs + montage (`images/`, `tissue_interaction_montage.png`) |

Highlights

* **Every body part**: 12 presets (abdomen, kidney, thyroid, breast, cardiac, carotid, MSK, obstetric, brain, lung, prostate, generic)
  with typical probe class / frequency, plus a 40-entry tissue library and user-defined tissues.
* **Contrast classes**: water, cyst fluid, blood, **air / gas**, lung, bone, calcification, hypo/hyper-echoic tumours, abscess and **microbubble contrast agent** are all distinct classes with their own impedance, attenuation and backscatter.
* **Real-life effects**: Fresnel reflection at oblique incidence (critical angle included), specular directivity, speckle, frequency-dependent attenuation (spectrum down-shift with depth), acoustic shadowing, tissue-dependent time of flight.
* **Drop-in contract**: output key names and shape match what `return_propagation()` expects (see [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md)).
* Deterministic (`seed`), tested (`pytest`), CI workflow included.

## Single-file version
`single_file/tissue_interaction_complete.py` is the **entire package in one file** (generated, bit-identical results). Copy it next to your notebook:
```bash
python tissue_interaction_complete.py selftest                       # 6 quick physics/interface checks
python tissue_interaction_complete.py from-ap --forward forward.npz --out outputs/echo
```
```python
import tissue_interaction_complete as ti        # same API as the package
```
Regenerate it after any change with `python scripts/build_single_file.py`.

## Install

```bash
git clone https://github.com/<your-username>/NITK-UsoundSim-Tissue-Interaction.git
cd NITK-UsoundSim-Tissue-Interaction
pip install -e .            # numpy, scipy, matplotlib, pillow, pyyaml
pip install -e ".[dev]"     # + pytest
```
Colab: open `notebooks/T4_tissue_interaction.ipynb` (first cell installs the package).

## Quick start

### 1. Demo on a built-in phantom
```bash
python -m tissue_interaction demo --body-part abdomen --out outputs
python -m tissue_interaction list-body-parts
python -m tissue_interaction list-tissues
python -m tissue_interaction selftest
```

### 2. Main flow: Acoustic Propagation file -> echo signals for Receiver Beamforming
```bash
python -m tissue_interaction from-ap --forward forward_propagation_output.npz --out outputs/echo
#   optional medium:  --medium-points points_abdomen.csv --body-part abdomen      (heterogeneous attenuation / shadows)
```
```python
res = ti.run_from_ap("forward_propagation_output.npz")      # res.channel_rf == receiver_bf_input.npz['received_pressure']
```
Reads `incident_pressure`, `scatterer_xyz`, `pulse_t/pulse_p`, `element_x` (see `docs/INTERFACE_CONTRACT.md`).

### 2a. Phantom as POINTS / COORDINATES  ->  Receiver Beamforming
```bash
# tissue cloud: x_mm,z_mm,tissue   (full physics: boundaries, speckle, shadows)
python -m tissue_interaction run --points UsoundSim_Points_Dataset/tissue_clouds/points_abdomen.csv --body-part abdomen --out outputs/abdomen
# scatterer list: x_mm,z_mm,amp    (wire targets for testing the beamformer)
python -m tissue_interaction run --points UsoundSim_Points_Dataset/scatterer_lists/point_targets.csv --body-part generic --f0-mhz 5 --fs-mhz 60 --out outputs/targets
```
```python
res = ti.run_points("points.csv", ti.config_for_body_part("thyroid"), outdir="outputs/thyroid")
res.channel_rf        # (n_elements, n_samples)  == receiver_bf_input.npz['received_pressure']
```
Read **[`docs/TISSUE_INTERACTION_EXPLAINED.md`](docs/TISSUE_INTERACTION_EXPLAINED.md)** for the full explanation of inputs, outputs and the model.

### 2b. Phantom as an image (still supported)
```bash
# label mask (pixel value = label id) + mapping + pixel size
python -m tissue_interaction run --phantom sample_data/phantom_abdomen_labels.png \
       --mapping sample_data/phantom_abdomen_mapping.json --body-part abdomen --out outputs/my_abdomen

# colour-coded segmentation
python -m tissue_interaction run --phantom sample_data/example_color_phantom.png \
       --mapping sample_data/example_color_mapping.json --body-part thyroid --out outputs/colour

# grey-level anatomical image -> tissue classes (dark -> bright)
python -m tissue_interaction run --phantom sample_data/example_intensity_phantom.png --mode intensity \
       --mapping sample_data/example_intensity_mapping.json --body-part thyroid --out outputs/intensity
```
The pixel size is **mandatory** (`--spacing-mm 0.2`, `--spacing-mm dx dz`, `--width-mm`, or `spacing_mm` in the mapping file) because a
wrong scale silently gives wrong physics. The skin surface must be at the **top** row (use `--flipud` otherwise).

Mapping file (JSON or YAML, see [`configs/custom_tissues_example.json`](configs/custom_tissues_example.json)):
```json
{
  "spacing_mm": [0.2, 0.2],
  "labels": {"0": "coupling_gel", "1": "skin", "4": "liver",
             "5": {"name": "my_lesion", "base": "tumor_hypoechoic", "density": 1075, "speed": 1555, "backscatter_db": -8}},
  "colors": {"#ff0000": "liver", "0,0,255": "blood"},
  "ranges": [[0, 60, "water"], [60, 160, "liver"], [160, 256, "bone"]],
  "classes": ["fat", "liver", "bone"]
}
```
Use one of `labels` / `colors` / `ranges` / `classes` depending on the mode. Unknown label ids raise an error (or use `--loose`).

### 3. Python API
```python
import tissue_interaction as ti

ph  = ti.load_phantom("my_phantom.png", mapping="my_mapping.json", spacing_mm=0.2)   # or ti.make_phantom("thyroid")
cfg = ti.config_for_body_part("abdomen")          # probe + f0 typical for the region; edit any field afterwards
res = ti.run(ph, cfg, outdir="outputs/my_run")    # mock incident pulse
res.reflected.shape                               # (n_scatterers, n_samples)
```

### 4. Hand-off to / from the Acoustic Propagation team
```bash
python -m tissue_interaction scatterers --phantom ... --mapping ... --body-part abdomen --out hand_off   # stage 1
#  -> Acoustic Propagation: forward_propagation() on hand_off/tissue_scatterers.npz  ->  incident.npz
python -m tissue_interaction interact --scatterers hand_off/tissue_scatterers.npz --incident incident.npz --out hand_off  # stage 2
#  -> hand_off/tissue_interaction_output.npz  (feeds return_propagation())
```
Python equivalent: `run_stage1`, `load_scatterers`, `compute_reflected_pressure`, `save_reflected`.

## Outputs (2-D grayscale)

`images/*.png` (8-bit, row 0 = skin): `tissue_labels`, `density`, `speed_of_sound`, `impedance`, `attenuation`, `backscatter`,
`interface_reflectivity`, `scatterer_map`, `echo_energy` (energy of the module's actual output) and `qa_bmode_preview`
(a quick convolutional preview so you can check the module without the other teams; it is **not** the B-mode team's product).

![contrast targets](docs/images/montage_contrast_targets.png)

*Contrast-target phantom (`generic`): water, hypo-/hyper-echoic tumours, contrast agent, blood, gas, bone, calcification. Gas and bone cast full shadows.*

## Choosing sizes (memory)
`reflected_pressure_input` has `n_scatterers x n_samples x 4 bytes`. Defaults: <= 30 000 diffuse scatterers (+ boundary
scatterers). A 120 x 120 mm abdomen at 3 MHz then has about 1 scatterer per resolution cell (sparse speckle); crop the
field of view or raise `max_scatterers` for fully developed speckle (>= 6-10 per cell). Peak RAM for the abdomen demo is about 2.5 GB.

## Repository layout
```
src/tissue_interaction/
  tissue_library.py     tissue properties (rho, c, alpha, backscatter) + custom tissues
  body_presets.py       12 body-region presets + probe classes
  phantom.py            Phantom container, up-sampling
  phantom_io.py         load user phantoms (labels / colour / intensity)
  phantom_generators.py synthetic phantoms for every preset
  scatterers.py         stage 1: interface + diffuse scatterers, path effects
  interaction.py        stage 2: reflected pressure; mock incident pulse
  visualize.py          grayscale maps + QA preview
  outputs.py            npz / png / json writers and readers
  points_io.py          phantom as points / coordinates (tissue cloud, scatterer list)
  channel_data.py       stage 3: per-element echoes for Receiver Beamforming
  pipeline.py, cli.py   high-level API and command line
configs/  docs/  notebooks/T4_tissue_interaction.ipynb  sample_data/  scripts/  tests/
```

## Tests
```bash
pip install -e ".[dev]" && pytest -q      # 43 tests: Fresnel values, polarity, shadowing, time of flight, points input, channel-data focusing, I/O contract, all 12 body parts
```

## Documentation
* [`docs/TISSUE_INTERACTION_EXPLAINED.md`](docs/TISSUE_INTERACTION_EXPLAINED.md) - **start here**: role, inputs, outputs, model, assumptions, questions for the Receiver BF team
* [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md) - exact keys / shapes / units, and what to confirm with Acoustic Propagation
* [`docs/PHYSICS.md`](docs/PHYSICS.md) - equations and known simplifications
* [`docs/TISSUE_TABLE.md`](docs/TISSUE_TABLE.md) - every tissue, body preset and label legend

## Limitations / roadmap
* 2-D, straight-ray, single-scattering model (no refraction, reverberation, harmonics); library values are textbook mid-range, not Philips calibration data.
* Planned: 3-D phantoms (`y` axis is already in the scatterer file), per-transmit-event incident arrays, memory-mapped output for very large scatterer sets, calibration of probe presets against the real Philips transducer parameters.

## References (for the report)
Duck, *Physical Properties of Tissue* (1990); Szabo, *Diagnostic Ultrasound Imaging: Inside Out*; Bushberg et al., *The Essential Physics of Medical Imaging*; Jensen & Svendsen, Field II (1992); Treeby & Cox, k-Wave (2010).
