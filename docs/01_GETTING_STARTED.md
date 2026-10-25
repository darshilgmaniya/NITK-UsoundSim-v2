# Getting Started

Goal: run the full relay (Tissue Interaction -> Acoustic Propagation -> Tissue Interaction -> Acoustic Propagation) and open the result in under five minutes.

## 1. Requirements
| Need | Version |
|---|---|
| Python | 3.10 or newer |
| Packages | numpy, scipy, matplotlib, pillow, pyyaml (`pip install -r requirements.txt`), pytest for the tests |
| Memory | 2 GB free for the default phantoms, 4 GB for the abdomen phantom with 12,000 diffuse scatterers |
| Disk | under 50 MB for the repository; a full `generic` run writes about 500 MB of `.npz` files |
| OS | Linux, macOS, Windows (pure Python, no compiled code) |

## 2. Install
```bash
git clone https://github.com/OWNER/NITK-UsoundSim.git
cd NITK-UsoundSim
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Nothing else to install: the integration adds the two module folders to `sys.path` itself (`integration/01_ap_ti_relay/_paths.py`).

## 3. First run (about 5 seconds)
```bash
cd integration/01_ap_ti_relay
python run_relay.py --source points
```
Expected console output:
```
[setup ] 128 elements, pitch 0.154 mm, aperture 19.6 mm, f0 5.0 MHz, fs 40 MHz, attenuation owner = tissue
[run 1 ] TI stage 1      : 9 scatterers ...
[run 2 ] AP part 1       : incident_pressure (9, 2166), t = -1.2..52.9 us
[run 3 ] TI stage 2      : reflected_pressure_input (9, 2166), path_model=full
[run 4 ] AP part 2       : received_pressure (128, 2166)  -> Receiver BF
[extra ] TI stage 3      : channel data (128, 3053) (alternative path, tissue-aware c)
DONE. Baton for Receiver BF: .../outputs/points/receiver_bf_input.npz
```
Open `outputs/points/relay_overview.png`: nine wire targets, the incident and reflected waveform of one scatterer, the 128-element echo image, and a quick check image where all nine targets are in focus.

## 4. Check that everything works
```bash
pytest -q                                               # 7 relay tests, from integration/01_ap_ti_relay
(cd ../../teams/T3_Acoustic_Propagation/code && python -m pytest -q nitk_usoundsim)   # 25
(cd ../../teams/T4_Tissue_Interaction && PYTHONPATH=src python -m pytest -q)           # 43
```

## 5. Load the result in Python
```python
import numpy as np
d = np.load("outputs/points/receiver_bf_input.npz")
rx, t, ex = d["received_pressure"], d["t_axis"], d["element_x"]
print(rx.shape, t.shape, ex.shape)      # (128, T) (T,) (128,)
```

## 6. Try a real phantom
```bash
python run_relay.py --source generic --max-scatterers 12000     # about 1 minute
python run_relay.py --source abdomen --max-scatterers 12000     # about 3 minutes
```
Presets available: abdomen, brain, breast, cardiac, carotid, generic, kidney, lung, msk, obstetric, prostate, thyroid.

## 7. Colab
Open `integration/01_ap_ti_relay/relay_demo.ipynb`, uncomment the `git clone` line in the first cell, set your repository URL, run all cells.

## 8. Where next
| If you want to... | Read |
|---|---|
| use your own points or images | [02_USER_GUIDE.md](02_USER_GUIDE.md) |
| consume the output as the Receiver Beamforming team | [10_RECEIVER_BF_HANDOFF.md](10_RECEIVER_BF_HANDOFF.md) |
| understand the maths | [04_PHYSICS_AND_MATH.md](04_PHYSICS_AND_MATH.md) |
| something went wrong | [12_TROUBLESHOOTING_FAQ.md](12_TROUBLESHOOTING_FAQ.md) |
