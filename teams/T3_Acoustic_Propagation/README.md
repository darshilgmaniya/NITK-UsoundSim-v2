# Team 3: Acoustic Propagation (NITK-UsoundSim)

> **Update (Oct 2026): array enlarged from 64 to 128 elements** (`N_ELEMENTS` in `code/nitk_usoundsim/config.py`).
> Also added: `forward_propagation_many()` / `return_propagation_many()` (vectorised, checked against the
> originals by `test_batch_equivalence.py`). The explanation PDF in this folder still describes the 64-element
> version; every other file and the `output/` folder reflect 128 elements.

This folder shows the work of Team 3. Their code models how an ultrasound pulse travels
through tissue. Every element of a linear array sends a short Gaussian pulse. The code
works out the distance and travel time from each element to a point scatterer
(t = d / c, with c = 1540 m/s) and weakens each pulse by the tissue attenuation
(0.5 dB/MHz/cm). It adds all the pulses to get the wave that hits the scatterer, then sends
the echo back to every element. The result is a pressure array `pressure[element, time]`.
In the full simulator this step sits between transmit beamforming and receive beamforming.

A full explanation (theory, every function, input, output, questions) is in
**`Team_3_Acoustic_Propagation_Explanation.pdf`**.

## Folder contents

| Item | What it is |
|---|---|
| `run_team3.py` | The one script to run. It only calls the team's functions and does not change their code. |
| `code/nitk_usoundsim/` | The team's original code, copied unchanged: `acoustic_propagation.py`, `config.py`, `demo_acoustic_propagation.py`, `test_acoustic_propagation.py`, `check_arr.py`, `__init__.py`, `requirements.txt`, plus the `.npz`/`.png` files the team saved earlier |
| `input/team3_run_parameters.json` | Sample input: array, pulse and scatterer settings for the two runs |
| `output/` | Everything that `run_team3.py` makes |
| `requirements.txt` | Python packages needed |
| `Team_3_Acoustic_Propagation_Explanation.pdf` | The explanation document |

## Requirements

- Python 3.9 or newer
- `numpy` and `matplotlib` (listed in `requirements.txt`)

## How to run

1. Install Python 3 from https://www.python.org. On Windows, tick **"Add Python to PATH"** during installation.
2. Open a terminal in this folder (`Team_3_Acoustic_Propagation`).
   - macOS: right-click the folder in Finder and choose **New Terminal at Folder**.
   - Windows: open the folder, type `cmd` in the address bar and press Enter.
3. Install the packages:
   - macOS / Linux: `python3 -m pip install -r requirements.txt`
   - Windows: `python -m pip install -r requirements.txt`
4. Run the demo:
   - macOS / Linux: `python3 run_team3.py`
   - Windows: `python run_team3.py`
5. Open the `output/` folder to see the results.

### Optional: the team's own commands

The team's docstrings say to run their commands "from the usound_sim folder". In this
package, that folder is **`code/`** (the folder that contains `nitk_usoundsim/`):

```
cd code
python3 -m unittest nitk_usoundsim.test_acoustic_propagation -v
python3 -m nitk_usoundsim.demo_acoustic_propagation
```

Note: when you run the demo this way, it overwrites its PNG and `.npz` files inside
`code/nitk_usoundsim/`. `run_team3.py` does not do this. It sends all its files to `output/`.

## What you will see

The console shows four steps and a summary:

1. **Unit tests**: the team's 22 tests run, and the result is `22 of 22 tests passed -> ALL PASSED`.
2. **Team demo**: 128 elements, 5 MHz, lambda/2 pitch, with scatterers at 10, 20 and 30 mm. The round trips are
   12.987, 25.974 and 38.961 us, and they match 2z/c.
3. **Final probe**: the Philips L12-4 model (128 elements, 0.30 mm pitch, 8 MHz) with a scatterer at (0, 20) mm.
   The one-way time is **12.99 to 17.94 us** (centre element to edge element), and the round trip is
   25.97 to 35.87 us.
4. **Attenuation** at 20 mm depth: 4, 8 and 12 dB one way at 4, 8 and 12 MHz.

Files in `output/`:

| File | Meaning |
|---|---|
| `demo_geometry_pulse_delay_attenuation.png` | Team demo: array and scatterers, the 5 MHz pulse, delay vs depth, attenuation vs depth |
| `demo_pipeline_signals.png` | Team demo: incident wave, reflected wave, echo on element #64, echo image for all 128 elements |
| `part1_forward_propagation_outputs.npz` | Team demo data: incident pressure at each scatterer vs time |
| `part2_return_propagation_outputs.npz` | Team demo data: received pressure on all 128 elements vs time (the final module output) |
| `L12-4_final_probe_propagation.png` | Final probe run: pulse, incident wave at (0, 20) mm, travel time vs element, echo on all 128 elements |
| `L12-4_final_probe_outputs.npz` | Final probe run data (time axis, waves, delays, received pressure 128 x time) |
| `attenuation_vs_depth_4_8_12MHz.png` | Loss in dB and round-trip amplitude factor vs depth at 4, 8 and 12 MHz |
| `unit_tests_full_output.txt` | Full text printed by the unit tests |
| `console_log.txt` | Everything printed on the console |
| `console_screenshot.png` | The console output as a picture |

## Running the individual code files

`run_team3.py` runs everything at once. To run the team's own files one by one:

`nitk_usoundsim` is a Python package, so run its files from the `code/` folder (`cd code`) with `-m`:

| File | Command | What it does |
|---|---|---|
| `nitk_usoundsim/demo_acoustic_propagation.py` | `python3 -m nitk_usoundsim.demo_acoustic_propagation` | Team's own demo: saves the two demo figures and the part1/part2 `.npz` files in `code/nitk_usoundsim/`. |
| `nitk_usoundsim/test_acoustic_propagation.py` | `python3 -m unittest nitk_usoundsim.test_acoustic_propagation -v` | The 22 unit tests. |
| `nitk_usoundsim/check_arr.py` | `python3 nitk_usoundsim/check_arr.py` | Prints the arrays saved in `part2_return_propagation_outputs.npz`. It opens that file by the path `nitk_usoundsim/...`, so it must be started from `code/` (the VS Code Run button starts it from the wrong folder). |
| `nitk_usoundsim/acoustic_propagation.py`, `nitk_usoundsim/config.py` | `python3 -m nitk_usoundsim.acoustic_propagation` | Library modules used by the demo and the tests. |

In VS Code, open this team folder (File > Open Folder) or the whole `NITK-UsoundSim_Team_Projects` folder, pick a Python interpreter that has the packages from `requirements.txt` (bottom-right corner of VS Code), then open a file and press the Run button. The `.vscode/settings.json` file sets the import paths, so the files below run as they are.

## Expected runtime

About **2 to 5 seconds** in total on a MacBook (the script itself reports about 1.5 to 2 s).

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3` or `python`: command not found | Install Python. On Windows you can also try `py run_team3.py`. |
| `ModuleNotFoundError: numpy` / `matplotlib` | Run `python3 -m pip install -r requirements.txt` with the same Python you use to run the script. |
| `ModuleNotFoundError: nitk_usoundsim` | Run the command from inside `Team_3_Acoustic_Propagation`, and keep `code/` next to `run_team3.py`. |
| No plot windows open | This is normal. The plots are saved as PNG files in `output/`. |
| Permission error when writing `output/` | Copy the folder to a place where you can write, such as Desktop or Documents. |
