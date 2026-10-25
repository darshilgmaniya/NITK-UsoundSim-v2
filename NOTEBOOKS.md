# Notebooks (run in VS Code, Jupyter or Colab)

Same structure as the first project: one folder per team in `teams/`, each with its own notebook, plus an integration notebook for every relay step.
Each notebook starts with a "Step 0" cell that finds the project root and adds every team folder to `sys.path`, so nothing needs to be installed except `requirements.txt`.

| Notebook | What it runs |
|---|---|
| `teams/T1_Transducer/notebooks/T1_transducer.ipynb` | T1: material registry, probe geometry |
| `teams/T2_TX_Beamforming/notebooks/T2_tx_beamforming.ipynb` | T2: transmit delays, apodization, element pulses |
| `teams/T3_Acoustic_Propagation/notebooks/T3_acoustic_propagation.ipynb` | T3: pulse, propagation, `pressure[element, time]` |
| `teams/T4_Tissue_Interaction/notebooks/T4_tissue_interaction.ipynb` | T4: phantom to scatterers to reflected pressure |
| `teams/T5_RX_Beamforming/notebooks/T5_rx_beamforming.ipynb` | T5: channel data to beamformed RF |
| `teams/T6_Image_Reconstruction/notebooks/T6_image_reconstruction.ipynb` | T6: DC removal, band-pass, envelope |
| `teams/T7_B_mode/notebooks/T7_bmode_formation.ipynb` | T7: envelope, log compression |
| `teams/T8_Post_Image_Processing/notebooks/T8_post_image_processing.ipynb` | T8: scan conversion, despeckle |
| `integration/01_ap_ti_relay/relay_demo.ipynb` | T3 + T4 (the four relay runners) |
| `integration/02_full_relay/full_relay_demo.ipynb` | T3 + T4 + T5 (phantom to beamformed image) |
| `integration/03_full_pipeline/full_pipeline_demo.ipynb` | **Everything**: any phantom to final despeckled image (runners 1 to 8) |

## VS Code, step by step
1. Install Python 3.10+ and VS Code. Install the **Python** and **Jupyter** extensions (the folder suggests them).
2. `File > Open Folder...` and choose the `NITK-UsoundSim-v2` folder (the one that contains `.vscode`).
3. Terminal > New Terminal, then:
   ```
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Open a notebook from the table. Top right click **Select Kernel > Python Environments > .venv**.
5. Click **Run All** (or `Shift+Enter` cell by cell).

### Upload your own phantom while the code runs
- **Notebook** (`full_pipeline_demo.ipynb`): run the cell *Upload your own phantom*. An **Upload** button appears (Colab opens its own dialog). Select the phantom **and its mapping file together**, then run the next cell and everything below it. Nothing uploaded = the default `SOURCE` is used.
- **Terminal**: `python run_pipeline.py --upload` opens a file dialog; `python run_pipeline.py` with no arguments shows a menu (upload / sample / demo / preset).
- Accepted: image (`.png` `.jpg`) + mapping `.json`, a points `.csv` (`x_mm,z_mm,amp` or `x_mm,z_mm,tissue`), or an `(N, 3)` `.npy`. Uploaded files are copied to an `uploads/` folder.

To change the phantom without uploading, edit `SOURCE` in the settings cell (preset name, CSV, or an image plus a mapping JSON).

Colab: upload the zip, run `!unzip -q NITK-UsoundSim-v2.zip -d /content`, open any notebook; Step 0 finds the project.
