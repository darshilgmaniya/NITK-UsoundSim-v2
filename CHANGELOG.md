# Changelog

## Live file upload
- `usoundsim_pipeline/upload.py`: Upload button in the notebook (VS Code / Jupyter / Colab), file dialog and menu in `run_pipeline.py` (`--upload`, or no arguments). Accepts image + mapping `.json`, `.csv`, `(N, 3)` `.npy`. `sample_abdomen` / `sample_thyroid` also work in the CLI. 8 new tests.

## Structure change - teams T1 to T8
- `modules/` is now `teams/` with one folder per team, like the first project: T1 Transducer, T2 TX Beamforming, T3 Acoustic Propagation, T4 Tissue Interaction, T5 RX Beamforming, **T6 Image Reconstruction (new)**, T7 B-mode, T8 Post-Image Processing.
- New T6 stage between beamforming and B-mode (DC removal + zero-phase band-pass). Pipeline now has 8 runners; saved files are `06_recon_rf.npz`, `07_bmode_outputs.npz`, `08_final_image.npz`. `--no-recon` / `ReconConfig(enabled=False)` switches it off.
- T1 and T2 code copied from the first project with notebooks. In the v2 pipeline the probe comes from T3's config.
- One notebook per team folder and per integration step (see NOTEBOOKS.md), VS Code settings in `.vscode/`.

## 2026-10-14 (v2) - Integration step 3: B-mode, scan conversion and despeckling; ONE command from any phantom to the final image
This repository is **v2**, a separate copy of the project. The first project (`NITK-UsoundSim`, 64/128-element simulator with its own `teams/` folders) is not touched.
### Added
- `teams/T7_B_mode/`: Hilbert envelope + log compression (runner 6). Copied from the first project's `bmode_formation.py`; only change: the `config` import became a module constant (`DEFAULT_DYNAMIC_RANGE_DB = 50`). 3 tests.
- `teams/T8_Post_Image_Processing/`: scan conversion + guided-filter despeckling + quality metrics (runner 7). Copied from the first project's `post_processing.py`; changes: the `config` import became `DEFAULT_OUTPUT_SIZE`, and the default depth range is the whole `z_axis` (the beamformer now chooses the depth per phantom). 5 tests.
- `integration/03_full_pipeline/`: `run_pipeline(...)` and CLI `run_pipeline.py`. Calls `run_full` (runners 1-5) unchanged, then runners 6 and 7. Accepts every phantom input of the relay: presets, CSV points, image phantoms with a mapping file. Options for probe (`--f0-mhz`, `--pitch-mm`, `--n-elements`), receiver, dynamic range, despeckle. `--fit-aperture` sizes an image phantom to the array. 6 tests.
- `docs/17_FULL_PIPELINE.md`.
### Unchanged
- Modules 01, 02, 03 and integrations 01, 02 are the updated team code, byte for byte (their 25 + 43 + 23 + 7 + 14 tests pass).
### Notes
- Default probe is the updated Acoustic Propagation config: 128 elements, 5 MHz, pitch lambda/2 = 0.154 mm (aperture 19.6 mm). It is NOT the first project's Philips L12-4 setting (8 MHz, 0.30 mm). With 0.30 mm pitch at 8 MHz the pitch is 1.56 lambda and grating-lobe (X-shaped) artefacts appear.
- A phantom wider than the array is clipped to the aperture (the beamformer prints a note).
- The TI stage-3 comparison channel is off by default in the pipeline (`--with-ti-channel` turns it on); it is not used by the image.

## 2026-10-14 - Integration step 2: Receiver Beamforming joins the relay
### Added
- `teams/T5_RX_Beamforming/`: delay-and-sum beamformer (`das_beamform`, same call signature as the team's script), `beamform()`, input validation, preview B-mode, 23 tests, explanation document.
- The Receiver BF team's `liver_echo_simulation.py` kept unmodified in `demos/`; `receive_beamforming.py` shim lets it run.
- `integration/02_full_relay/`: runner 5 on top of integration 01, CLI `run_full_relay.py`, plug-in example for the team's own beamformer, 14 tests, `verification/run_verification.py`, notebook.
- `docs/16_RECEIVER_BEAMFORMING_AND_FULL_RELAY.md`; new sections in docs 06, 08, 10; `05_beamformed_outputs.npz` data contract.
### Notes
- The team's real `receive_beamforming.py` was not available; the module's `das_beamform` is a re-implementation matching the call signature.
- No change to modules 01 and 02 or to integration 01.

## 2026-10-13 - Integration step 1: Acoustic Propagation + Tissue Interaction
### Added
- `integration/01_ap_ti_relay/`: four-runner relay (AP forward -> TI -> AP return -> Receiver BF input), CLI, notebook, tests.
- `forward_propagation_many()` / `return_propagation_many()` in Acoustic Propagation (vectorised; verified equal to the single-scatterer originals).
### Documentation
- `docs/`: 15 numbered documents (getting started, user guide, architecture, physics, API, data dictionary, configuration, verification report, performance, Receiver BF hand-off, developer guide, troubleshooting, decision log, limitations, glossary), migration guide, consolidated PDF report, `CONTRIBUTING.md`.
- `integration/01_ap_ti_relay/verification/run_verification.py` reproduces every reported number; `examples/load_and_beamform.py` for the Receiver BF team.
- `RelayResult.timings` added (seconds per runner).
### Changed
- **Acoustic Propagation: `N_ELEMENTS` 64 -> 128** (`config.py`). Labels, docstrings, `.npz` descriptions and Team 3's runner/parameters follow the config. Pitch unchanged (lambda/2 = 0.154 mm).
- Team 3 `output/` regenerated for 128 elements.
### Unchanged
- `teams/T4_Tissue_Interaction/` is the original Subtask 04 repository, byte for byte (its CI file moved to the repo root).
### Known
- `Team_3_Acoustic_Propagation_Explanation.pdf` still describes the 64-element version.
