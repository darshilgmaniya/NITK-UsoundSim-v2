# T6 - Image Reconstruction (runner 6)

`image_reconstruction(beamformed_rf, z_axis, c=1540, f0_hz=5e6, config=ReconConfig()) -> ReconResult`

Comes **after Receive Beamforming (T5)** and **before B-mode Formation (T7)**.
DC removal per scan line, zero-phase band-pass around the centre frequency (default 2-8 MHz at 5 MHz), Hilbert envelope (for inspection; T7 recomputes it), depth axis passed through.
Scan conversion stays in T8 (it needs the final pixel grid).

Source: the team's sample notebook `UsoundIMGRECsample` (steps 2-4), written as a function so it can sit in the pipeline.
`ReconConfig(enabled=False)` turns the stage off and the pipeline output is then identical to the version without T6.

Test: `python -m pytest -q tests`. Notebook: `notebooks/T6_image_reconstruction.ipynb`.
