# Migration Guide: 64 to 128 elements

## What changed
| Item | Before | After |
|---|---|---|
| `config.N_ELEMENTS` | 64 | **128** |
| Aperture (pitch 0.154 mm) | 9.70 mm | 19.56 mm |
| `received_pressure` shape | (64, T) | (128, T) |
| `element_x` length | 64 | 128 |
| Plot titles, docstrings, `.npz` descriptions | "64" hard-coded | read `config.N_ELEMENTS` |
| Team 3 `run_team3.py`, parameters JSON, README | 64 | 128 |
| Team 3 `output/` | 64-element results | regenerated for 128 |
Not changed: `N_SCANLINES = 64` (image lines), pitch rule (lambda/2), pulse, sampling, attenuation.

## Checklist for code that consumed the old output
- [ ] Do not hard-code 64: use `received_pressure.shape[0]` or `n_elements` from the file.
- [ ] Element positions: read `element_x` (centred, increasing).
- [ ] Apodisation windows and any per-element arrays must have 128 entries.
- [ ] Channel loops and memory buffers: size by element count.
- [ ] Amplitudes: the sum of 128 unnormalised pulses is larger; for the nine wire targets the received peak rises 1.7x (7.3 to 12.3) and rms 1.6x (0.65 to 1.05). The factor depends on geometry; re-check thresholds.
- [ ] Image width: lateral FWHM improves by 1.7x to 1.9x; re-tune display and metrics.
- [ ] Compute time of AP parts: about 1.8x.
- [ ] Regenerate any saved `.npz` produced with 64 elements; old files have shape (64, T).

## Code changes made
1. `config.py`: `N_ELEMENTS = 128`.
2. `acoustic_propagation.py`, `demo_acoustic_propagation.py`: documentation and labels made dynamic.
3. `run_team3.py`, `input/team3_run_parameters.json`, `README.md` of module 01: 64 replaced by 128.
4. `forward_propagation_many`, `return_propagation_many` added (not a 64/128 matter, but needed for real phantoms).

## How to go back
Set `N_ELEMENTS = 64` in `config.py` or run `run_relay.py --n-elements 64`. Everything else follows the config.

## Measured effect
See [08_VERIFICATION_REPORT.md](08_VERIFICATION_REPORT.md), V2 and V3.

![resolution](verification/resolution_64_vs_128.png)
