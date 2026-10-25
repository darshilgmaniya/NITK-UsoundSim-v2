# Configuration Reference

## 1. Where settings live
| Layer | File / object | Controls |
|---|---|---|
| Array and medium (single source of truth) | `teams/T3_Acoustic_Propagation/code/nitk_usoundsim/config.py` | element count, pitch, f0, fs, c, attenuation, pulse |
| Relay | `RelayConfig` (integration) | defaults to the values above; adds attenuation owner, transmit delays, budgets |
| Tissue Interaction | `TissueInteractionConfig` built by `make_ti_config` | derived from `RelayConfig`; you rarely edit it |
Rule: to change the array, edit `config.py` (or pass a `RelayConfig`); do not edit the Tissue Interaction defaults.

## 2. `config.py` (Acoustic Propagation)
| Constant | Value | Unit | Meaning |
|---|---|---|---|
| `C` | 1540.0 | m/s | speed of sound used for all AP delays |
| `ALPHA_0` | 0.5 | dB/MHz/cm | uniform attenuation coefficient |
| `F0` | 5.0e6 | Hz | centre frequency |
| `FS` | 40.0e6 | Hz | sampling frequency |
| **`N_ELEMENTS`** | **128** | - | number of elements (**was 64**) |
| `LAMBDA` | C / F0 = 0.308 mm | m | wavelength |
| `PITCH` | LAMBDA / 2 = 0.154 mm | m | element spacing |
| `N_CYCLES` (in `acoustic_propagation.py`) | 3 | - | pulse cycles |
| `X_MIN, X_MAX` | -0.010, +0.010 | m | lateral field of view used by the original standalone demo |
| `Z_MIN, Z_MAX` | 0, 0.040 | m | depth field of view of the demo |
| `N_SCANLINES` | 64 | - | number of image lines in the demo (not the element count) |
| `PHANTOM_POINTS` | 5 wire points | - | default phantom of the standalone demo |
| `DYNAMIC_RANGE_DB`, `OUTPUT_SIZE` | 50, (512, 512) | dB, px | image display settings of the demo |

Derived: aperture = (N_ELEMENTS - 1) x PITCH = 19.56 mm. The relay does not use the field-of-view constants.

## 3. `RelayConfig`
See the field table in [05_API_REFERENCE.md](05_API_REFERENCE.md#relayconfig--dataclass).

## 4. Tissue Interaction fields set by the relay
| Field | Value from relay |
|---|---|
| `f0_hz`, `c0`, `fs_hz` | from `RelayConfig` |
| `probe` | `ProbeConfig(geometry="linear", n_elements, pitch_m)` |
| `path_model` | `"full"` (owner tissue) or `"none"` (owner ap) |
| `max_scatterers`, `seed` | from `RelayConfig` |
Other fields keep Tissue Interaction defaults (for example `spreading_exponent`, directivity, texture settings); see `teams/T4_Tissue_Interaction/configs/default.yaml` and `docs/`.

## 5. Which settings interact
| If you change | Also check |
|---|---|
| `F0` | `PITCH` follows automatically (lambda/2); `FS` should stay at least 4 x the highest content (about f0 + 2 MHz) |
| `N_ELEMENTS` | aperture vs the phantom width; compute time grows linearly |
| `ALPHA_0` | has no effect when `attenuation_owner="tissue"` |
| `C` | the relay uses one c for all delays; tissue-specific c appears only in the TI stage 3 alternative |
| `FS` | memory and time scale linearly with it |

## 6. Recommended presets
| Use | Settings |
|---|---|
| Fast check | `--source points`, defaults |
| Realistic phantom | `--source generic --max-scatterers 12000`, owner `tissue` |
| Reproduce original Team 3 behaviour | `--attenuation-owner ap`, reflector list |
| Resolution study | `--n-elements 64` vs `128` on `points` |
| Low memory | `--max-scatterers 2000`, `also_ti_channel=False` |
