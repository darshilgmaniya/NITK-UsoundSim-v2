# Data Dictionary - hand-over files

Written by `run_relay(..., outdir=...)` / `run_relay.py`. All are NumPy `.npz` files (`np.load(path)`). Shapes below are from a real run of nine wire targets (N = 9, T = 2166, M = 128). Text entries load as 0-d string arrays; use `str(d["key"])`.

## Conventions
| Item | Rule |
|---|---|
| Positions | metres; `scatterer_xyz` columns x (lateral), y (always 0), z (depth) |
| Time | seconds; `t_axis` uniform, step 1/fs (25 ns), starts at about -1.2 us; time zero = centre of the transmit pulse at an element with zero delay |
| Pressure | arbitrary units, linear in scatterer strength |
| Row order | row i of every (N, T) array = scatterer i of `scatterer_xyz` |
| Element order | row m of every (M, T) array = element m, `element_x` increasing from -9.78 mm to +9.78 mm |

## 01_tissue_scatterers.npz  (runner 1, Tissue Interaction)
| Key | Type, shape | Description |
|---|---|---|
| `scatterer_xyz` | float64 (N,3) | positions [m] |
| `scatterer_xyz_columns`, `scatterer_xyz_units` | str | `x y z`, `m` |
| `scatterer_amp` | float32 (N,) | signed reflectivity (negative flips polarity) |
| `scatterer_kind`, `scatterer_kind_meaning` | int8 (N,), str | 0 diffuse (speckle), 1 interface (specular) |
| `scatterer_row_labels` | str (N,) | label per row |
| `scatterer_tof_s` | float32 (N,) | tissue-aware time of flight (used by stage 3 only) |
| `tissue_names` | str (K,) | tissues present |
| `f0_hz`, `config_json` | float, str | settings used |
| `sc_*` (`sc_xyz`, `sc_amp`, `sc_kind`, `sc_tissue_idx`, `sc_tissue_idx_pre`, `sc_scatter_exp`, `sc_path_atten_np`, `sc_path_y_eff`, `sc_path_tx_loss`, `sc_tof`, `sc_path_len`, `sc_incid_angle_deg`) | arrays (N,) | the internal scatterer state needed to rebuild the set |

## 02_part1_forward_propagation_outputs.npz  (runner 2, Acoustic Propagation)
| Key | Type, shape | Description |
|---|---|---|
| `incident_pressure` | float32 (N,T) | pressure arriving at each scatterer |
| `incident_pressure_shape_meaning`, `incident_pressure_row_labels` | str | `(n_scatterers, n_time_samples)`, labels |
| `t_axis`, `t_axis_units` | float64 (T,), str | time axis, `s` |
| `scatterer_xyz` (+ columns, units) | float64 (N,3) | positions |
| `pulse_t`, `pulse_p` | float64 (96,) | transmit pulse |
| `element_x`, `element_x_units` | float64 (M,), str | element positions |
| `description` | str | one-line purpose |

## 03_tissue_interaction_output.npz  (runner 3, Tissue Interaction)
| Key | Type, shape | Description |
|---|---|---|
| `reflected_pressure_input` | float32 (N,T) | reflected / scattered pressure at each scatterer |
| `..._shape_meaning`, `..._units`, `..._row_labels` | str | metadata |
| `t_axis`, `t_axis_units` | float64 (T,), str | same axis as file 02 |
| `scatterer_xyz` (+ columns, units), `scatterer_kind`, `scatterer_tissue_idx`, `tissue_names` | | scatterer context |
| `path_model` | str | `full` or `none` |
| `f0_hz` | float | centre frequency |

## 04_part2_return_propagation_outputs.npz  (runner 4, Acoustic Propagation)
All keys of `receiver_bf_input.npz` below, plus `reflected_pressure_input` (float32 (N,T)) and `reflected_pressure_input_row_labels`.

## receiver_bf_input.npz  (the hand-off to Receiver Beamforming)
| Key | Type, shape | Description |
|---|---|---|
| **`received_pressure`** | float32 (128, T) | pressure at each receive element vs time |
| `received_pressure_shape_meaning`, `received_pressure_units`, `received_pressure_row_labels` | str | `(128 rx_elements, n_time_samples)`, `arbitrary pressure units (a.u.)`, `rx_element_000` ... |
| `t_axis`, `t_axis_units`, `sampling_rate_hz` | float64 (T,), str, float | time axis, `s`, 40e6 |
| `element_x`, `element_z`, `element_x_units` | float64 (128,) | element positions (z = 0), `m` |
| `n_elements`, `pitch_m`, `f0_hz`, `c0_nominal_m_s` | int, float | 128, 0.000154, 5e6, 1540 |
| `attenuation_owner` | str | `tissue` or `ap` |
| `scatterer_xyz` (+ columns, units) | float64 (N,3) | ground-truth positions (for validation only; a real receiver does not know them) |
| `description` | str | purpose |

## alt_ti_stage3_channel_data.npz  (alternative echo model, optional)
| Key | Type, shape | Description |
|---|---|---|
| `received_pressure` | float32 (128, T') | echo with tissue-aware delays, spreading loss, element directivity |
| `t_axis` | float64 (T',) | its own time axis; **T' differs from T** (3053 vs 2166 in the example) |
| `element_x`, `description` | | |
Use this file or `receiver_bf_input.npz`, never both.

## Size
Storage is about `4 bytes x N x T` per (N,T) array: N = 9,592, T = 4,938 gives 190 MB per array uncompressed. The hand-off file is small (128 x T x 4 bytes, about 2.5 MB at T = 4,900).

## Compatibility rules
1. New keys may be added; existing keys never change meaning or shape convention.
2. Anything that changes shape order or units needs a version note in `CHANGELOG.md` and a notice to the receiving team.
3. The per-scatterer files (01 to 03) share row order; never sort one without the others.


---

## File 05 - `05_beamformed_outputs.npz` (Receiver Beamforming -> B-mode image formation)
Written by `integration/02_full_relay` and by `receiver_beamforming.save_beamformed()`.

| Key | Shape | Unit | Meaning |
|---|---|---|---|
| `beamformed_rf` | (n_depth, n_lines) | a.u. | signed RF after delay-and-sum; **the data to hand on** |
| `z_axis` | (n_depth,) | m | depth, spacing c / (2 fs) |
| `x_lines` | (n_lines,) | m | lateral position of each line |
| `envelope_preview` | (n_depth, n_lines) | a.u. | abs(Hilbert) of `beamformed_rf`, preview only |
| `bmode_db_preview` | (n_depth, n_lines) | dB | log-compressed preview, 0 to `-dynamic_range_db` |
| `dynamic_range_db` | scalar | dB | preview dynamic range |
| `apodization`, `f_number` | text, scalar | | receive settings used (`f_number` = -1 means full aperture) |
| `sampling_rate_hz`, `c0_nominal_m_s`, `f0_hz`, `n_elements` | scalar | | copied from the hand-off |
