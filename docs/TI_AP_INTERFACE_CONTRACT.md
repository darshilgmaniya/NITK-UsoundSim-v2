# Interface contract - Tissue Interaction <-> Acoustic Propagation

The Tissue Interaction module sits between the two halves of the Acoustic Propagation module:

```
Transducer -> Transmit BF -> Acoustic Propagation  forward_propagation()
                                     |  scatterer positions (x, z)  ^  incident pressure at each scatterer
                                     v                              |
                          +--------------------------+              |
   user phantom --------> |   TISSUE INTERACTION     | -------------+
   (png / npy / ...)      |  stage 1: phantom -> scatterer list      |
                          |  stage 2: incident -> reflected pressure |
                          +--------------------------+
                                     | reflected_pressure_input  (n_scatterers, n_samples)
                                     v
                  Acoustic Propagation  return_propagation()  -> received_pressure at the RX elements
                                     v
                       Receiver BF -> Image Reconstruction -> B-mode -> Post-processing
```

Tissue Interaction does **not** do any per-element delay-and-sum. One reflected waveform per scatterer is
produced; `return_propagation()` takes care of delays and spreading to the receive elements.

## Stage 1 - phantom -> scatterer list  (`tissue_scatterers.npz`)

Produced by `run_stage1()` / `python -m tissue_interaction scatterers ...`.

| key | shape | meaning |
|---|---|---|
| `scatterer_xyz` | (N, 3) | metres. columns `x` (lateral, 0 = phantom centre), `y` (= 0, 2-D), `z` (depth, 0 = skin) |
| `scatterer_xyz_columns` | (3,) | `['x','y','z']` |
| `scatterer_xyz_units` | () | `'m'` |
| `scatterer_amp` | (N,) | signed intrinsic reflectivity (pressure ratio), **before** path effects |
| `scatterer_kind` | (N,) | 0 = diffuse (speckle), 1 = interface (specular) |
| `scatterer_row_labels` | (N,) | e.g. `s00012:I:liver>bile` |
| `scatterer_tof_s` | (N,) | one-way time of flight with the **local** speed of sound of the phantom |
| `sc_*`, `tissue_names`, `config_json` | | everything needed to reload the set (`load_scatterers`) |

**Row order is the contract.** Row *i* of the incident-pressure array you hand back, and row *i* of the
reflected-pressure output, always refer to scatterer *i* of this file.

`tissue_property_maps.npz` additionally exports `speed_m_s`, `density_kg_m3`,
`attenuation_db_cm_at_f0`, `impedance_rayl`, `backscatter_db`, `tissue_index`, `x_axis_m`, `z_axis_m`
- Acoustic Propagation can use the sound-speed and attenuation maps for a heterogeneous forward model.

## Stage 2 - incident pressure -> reflected pressure  (`tissue_interaction_output.npz`)

Input (from Acoustic Propagation `forward_propagation()`): `incident_pressure` `(N, n_samples)` and `t_axis`
`(n_samples,)` in seconds, rows in stage-1 order. (Other key names: pass `--incident-key` / `key=`.)

Output key names deliberately copy the mock that `return_propagation()` already consumes:

| key | shape | meaning |
|---|---|---|
| `reflected_pressure_input` | (N, n_samples) float32 | **drop-in replacement for the mock** |
| `t_axis`, `t_axis_units` | (n_samples,), () | same time axis as the incident pressure, `'s'` |
| `reflected_pressure_input_row_labels` | (N,) | scatterer labels |
| `reflected_pressure_input_shape_meaning` | () | `'(n_scatterers, n_samples)'` |
| `scatterer_xyz`, `scatterer_xyz_columns`, `scatterer_xyz_units` | | same as stage 1 |
| `scatterer_kind`, `tissue_names`, `scatterer_tissue_idx`, `path_model`, `f0_hz` | | extras |

## Things to confirm with the Acoustic Propagation owner

1. **Attenuation / shadowing - who applies it?**  Default `path_model: full` applies two-way,
   frequency-dependent attenuation and interface transmission loss (shadowing behind bone, gas, lung)
   *inside* this module, so the result is realistic even if the forward model assumes a uniform medium.
   If Acoustic Propagation already includes tissue attenuation, set `path_model: none` to avoid counting it twice.
2. **Units.** Positions are in metres, time in seconds. Pressure units follow whatever the incident array uses.
3. **Straight-ray path model.** Path effects use a straight line from the transducer
   (`linear`: vertical, `sector`: radial from the apex). Give the real geometry via the `probe:` block of the config.
4. **Time axis.** `t_axis` must be uniformly sampled and long enough to contain the incident pulse at the deepest scatterer.
5. **Memory.** The array is `N x n_samples x 4 bytes` (30 000 x 4 600 = 550 MB). Use `out=` with a memmap or process the
   scatterers in batches if the machine is small.

## Stage 3 - per-element echoes for Receiver Beamforming  (`receiver_bf_input.npz`)

| key | shape | meaning |
|---|---|---|
| `received_pressure` | (n_elements, n_samples) float32 | echo recorded at each receive element |
| `received_pressure_shape_meaning` | () | `'(n_elements, n_samples)'` |
| `t_axis`, `t_axis_units`, `sampling_rate_hz` | (n_samples,), (), () | seconds; time zero = start of transmission |
| `element_x`, `element_z`, `element_xz`, `element_x_units` | (n_el,), (n_el,), (n_el, 2), () | element positions in metres |
| `f0_hz`, `c0_nominal_m_s`, `probe_geometry`, `probe_name` | | nominal acquisition parameters |
| `scatterer_xyz*`, `path_model`, `config_json` | | provenance |

Use **either** stage 2 (`reflected_pressure_input` -> Acoustic Propagation `return_propagation()`) **or** stage 3 (this file ->
Receiver Beamforming). Never feed one into the other: both already include the return path.

## Input from Acoustic Propagation Part 1 (`forward_propagation()`)

Read by `load_ap_forward()` / `from-ap`. Required: `incident_pressure (N, n_samples)`, `scatterer_xyz (N, 3)`, a time axis
(`t_axis`, or `pulse_t` / `--fs-mhz`). Used when present: `incident_pressure_row_labels`, `pulse_t`, `pulse_p`, `element_x`
(+ `element_x_units`), `scatterer_xyz_units`, `scatterer_amp`. Missing units are assumed to be metres and reported.
Row *i* of the incident pressure must belong to `scatterer_xyz[i]`; the output keeps that order and the row labels.
