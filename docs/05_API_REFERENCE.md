# API Reference

Import setup when running outside `integration/01_ap_ti_relay/`:
```python
import sys; sys.path[:0] = ["<repo>/integration/01_ap_ti_relay"]
import _paths                      # adds teams/T3_*/code and teams/T4_*/src
```

## 1. `usoundsim_relay` (integration layer)

### `RelayConfig`  (dataclass)
| Field | Type, default | Meaning |
|---|---|---|
| `n_elements` | int, `config.N_ELEMENTS` (128) | array size |
| `pitch_m` | float, `config.PITCH` (0.154 mm) | element spacing |
| `f0_hz` | float, `config.F0` (5e6) | centre frequency |
| `fs_hz` | float, `config.FS` (40e6) | sampling rate |
| `c0` | float, `config.C` (1540) | nominal speed of sound [m/s] used for delays |
| `alpha0` | float, `config.ALPHA_0` (0.5) | AP uniform attenuation [dB/MHz/cm], used only if `attenuation_owner="ap"` |
| `n_cycles` | int, `ap.N_CYCLES` (3) | pulse cycles |
| `time_margin_fraction` | float, 0.15 | extra time after the last echo |
| `attenuation_owner` | `"tissue"` or `"ap"`, `"tissue"` | who applies attenuation |
| `tau_tx` | ndarray or None | per-element transmit delay [s], shape (n_elements,) |
| `max_scatterers` | int, 8000 | diffuse scatterer budget in Tissue Interaction |
| `seed` | int, 0 | random seed for speckle |
| `also_ti_channel` | bool, True | also run TI stage 3 as an alternative echo model |

### `make_ti_config(rc, **overrides) -> TissueInteractionConfig`
Builds the Tissue Interaction config from `rc`: same f0, c0, fs, array (linear, `n_elements`, `pitch_m`), `path_model = "full"` if the owner is `"tissue"` else `"none"`. Extra keyword arguments are set as attributes on the config (for example `spreading_exponent=1.0`). Raises `ValueError` for an unknown owner.

### `run_relay(source, rc=None, *, mapping=None, spacing_mm=None, background="generic_soft", units="mm", outdir=None, verbose=True, **ti_overrides) -> RelayResult`
| Parameter | Meaning |
|---|---|
| `source` | `Phantom` object, preset name (`"generic"`, `"abdomen"`, ...), path to points file (`.csv .txt .npz .npy`), `PointCloud`, `np.ndarray` (N x 2 or 3 or 4: x, z[, amp]), `dict`, or an image path (needs `mapping`) |
| `rc` | `RelayConfig`; default `RelayConfig()` |
| `mapping`, `spacing_mm` | for image sources and tissue clouds |
| `background` | tissue name for path attenuation of a reflector list |
| `units` | units of point coordinates: `"m"`, `"mm"`, `"cm"`, `"um"` |
| `outdir` | if given, writes the hand-over files ([06_DATA_DICTIONARY.md](06_DATA_DICTIONARY.md)) |
| `verbose` | print one line per runner |
| `**ti_overrides` | forwarded to `make_ti_config` |
Raises `ValueError` for unknown owner or a source that cannot be interpreted.

### `RelayResult`  (dataclass)
| Field | Shape / type | Meaning |
|---|---|---|
| `cfg` | `TissueInteractionConfig` | config used |
| `rc` | `RelayConfig` | settings used |
| `phantom` | `Phantom` or None | None for reflector lists |
| `scatterers` | `ScattererSet` | `.xyz` (N,3) m, `.amp`, `.kind`, `.n`, `.row_labels()` |
| `element_x` | (M,) | element x positions [m] |
| `t_axis` | (T,) | seconds, starts at about -1.2 us |
| `incident` | (N, T) float32 | runner 2 output |
| `reflected` | (N, T) float32 | runner 3 output |
| `received_pressure` | (M, T) float64 | runner 4 output, hand-off to Receiver BF |
| `pulse_t`, `pulse_p` | (96,) | the transmit pulse |
| `ti_channel`, `ti_channel_t` | (M, T') or None | TI stage 3 alternative and its time axis (T' differs from T) |
| `log` | list of dict | runner name, owner, array shapes handed over |
| `files` | dict | paths written when `outdir` is set |
| `timings` | dict | seconds per runner: `runner1_ti_stage1`, `runner2_ap_part1`, `runner3_ti_stage2`, `runner4_ap_part2`, `extra_ti_stage3` |

### `usoundsim_relay.verify` (check tools, not the BF team's code)
| Function | Description |
|---|---|
| `das_image(rx, t_axis, element_x, xs, zs, c=1540.0)` | delay-and-sum envelope image on grid `xs` (m, lateral) by `zs` (m, depth), plane-wave transmit model; returns (len(zs), len(xs)) |
| `locate(rx, t_axis, element_x, x0, z0, half_x=1.5e-3, half_z=1.2e-3, step=0.04e-3, c=1540.0)` | position (x, z) of the brightest point near (x0, z0) |

## 2. `nitk_usoundsim` (Module 01, Acoustic Propagation)

### Added for the integration (vectorised)
**`forward_propagation_many(scatterer_x, scatterer_z, element_x, t_axis, f0=F0, c=C, alpha0=ALPHA_0, n_cycles=N_CYCLES, tau_tx=None, dtype=np.float32, chunk=128, support_sigma=5.0) -> (N, T)`**
Incident pressure at N scatterers. `support_sigma` is the half-width, in Gaussian sigmas, within which each pulse is evaluated.

**`return_propagation_many(reflected, t_axis, scatterer_x, scatterer_z, element_x, f0=F0, c=C, alpha0=ALPHA_0, chunk=256, rel_threshold=1e-6) -> (M, T)`**
Received pressure at M elements from N reflected waveforms. `reflected` is (N, T), row i = scatterer i. Raises `ValueError` if row count and scatterer count differ.

### Original functions (unchanged)
| Function | Purpose |
|---|---|
| `generate_pulse(fs, f0, n_cycles) -> (t, p)` | transmit pulse samples |
| `create_linear_array(n_elements=N_ELEMENTS, pitch=PITCH) -> x` | element x positions, centred on 0 |
| `calculate_tx_distance(element_x, x_s, z_s)`, `calculate_rx_distance(...)` | distance element to scatterer |
| `calculate_propagation_delay(distance, c)` | distance / c |
| `calculate_attenuation(distance, f0, alpha0)` | amplitude factor, 10^(-alpha0 f_MHz d_cm / 20) |
| `forward_propagation(scatterer, element_x, t_axis, f0, c, alpha0, n_cycles, tau_tx)` | one scatterer; returns a 3-tuple whose first item is the incident wave on `t_axis` |
| `mock_tissue_interaction(incident_wave, scatterer)` | stand-in reflector (scales by `amp`) |
| `return_propagation(reflected_wave, t_axis, scatterer, element_x, f0, c, alpha0)` | one scatterer; 3-tuple whose first item is pressure (M, T) |
| `simulate_acoustic_propagation(phantom_points, n_elements, pitch, f0, fs, c, alpha0, n_cycles, time_margin_fraction) -> dict` | the original standalone pipeline with the mock reflector |

### `config.py` constants
See [07_CONFIGURATION_REFERENCE.md](07_CONFIGURATION_REFERENCE.md).

## 3. `tissue_interaction` (Module 02) functions used by the relay
| Function | Role in the relay |
|---|---|
| `make_phantom(name)` | preset phantom |
| `load_phantom(src, mapping=, spacing_mm=)` | image phantom |
| `load_points(src, units=)` | points file to `PointCloud` |
| `tissue_cloud_to_phantom(pc, spacing_mm, background, None)` | tissue cloud to phantom |
| `build_scatterers(phantom, cfg)` | runner 1 for phantoms |
| `scatterers_from_points(pc, cfg, background)` | runner 1 for reflector lists |
| `compute_reflected_pressure(scat, incident, t_axis, cfg)` | runner 3 |
| `compute_channel_data(scat, reflected, t_axis, cfg, element_xz=)` | alternative stage 3 |
| `save_scatterers`, `save_reflected` | write `01_` and `03_` files |
Full documentation: `teams/T4_Tissue_Interaction/README.md` and `docs/`.

## 4. Error messages you may meet
| Message (start) | Cause |
|---|---|
| `attenuation_owner must be 'tissue' or 'ap'` | typo in `RelayConfig.attenuation_owner` |
| `incident must have shape (N, n_samples)` | rows of `incident` do not match the scatterer list (TI) |
| `reflected has N rows but M scatterer positions were given` | mismatch passed to `return_propagation_many` |
| `max_scatterers=... limits speckle density` (warning) | fewer diffuse scatterers than needed for fully developed speckle; raise `--max-scatterers` |
