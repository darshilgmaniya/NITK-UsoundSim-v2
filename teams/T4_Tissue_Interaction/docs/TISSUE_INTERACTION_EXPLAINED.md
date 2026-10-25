# Tissue Interaction module - what it is, what goes in, what comes out, how it works

*(Subtask 04 of NITK-UsoundSim. Written so you can explain it at a review, to the Receiver Beamforming team, or in the project report.)*

---

## 1. The big picture

The team is building an end-to-end, in-silico ultrasound scanner in Python. The signal flows through seven stages:

| # | Team | Question this stage answers | Main output |
|---|---|---|---|
| 1 | Transducer | How does the probe turn electrical excitation into sound (and back)? | acoustic wave / echo |
| 2 | Transmit Beamforming | Which delays/apodisation focus or steer the transmitted beam? | transmitted beam |
| 3 | Acoustic Propagation | How does that wave travel through the medium to the tissue? | wave at the tissue |
| **4** | **Tissue Interaction (you)** | **What does the tissue do to the wave - how much is reflected / scattered, and what arrives back at the probe?** | **echo signals** |
| 5 | Receiver Beamforming | How are the element signals delayed and summed into focused lines? | beamformed RF |
| 6 | Image Reconstruction | Envelope, scan conversion | reconstructed image |
| 7 | B-mode formation + Post-processing | Log compression, display, denoising | final image |

**Your job in one sentence:** *take a description of the body (phantom) and the sound wave that reaches it, and produce the echoes that the probe's elements would record.* Everything the Receiver Beamforming team does (focusing, delay-and-sum) starts from your output.

Two physical mechanisms make echoes, and both are in your module:

* **Specular reflection** at boundaries between two tissues of different acoustic impedance (organ walls, vessel walls, bone surface, gas pockets). Strong, directional.
* **Diffuse (Rayleigh-type) scattering** from microscopic structure inside a tissue. Weak, random - this is what produces the *speckle* texture of a real ultrasound image.

---

## 2. What you are doing (your role, in plain words)

1. **Describe the medium** - represent the patient/phantom as a set of coordinates, each with a tissue type (or acoustic properties).
2. **Turn tissue into acoustics** - look up density, speed of sound, impedance, attenuation and backscatter for every tissue (library of about 40; custom tissues allowed).
3. **Create scatterers** - place reflectors along every tissue boundary (specular) and random scatterers inside tissues (speckle).
4. **Apply what the tissue does to the wave** - reflection coefficients, attenuation that grows with frequency and depth, shadows behind bone/gas, tissue-dependent time of flight.
5. **Compute the reflected pressure** of every scatterer from the incident pressure that reaches it.
6. **Propagate the echoes back to the receive elements** (delay, spreading, element directivity) and sum them into per-element signals.
7. **Hand over** the per-element signals to Receiver Beamforming, plus 2-D grayscale images for checking and for the report.

---

## 3. Inputs (what goes in)

### 3.1 The phantom - now supplied as POINTS / COORDINATES

Units: x lateral, z depth (z = 0 is the probe face / skin, depth grows downward), default unit millimetres (`--units m|mm|cm|um`).

| Input style | File columns | Meaning | What the module does |
|---|---|---|---|
| **Tissue cloud** | `x_mm, z_mm, tissue` (or per-point `density, speed, atten_db_cm_mhz[, backscatter_db...]`) | "at this coordinate there is liver / bone / water ..." | Rasterises the cloud (nearest point wins) -> full physics: boundaries, speckle, heterogeneous attenuation, shadows |
| **Scatterer list** | `x_mm, z_mm, amp[, tissue]` | "a reflector of strength `amp` sits here" (the (x, z, amp) format used by Acoustic Propagation) | Uses the points as given; path loss through one homogeneous background tissue |
| Image / array (still supported) | label PNG, colour PNG, grey-level image, `.npy` | segmentation or anatomical image + pixel size | Same as tissue cloud after pixel -> tissue mapping |

Accepted containers: CSV (header row), `.npz`, `.npy` (N x 2/3/4), Python dict/array. Sample files: `UsoundSim_Points_Dataset/`.

### 3.2 The real flow: the Acoustic Propagation `forward_propagation()` file  *(main input)*

The Acoustic Propagation team (Part 1) delivers the incident mechanical pressure that reaches **each scatterer** from the 64-element TX array. The module reads that one file and needs nothing else:

| key in the AP file | used for |
|---|---|
| `incident_pressure (n_scatterers, n_samples)` | the wave arriving at each scatterer; **row i belongs to `scatterer_xyz[i]`** |
| `scatterer_xyz` (+ `_columns`, `_units`) | the scatterer positions (these *are* the scatterers of this run, same order) |
| `incident_pressure_row_labels` | copied unchanged to the output rows |
| `t_axis` (name auto-detected; or `pulse_t` / `--fs-mhz`) | time axis, sampling rate |
| `pulse_t`, `pulse_p` | centre frequency (FFT peak of the TX pulse) -> attenuation and scattering are evaluated at the right f0; pulse is also forwarded to the output |
| `element_x` | element geometry -> used as the **receive** elements (assumes the same 64-element array receives) |
| `scatterer_amp` (optional) | scatterer reflectivity; if absent a unit reflector (`--default-amp`) is used, or the tissue's own reflectivity with `--amp-from-tissue` and a medium |

Tissue enters in two optional ways: nothing (homogeneous soft-tissue background, `--background`), or a tissue cloud / phantom (`--medium-points`, `--medium-phantom`) so that attenuation, shadows and speeds follow the real tissue layout between probe and scatterer.

```bash
python -m tissue_interaction from-ap --forward forward.npz --out outputs/echo                       # homogeneous medium
python -m tissue_interaction from-ap --forward forward.npz --medium-points points_abdomen.csv --body-part abdomen --out outputs/echo
```
```python
res = ti.run_from_ap("forward.npz", phantom=None)       # or phantom=ti.tissue_cloud_to_phantom(ti.load_points("pts.csv"))
res.channel_rf  # (n_elements, n_samples) -> Receiver Beamforming
```

*Note on tiny numbers:* values such as `-4.8e-142` in the rows are just the far tails of the Gaussian TX pulse (the pulse has not reached that scatterer yet). They are valid and are handled as is.

If the AP file does not exist yet, the module can generate a mock incident pulse (or `ti.emulate_ap_forward(...)` writes a file with the AP keys for testing).

`incident_pressure` `(n_scatterers, n_samples)` and `t_axis` (s). Row *i* = pressure waveform arriving at scatterer *i*.
If you do not have it yet, the module generates a **mock incident pulse** (Gaussian-modulated sinusoid, tissue-aware arrival time, spreading loss) so you can test alone.

### 3.3 Probe / configuration (from the Transducer + Transmit BF teams)

Centre frequency `f0`, sampling rate `fs` (use >= 10 x f0), F-number, probe geometry (`linear`, or `sector` with apex position; apex behind the face = curvilinear), number of receive elements (default 64), pitch (default 0.3 mm), or explicit element positions (`--elements`). Everything is in one YAML/Python config; body-part presets give sensible defaults.

---

## 4. Outputs (what comes out)

| File | Goes to | Content |
|---|---|---|
| **`receiver_bf_input.npz`** | **Receiver Beamforming** | `received_pressure (n_elements, n_samples)`, `t_axis`, `sampling_rate_hz`, `element_x`, `element_z`, `f0_hz`, `c0_nominal_m_s`, probe info |
| `tissue_interaction_output.npz` | Acoustic Propagation (only if its `return_propagation()` stays in the chain) | `reflected_pressure_input (n_scatterers, n_samples)` + labels |
| `tissue_scatterers.npz` | Acoustic Propagation (to compute the incident pressure) | `scatterer_xyz`, amplitudes, kinds, time of flight |
| `tissue_property_maps.npz` | Acoustic Propagation | sound speed, density, attenuation, impedance maps |
| `images/*.png`, `*_montage.png` | report / checking | 2-D grayscale maps (tissue labels, density, speed, impedance, attenuation, backscatter, boundary reflectivity, scatterer strength, echo energy, channel-data envelope) |

> **Do not propagate twice.** If the Acoustic Propagation team's `return_propagation()` still takes the per-scatterer reflected waveforms to the elements, give *them* `tissue_interaction_output.npz`. If the Receiver Beamforming team consumes your output directly, give *them* `receiver_bf_input.npz`. Never feed one into the other.

![channel data](images/channel_data_point_targets.png)

*15 point targets: each target becomes a hyperbola across the elements - exactly what a beamformer expects.*

---

## 5. How the model runs (step by step)

```
points / coordinates ──► [A] input adapter ──► Phantom grid (tissue per cell)
                                   │
                                   ▼
                         [B] acoustic maps  (rho, c, Z, alpha, backscatter)
                                   │
                                   ▼
                         [C] scatterer generation  (interface + diffuse)
                                   │   + path integration from the probe (attenuation, shadow, time of flight)
                                   ▼
        incident pressure ─► [D] reflected pressure per scatterer
                                   │
                                   ▼
                         [E] return path to the elements ─► received_pressure (n_elements, n_samples)
```

**[A] Input adapter.** Points are read in the given units. A tissue cloud is rasterised on a regular grid (spacing = median nearest-neighbour distance unless given); cells farther than 2.5 spacings from any point become background. The grid keeps absolute coordinates (x centred on 0, z from 0), so output positions stay in the probe's frame. A scatterer list skips the grid.

**[B] Acoustic maps.** For every tissue: density rho, speed c, impedance Z = rho c, attenuation alpha(f) = alpha0 f^y (dB/cm), backscatter level (dB re liver), backscatter frequency exponent.

**[C] Scatterers.**
* *Interface scatterers*: one every lambda/2 along each boundary. Reflection coefficient with oblique incidence and Snell's law
  R(theta) = (Z2 cos theta1 - Z1 cos theta2) / (Z2 cos theta1 + Z1 cos theta2); total reflection beyond the critical angle; sign kept (soft tissue -> gas flips polarity). A directivity term makes boundaries bright only when they face the probe.
* *Diffuse scatterers*: random points (default 6 per resolution cell) with Gaussian amplitudes scaled by the tissue backscatter level; anechoic tissues (water, cyst, bile, gas) get none; a smooth random field adds organ-scale texture.
* *Path integration* along the straight ray probe -> scatterer: two-way attenuation, interface transmission loss prod(1 - R^2) (shadows), and time of flight sum(ds / c).

**[D] Reflected pressure.** For scatterer *i*:
`reflected_i(t) = amp_i * Tx_i * IFFT{ FFT{incident_i}(f) * (f/f0)^m_i * exp(-L_i (f/f0)^y_i) }`
The frequency-domain terms give frequency-dependent scattering and the spectrum down-shift with depth.

**[E] Return path.** For every scatterer and receive element: travel time r/c (tissue-aware mean speed), spreading (r_ref/r)^p, cosine element directivity, attenuation correction for the real return distance; fractional delays by cubic interpolation; sum over scatterers -> `received_pressure[e, t]`. Time zero = start of transmission.

---

## 6. What it uses

* **Python 3** with **NumPy** (vectorised array math), **SciPy** (`ndimage` for resampling/smoothing, `spatial.cKDTree` for point -> grid, `signal.hilbert`, FFT), **Matplotlib** + **Pillow** (grayscale outputs), **PyYAML** (config). No GPU, no closed-source code.
* **Physics models**: Fresnel/Snell reflection, power-law attenuation, Rayleigh-type scattering, Gaussian-modulated pulse, ray-based path integration, 2-D/3-D spreading law.
* **Engineering**: dataclass configs, CLI (`python -m tissue_interaction ...`), Colab notebook, `pytest` suite (43 tests), GitHub Actions CI.

---

## 7. Why it works for every kind of input

1. **Geometry-agnostic representation.** Whatever the source - segmentation image, CT/MRI grey levels, a list of tissue coordinates, or a bare list of reflectors - it is converted to the same two internal objects: a tissue grid and a scatterer list. Everything after that is identical, so adding a new input type means writing only a small adapter.
2. **Physics lives in the tissue, not in the body part.** A thyroid, a liver and a fetus differ only in which tissues occupy which coordinates and which probe is used. The tissue library (40 tissues: soft organs, fluids, fat, muscle, tendon, bone, gas, lung, tumours, abscess, calcification, microbubble contrast agent) and the 12 body-part presets cover the common cases.
3. **Extensible properties.** Any tissue can be redefined or created (JSON/YAML mapping, per-point density/speed/attenuation columns) - useful for the professor's request to model different tissue densities and to distinguish water, air, tumour and contrast agent.
4. **Unit- and orientation-safe.** `--units` handles m/mm/cm/um; the loader rejects negative depths, unknown tissues, missing scale, and wrong array shapes with explicit error messages instead of silently producing wrong physics.
5. **Probe-agnostic.** Linear, phased and curvilinear geometries and arbitrary element positions are supported through the same config.
6. **Two ways to connect to teammates.** Per-scatterer file for Acoustic Propagation, per-element file for Receiver Beamforming.

---

## 8. How it was checked

* Fresnel coefficient for water -> bone equals the analytic value; soft tissue -> gas gives negative polarity.
* A bone layer produces a shadow (transmission loss, higher attenuation behind it); time of flight uses local speeds.
* Perpendicular boundaries are much brighter than oblique ones; anechoic cysts contain no diffuse scatterers.
* **Beamforming check:** a minimal delay-and-sum applied to the channel data of 9 wire targets focuses every target at its true coordinate (error below the 0.05 mm test grid). This confirms delays, geometry and the time axis (`examples/verify_with_das.py`).
* All 12 body-part presets run end to end; two-stage hand-off reproduces the single-stage result exactly; file formats are tested.

---

## 9. Assumptions and limits (say these before anyone asks)

* 2-D model (the file already has a `y` column for a later 3-D extension - the professor asked for 3-D).
* Straight rays: no refraction, no multiple scattering or reverberation, no harmonic imaging; microbubbles are modelled by raised *linear* backscatter.
* One transmit event per call; loop over events for scanline or compounding sequences.
* Tissue values are textbook mid-range numbers, probe presets are class-typical (modelled loosely on Philips probe families) - **not** calibrated Philips data.
* Scatterer-list input uses a homogeneous background for attenuation; use a tissue cloud when attenuation differs between regions.
* Speckle quality depends on scatterer density; the default (30 000 scatterers) is sparse for a 120 x 120 mm field.
* Memory: `n_scatterers x n_samples x 4 bytes` for the reflected array (about 550 MB at 30 000 x 4 600).

---

## 10. Questions to settle with the Receiver Beamforming team

0. **Attenuation double counting:** does `forward_propagation()` already include tissue attenuation? If yes use `--path-model none`; if it models only delay and spreading (usual for a first version) keep the default `full`.
1. **Array orientation:** `received_pressure` is `(n_elements, n_samples)`. Transpose if they want samples first.
2. **Time zero:** start of transmission (reference of the incident pressure). They need the transmit delay profile / angle from the Transmit BF team to compute transmit time for each pixel.
3. **Sampling rate and units:** SI units (m, s); `sampling_rate_hz` is stored in the file.
4. **Assumed sound speed:** the data carry the phantom's real tissue speeds; the beamformer will normally assume 1540 m/s, which is what produces realistic focusing error in non-uniform tissue.
5. **Element positions:** default 64 elements, pitch 0.3 mm; replace with the Transducer team's `element_x` through `--elements`.
6. **Record length:** the record ends shortly after the last echo; they can zero-pad.

---

## 11. Quick commands

```bash
# phantom as points  ->  channel data for Receiver BF (+ images)
python -m tissue_interaction run --points UsoundSim_Points_Dataset/tissue_clouds/points_abdomen.csv --body-part abdomen --out outputs/abdomen
# wire targets (resolution test for the beamformer)
python -m tissue_interaction run --points UsoundSim_Points_Dataset/scatterer_lists/point_targets.csv --body-part generic --f0-mhz 5 --fs-mhz 60 --out outputs/targets
# two-stage hand-off with Acoustic Propagation
python -m tissue_interaction scatterers --points my_points.csv --body-part kidney --out hand_off
python -m tissue_interaction interact --scatterers hand_off/tissue_scatterers.npz --incident incident.npz --out hand_off
```

```python
import tissue_interaction as ti
cfg = ti.config_for_body_part("thyroid")
res = ti.run_points("points_thyroid.csv", cfg, outdir="outputs/thyroid")   # tissue cloud or scatterer list
res.channel_rf.shape        # (n_elements, n_samples)  -> Receiver Beamforming
```

---

## 12. Glossary

**Phantom** - a digital (or physical) test object with known tissues. **Scatterer** - a point that reflects/scatters sound. **Impedance Z = rho c** - sets how much is reflected at a boundary. **Speckle** - granular texture caused by interference of many tiny scatterers. **Attenuation** - loss of amplitude with distance, growing with frequency. **Channel data** - raw signal of each receive element before beamforming. **F-number** - focal depth / aperture width.
