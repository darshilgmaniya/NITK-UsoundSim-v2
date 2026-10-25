# Verification and Validation Report - Integration 01 (AP + TI relay, 128 elements)

| Item | Value |
|---|---|
| Scope | `integration/01_ap_ti_relay` with `teams/T3_Acoustic_Propagation` (128 el) and `teams/T4_Tissue_Interaction` |
| Date | 2026-10-13 |
| Reproduce | `cd integration/01_ap_ti_relay && python verification/run_verification.py` (writes `docs/verification/results.json` and the two figures) |
| Setup | 128 elements, pitch 0.154 mm (lambda/2), f0 5 MHz, fs 40 MHz, c 1540 m/s, 3-cycle Gaussian pulse, unfocused transmit |

## 1. Summary
| # | Question | Result | Verdict |
|---|---|---|---|
| V1 | Do the vectorised AP functions equal the originals? | max relative error 6.5e-7 (forward), 6.8e-7 (return) | pass |
| V2 | Do targets appear where they were placed? | max error 20 um lateral, 40 um axial (128 el) | pass |
| V3 | Does 128 elements improve lateral resolution? | lateral FWHM 1.7x to 1.9x finer than 64 elements | pass |
| V4 | Is the chain linear in scatterer strength? | max deviation 5e-8 | pass |
| V5 | Is attenuation counted exactly once? | owner "ap": 0.1000 (= -20 dB expected); owner "tissue": 1.0000 | pass |
| V6 | Do the two echo paths (AP part 2 vs TI stage 3) agree? | image correlation 0.980, peak positions identical | pass |
| V7 | Is it fast and light enough? | 9.6k scatterers: about 33 s, 1.6 GB peak | pass (see [09_PERFORMANCE.md](09_PERFORMANCE.md)) |
| T | Automated tests | 25 (AP) + 43 (TI) + 7 (relay) = 75 passed | pass |

## 2. Test inventory
| Suite | Location | Count | What it protects |
|---|---|---|---|
| Acoustic Propagation | `teams/T3_Acoustic_Propagation/code/nitk_usoundsim/test_acoustic_propagation.py` | 22 | pulse shape, distances, delays, attenuation, forward/return maths, geometry |
| Batch equivalence | `.../test_batch_equivalence.py` | 3 | vectorised = original; array has 128 elements |
| Tissue Interaction | `teams/T4_Tissue_Interaction/tests/` | 43 | phantoms, scatterers, interaction, file formats, CLI |
| Relay | `integration/01_ap_ti_relay/tests/test_relay.py` | 7 | 128 elements end to end, contract shapes and order, focusing, TI stage 3 position, attenuation modes, linearity, files |

## 3. Results in detail

### V1 - Batch equivalence
40 random scatterers (x in +-8 mm, z in 5 to 35 mm, random strengths), 128 elements. Reference = original single-scatterer functions summed in a loop.

| Function | Max abs error | Signal peak | Relative error |
|---|---|---|---|
| `forward_propagation_many` | 6.4e-6 | 9.78 | 6.5e-7 |
| `return_propagation_many` | 8.8e-6 | 12.99 | 6.8e-7 |

The small difference comes from evaluating each pulse only within +-5 Gaussian sigmas (cut at about e^-12.5 of the peak) and trimming reflected waveforms below 1e-6 of their peak.

### V2 - Focus accuracy (9 wire targets, x = -6, 0, 6 mm and z = 10, 20, 30 mm)
Delay-and-sum of `received_pressure` (check beamformer in `verify.py`), peak searched on a 40 um grid.

| Array | Max lateral error | Max axial error | RMS position error |
|---|---|---|---|
| 128 elements | 20 um | 40 um | 27 um |
| 64 elements | 100 um | 40 um | 55 um |

All errors are within one search-grid step (40 um) except the 64-element lateral case at the edge targets, where the wider beam makes the peak flatter. The test threshold is 250 um.

### V3 - Resolution, 64 versus 128 elements
Single-target -6 dB (half-maximum) widths of the delay-and-sum envelope.

| Depth | Lateral FWHM 64 el | Lateral FWHM 128 el | Ratio | Axial FWHM 64 el | Axial FWHM 128 el |
|---|---|---|---|---|---|
| 10 mm | 0.40 mm | 0.23 mm | 1.74 | 0.54 mm | 0.53 mm |
| 20 mm | 0.78 mm | 0.40 mm | 1.92 | 0.51 mm | 0.55 mm |
| 30 mm | 1.14 mm | 0.59 mm | 1.94 | 0.57 mm | 0.55 mm |

Reading: the aperture doubles (9.7 mm to 19.6 mm), so lateral width roughly halves. Axial width does not change, because it is set by the pulse length, not the array. For orientation, lambda x z / D gives 0.31 mm at z = 20 mm for 128 elements, the same order as the measured 0.40 mm. The 128-element beam also has lower side lobes (figure below).

![resolution](verification/resolution_64_vs_128.png)

### V4 - Linearity
Scaling all scatterer strengths by 0.1 and by 10 scales `received_pressure` by the same factor, maximum relative deviation 5.0e-8 and 3.3e-8 (float32 rounding).

### V5 - Attenuation counted exactly once
One scatterer at z = 40 mm, AP attenuation coefficient changed from 0 to 0.5 dB/MHz/cm.

| `attenuation_owner` | Received peak ratio (0.5 over 0) | Expected |
|---|---|---|
| `"ap"` | 0.1000 | 0.1000 (two-way 2 x 0.5 x 5 x 4 = 20 dB) |
| `"tissue"` | 1.0000 | 1.0000 (AP forced to 0, TI owns attenuation) |

### V6 - Two echo paths
AP part 2 (uniform c, no spreading or directivity) and TI stage 3 (tissue-aware delays, spreading, directivity) on the 9 wire targets. Delay-and-sum image correlation 0.980; peak positions differ by 0 um. They are two models of the same echo, not two copies: use one.

## 4. Requirements trace
| Requirement | Evidence |
|---|---|
| R1 Array has 128 elements | config, `test_element_count_is_128`, `test_128_elements_everywhere`, output shape (128, T) |
| R2 Modules run standalone | each suite runs from its own folder in CI |
| R3 One baton per hand-over with fixed row order | `test_row_order_and_shapes_are_the_contract`, files `01_` to `04_` |
| R4 No double attenuation | V5 |
| R5 Result is physically placed correctly | V2, `test_targets_focus_at_true_coordinates` |
| R6 Original AP behaviour unchanged | the 22 original AP tests still pass; batch functions are additions |

## 5. Not covered
- Absolute pressure calibration (output is arbitrary units).
- Comparison against an independent simulator (Field II, k-Wave) or measured data.
- Focused transmit, curvilinear or phased geometries in the AP part (AP is linear-array only).
- Elevation (3-D) effects; everything is 2-D.
See [14_LIMITATIONS_AND_ROADMAP.md](14_LIMITATIONS_AND_ROADMAP.md).


---

# Integration 02 - Receiver Beamforming and the full relay (added 2026-10-14)

Reproduce with `python integration/02_full_relay/verification/run_verification.py` (about 1 minute; add `--presets` for the body-part runs).
Raw numbers: `docs/verification/results_integration02.json`. Unit tests: module 03 has 23, integration 02 has 14.

## I2-1  Focusing accuracy, nine wire targets (128 elements, Hann, F# 1.0)
Largest distance between true target position and the brightest image sample: **0.036 mm lateral, 0.010 mm in depth**
(the image sample spacing is 0.154 mm laterally and 0.019 mm in depth).

## I2-2  Resolution and side lobes, one wire at z = 20 mm (line spacing 0.02 mm)
| Apodisation / F-number | Lateral FWHM [mm] | Axial FWHM [mm] | Highest lateral side lobe [dB] |
|---|---|---|---|
| hann / F#=1 | 0.641 | 0.548 | -36.8 |
| hann / F#=2 | 1.259 | 0.547 | -40.4 |
| hamming / F#=1 | 0.589 | 0.548 | -30.5 |
| hamming / F#=2 | 1.144 | 0.548 | -41.5 |
| blackman / F#=1 | 0.734 | 0.548 | -41.6 |
| blackman / F#=2 | 1.447 | 0.547 | -42.5 |
| tukey / F#=1 | 0.508 | 0.549 | -14.5 |
| tukey / F#=2 | 0.989 | 0.548 | -25.6 |
| rect / F#=1 | 0.404 | 0.550 | -12.6 |
| rect / F#=2 | 0.764 | 0.549 | -13.8 |

Read: Hann trades a wider main lobe for much lower side lobes than the rectangular window; a larger F-number narrows the active aperture and widens the main lobe.

## I2-3  Linearity through all five runners
Doubling the amplitude of one scatterer doubles the beamformed output exactly: ratio **2.000000**.

## I2-4  Array size
| Elements | Aperture half-width [mm] | Wire targets inside aperture | Max lateral error [mm] | Max depth error [mm] |
|---|---|---|---|---|
| 64 | 4.85 | 3 of 9 | 0.077 | 0.030 |
| 96 | 7.32 | 9 of 9 | 0.036 | 0.010 |
| 128 | 9.78 | 9 of 9 | 0.036 | 0.010 |

With 64 elements the aperture is only +-4.85 mm, so the wires at x = +-6 mm are outside the array. By default the image is limited to the aperture, and the beamformer reports this in `BeamformResult.notes` (override with `RxConfig(x_range_m=...)`; image quality at the aperture edge is lower).

## I2-5  File path equals memory path
Beamforming from `receiver_bf_input.npz` instead of the in-memory baton differs by at most 4.5e-08 (the file stores float32).

## I2-6  Input matrix (every case ran to a finite, non-blank image)
Body-part presets used a budget of 1500 scatterers requested (the generator still adds interface scatterers); timings are for a single CPU core and include all five runners.

| Source | Scatterers | Channel data | Image (depth x lines) | Time [s] |
|---|---|---|---|---|
| points (9 wires) | 9 | 128 x 2166 | 1715 x 85 | 4.9 |
| CSV x,z,amp | 3 | 128 x 2223 | 1819 x 52 | 3.4 |
| CSV x,z,tissue (0.5 mm grid) | 8213 | 128 x 2726 | 2208 x 114 | 29.1 |
| image phantom + mapping (0.2 mm) | 22835 | 128 x 5711 | 4312 x 127 | 89.3 |
| single scatterer | 1 | 128 x 977 | 676 x 7 | 0.1 |
| shallow target (2-3 mm) | 2 | 128 x 810 | 312 x 13 | 0.1 |
| 64 / 96 elements | 9 | 64 x 2047 / 96 x 2102 | 1715 x 64 / 1715 x 85 | 0.4 / 0.7 |
| attenuation owner = AP | 9 | 128 x 2166 | 1715 x 85 | 1.4 |
| generic | 3078 | 128 x 4864 | 3794 x 127 | 17.2 |
| abdomen | 9211 | 128 x 8387 | 6387 x 127 | 58.5 |
| kidney | 6453 | 128 x 8135 | 6387 x 127 | 34.5 |
| thyroid | 3834 | 128 x 3110 | 2233 x 127 | 18.2 |
| breast | 3801 | 128 x 3686 | 2753 x 127 | 35.6 |
| cardiac | 7950 | 128 x 9264 | 7431 x 127 | 88.4 |
| musculoskeletal | 3667 | 128 x 3318 | 2493 x 127 | 16.3 |
| obstetric | 9484 | 128 x 10013 | 7945 x 127 | 103.6 |
| brain | 6183 | 128 x 6106 | 4830 x 127 | 75.9 |
| lung | 3159 | 128 x 3645 | 2753 x 127 | 22.0 |

Not tested here: real (non-simulated) channel data, steered or compounded transmit, curved arrays, 3-D.
