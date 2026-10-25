# Limitations, Risks and Roadmap

## 1. Known limitations
| # | Limitation | Effect | Where it comes from |
|---|---|---|---|
| L1 | 2-D only (x, z; y = 0) | no elevation focusing or out-of-plane scatter | both modules |
| L2 | AP uses one speed of sound (1540 m/s) | no refraction or aberration in the main output; delays slightly wrong in tissue with different c | AP model |
| L3 | AP has no element directivity or 1/r spreading | wide-angle and deep echoes over-weighted vs reality; edge scatterers (array is +-9.78 mm) | AP model |
| L4 | Linear array only in the AP part | no curvilinear or phased-sector geometry in the main path | AP model |
| L5 | Unfocused transmit | lower SNR and ghost arcs for a plane-wave beamformer | Team 3 default |
| L6 | No noise, no probe bandwidth limit other than the pulse | optimistic images | scope |
| L7 | Arbitrary pressure units | cannot compare against measured volts or pascals | scope |
| L8 | Linear acoustics | no harmonics, nonlinearity, or multiple scattering | both modules |
| L9 | Diffuse scatterer budget | few scatterers give weaker speckle statistics; a warning appears | TI |
| L10 | Cost grows with N x M x T | abdomen at 12,000 scatterers takes minutes and 2 GB+ | implementation |
| L11 | No independent validation against Field II, k-Wave or a physical phantom | accuracy is verified internally (V1 to V6) | scope |
| L12 | FOV constants in `config.py` (+-10 mm) are narrower than many TI phantoms | the relay ignores them; only the aperture limits what is insonified | config |
| L13 | Team 3's explanation PDF still describes 64 elements | documentation mismatch | not regenerated |

## 2. Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Teams disagree on attenuation owner or final baton (D4, D5) | medium | rework of one runner | decide at the next team meeting; both options already implemented |
| Different time-axis conventions in later modules | medium | misaligned echoes | contract: shared `t_axis`, time zero at pulse centre ([06_DATA_DICTIONARY.md](06_DATA_DICTIONARY.md)) |
| Memory limits on student laptops for large phantoms | high | crashes | defaults of 8,000 scatterers, guidance in [09_PERFORMANCE.md](09_PERFORMANCE.md), Colab notebook |
| Silent change of row order when someone filters scatterers | low | wrong images | row-order tests; never filter one array without the others |

## 3. Roadmap
| Step | Item | Owner |
|---|---|---|
| Next | Confirm D4 and D5 with Tissue Interaction and Acoustic Propagation teams | integration |
| Next | Receiver Beamforming module as `teams/T5_*` and `integration/02_*` | Receiver BF team |
| Next | Transmit Beamforming module feeding `tau_tx` | Transmit BF team |
| Soon | Add element directivity and 1/r spreading to AP part 2 (or adopt TI stage 3) | AP team |
| Soon | Regenerate Team 3 explanation PDF for 128 elements | AP team |
| Later | Numba or multiprocessing for AP part 2 | integration |
| Later | Electronic noise and ADC model | new module |
| Later | Validation against Field II / k-Wave on wire and cyst phantoms | integration |
| Later | 3-D phantoms | TI team |
