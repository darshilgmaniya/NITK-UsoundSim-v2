# 17 - Full pipeline: any phantom to the final image

Integration step 3 adds two runners after the Receiver Beamformer and one command that runs the whole chain.

| Runner | Module | Input | Output |
|---|---|---|---|
| 1-4 | `integration/01_ap_ti_relay` | phantom | `received_pressure` (128, T) |
| 5 | `teams/T5_RX_Beamforming` | `received_pressure` | `beamformed_rf` (depth, lines), `z_axis`, `x_lines` |
| 6 | `teams/T7_B_mode` | `beamformed_rf` | `bmode` (depth, lines) in [0, 1] |
| 7 | `teams/T8_Post_Image_Processing` | `bmode`, `x_lines`, `z_axis` | `image` (H, W) scan-converted, `final` (H, W) despeckled, `x_img`, `z_img` |

## Data contract of the new batons
`06_bmode_outputs.npz`: `bmode` (depth, lines), `z_axis` (depth,) [m], `x_lines` (lines,) [m].
`07_final_image.npz`: `image`, `final` (H, W) in [0, 1], `x_img` (W,), `z_img` (H,) [m]. Row = depth, column = lateral. Pixels are square (about 0.04 to 0.3 mm depending on the phantom).

## Why scan conversion is only a resample
A linear array has parallel vertical scan lines, so (depth x line) is already a Cartesian image. Runner 7 resamples it onto square pixels with bilinear interpolation, using the real axes, never sample indices.

## Why a time-axis rule does not apply in runners 6 and 7
Delay-to-sample lookups happen only in the beamformer, with `np.interp` on the real `t_axis` (which starts before 0). Runners 6 and 7 work on beamformed data that already has physical axes.

## Despeckling
Self-guided filter (He et al. 2013), radius 4 pixels, eps 0.001, from the first project's notebook where it was chosen out of ten classical filters on real clinical images. Parameters were tuned on breast images, not on this simulator. `--despeckle none` skips it.

## Known limits
- Phantoms wider than the array are clipped to the aperture. Use `--fit-aperture`, `--spacing-mm` or a wider probe.
- A pitch above lambda gives grating-lobe artefacts (seen with 0.30 mm at 8 MHz). The default 0.154 mm at 5 MHz does not.
- Straight rays, constant speed of sound in the beamformer: no refraction or shadow modelling at the image stage.
- Speckle statistics depend on `--max-scatterers`: below about 6 scatterers per resolution cell the speckle is under-developed (the tissue module warns).
