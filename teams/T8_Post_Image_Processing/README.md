# Module 05 - Scan conversion and post-processing (runner 7)

```python
from post_processing_stage import scan_convert, postprocess
image, x_img, z_img = scan_convert(bmode, x_lines, z_axis)     # resample onto square display pixels
final = postprocess(image)                                     # guided-filter despeckling (r=4, eps=0.001)
```
Also: `speckle_index`, `cnr`, `gcnr`, `edge_sharpness`, `lesion_ring`, `to_uint8` (quality metrics).

Source: the first project's `teams/T7_Image_Recon_Post_Processing/post_processing.py`. Changes: `config.OUTPUT_SIZE` became
`DEFAULT_OUTPUT_SIZE = (512, 512)`; the default depth range is the whole `z_axis` (the first project fixed it to 0-40 mm).
The module docstring has the algorithm, the filter choice and the limitations. Needs `opencv-python` (see `requirements.txt` of the repo).

Test: `python -m pytest -q tests` (5 tests): axes and square pixels, default depth range, shape check, speckle index drops, bad arguments.
