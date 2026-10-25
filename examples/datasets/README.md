# Trying a downloaded dataset

The pipeline needs a **phantom**: a picture where every pixel says which tissue it is (plus a mapping JSON).
Real ultrasound pictures are not phantoms, but the **segmentation masks** that come with public datasets are.

## BUS-UCLM (breast ultrasound, 683 images, masks included)
Download from https://data.mendeley.com/datasets/7fvgj4jsp7/3 and unzip. Take any file from the masks folder
(RGB: black = normal tissue, green = benign lesion, red = malignant lesion).

```
cd integration/03_full_pipeline
python run_pipeline.py --source /path/to/mask.png --mapping ../../examples/datasets/bus_uclm_mapping.json --fit-aperture
```
`bus_uclm_mapping.json` maps black to `breast_glandular` and green/red to `tumor_hypoechoic`.
Edit it to use other tissues (names: `python -c "import tissue_interaction as ti; print([t.name for t in ti.list_tissues()])"` after adding the paths, or see `teams/T4_Tissue_Interaction`).

If the picture is bigger than the probe aperture (19.6 mm) keep `--fit-aperture`: it rescales the pixel size so the phantom fills 95 % of the aperture.
Pixel colours must match the mapping exactly, so save the mask as PNG (not JPG).
