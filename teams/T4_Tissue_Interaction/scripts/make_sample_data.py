"""Regenerate sample_data/: label PNGs + mapping JSON + colour/intensity examples."""
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import tissue_interaction as ti

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(OUT, exist_ok=True)

for bp, sp in [("abdomen", 0.25), ("thyroid", 0.1), ("generic", 0.15)]:
    ph = ti.make_phantom(bp, sp)
    Image.fromarray(ph.index.astype(np.uint8), "L").save(f"{OUT}/phantom_{bp}_labels.png")
    json.dump({"spacing_mm": [ph.dx * 1e3, ph.dz * 1e3],
               "labels": {str(i): t.name for i, t in enumerate(ph.tissue_list)}},
              open(f"{OUT}/phantom_{bp}_mapping.json", "w"), indent=2)

# colour-coded example (thyroid)
ph = ti.make_phantom("thyroid", 0.1)
pal = [(0, 0, 0), (255, 220, 180), (255, 255, 0), (200, 80, 80), (255, 0, 0), (255, 255, 255),
       (0, 120, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
rgb = np.zeros(ph.index.shape + (3,), np.uint8)
cmap = {}
for i, t in enumerate(ph.tissue_list):
    rgb[ph.index == i] = pal[i]
    cmap["#%02x%02x%02x" % pal[i]] = t.name
Image.fromarray(rgb, "RGB").save(f"{OUT}/example_color_phantom.png")
json.dump({"spacing_mm": 0.1, "colors": cmap}, open(f"{OUT}/example_color_mapping.json", "w"), indent=2)

# grey-level "anatomical" example: classes by impedance, blurred + noisy
order = np.argsort([t.impedance for t in ph.tissue_list])
grey = np.zeros(ph.index.shape, float)
for rank, i in enumerate(order):
    grey[ph.index == i] = 15 + rank * (225 / max(len(order) - 1, 1))
rng = np.random.default_rng(1)
grey = ndimage.gaussian_filter(grey, 0.8) + rng.normal(0, 3, grey.shape)
Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8), "L").save(f"{OUT}/example_intensity_phantom.png")
json.dump({"spacing_mm": 0.1, "classes": [ph.tissue_list[i].name for i in order]},
          open(f"{OUT}/example_intensity_mapping.json", "w"), indent=2)
print("sample_data written")
