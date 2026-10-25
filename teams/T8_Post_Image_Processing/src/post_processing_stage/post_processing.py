"""
post_processing.py
==================

T7 image reconstruction + post-processing for the NITK-UsoundSim integration.

    scan_convert(bmode, x_axis, z_axis, ...) -> (image, x_out, z_out)
    postprocess(image, method="guided", r=4, eps=0.001, gamma=1.0) -> image

plus the no-reference / lesion quality metrics used to test it
(speckle_index, cnr, gcnr, edge_sharpness).

Inputs
------
bmode   : (n_depths, n_lines) B-mode brightness in [0, 1] from T6's
          bmode_formation() (log-compressed, already on physical axes).
x_axis  : (n_lines,) scan-line x-positions [m] (BeamformResult.x_lines).
z_axis  : (n_depths,) depths [m] (das_beamform()'s z_axis).

Outputs
-------
scan_convert : image (H, W) in [0, 1] on a uniform grid, plus its axes
               x_out (W,), z_out (H,) [m]. Row = depth, column = lateral.
postprocess  : (H, W) float in [0, 1], despeckled.

Scan conversion (linear array -> rectangular, no fan)
-----------------------------------------------------
A linear array's scan lines are parallel and vertical, so the (depth x line)
grid is already a Cartesian image; scan conversion is only a resample onto
display pixels. Steps: crop depth to z_range (default Z_MIN..Z_MAX, the
displayed depth; the RF is recorded to 60 mm only for T5's test), then
bilinear interpolation (scipy RegularGridInterpolator) of the B-mode onto
the output grid. No polar / sector geometry (that was the old T7 notebook's
bug for this probe).

keep_aspect=True (default): square pixels, the largest grid that fits inside
output_size (40 mm x 10 mm into (512, 512) -> 512 x 129 px, 0.078 mm/px).
keep_aspect=False: stretch to exactly output_size (non-square pixels).
Square pixels matter because postprocess()'s filter radius is in pixels:
with 4:1 stretched pixels a 9x9 px window would cover 0.7 mm axially but
only 0.18 mm laterally, i.e. smooth speckle in one direction only.

Post-processing (despeckling)
-----------------------------
The self-guided filter (He et al. 2013) from T7's notebook
"262SP009_Post_Image_Processing_v2.ipynb" (Section 4, `guided`), unchanged:
the notebook's conclusion picked it out of 10 filters (median, Lee, SRAD, NLM,
wavelet, OBNLM, BM3D, guided, bilateral, TV) on real clinical images -- it
lowered the speckle index 0.22 -> 0.14 on the breast test set while keeping
92 % of lesion-edge sharpness. With guide G = I, window w_k of radius r:

    a_k = cov_k(I, I) / (var_k(I) + eps),   b_k = mean_k(I) - a_k * mean_k(I)
    q   = mean_w(a) * I + mean_w(b)

Defaults r = 4, eps = 0.001: the notebook's tuned fixed value (its auto-eps
fit is weak, r = 0.54, and was fitted on breast noise levels only). The
image is quantised to uint8 exactly as in the notebook, filtered, and
returned as float in [0, 1]. method="none" skips the filter.
Grey-level mapping: out = filtered ** gamma (gamma = 1.0, the default, leaves
it unchanged; < 1 brightens dark regions, > 1 darkens them).

TIME-AXIS RULE (README Section 6) -- not applicable here: no delay-to-sample
lookup happens in this module. Its inputs already carry physical axes
(das_beamform() did the np.interp lookups on the real t_axis); resampling
uses those axes, never sample indices.

Assumptions / limitations
-------------------------
- Bilinear resampling of log-compressed data (standard display practice);
  upsampling 64 lines to ~129 px adds no lateral resolution.
- Filter parameters were tuned on real clinical images (breast), not on this
  simulator; they are reused, not re-tuned.
- Metric definitions are the notebook's (SI 7x7, CNR/gCNR ring of 10 px),
  so pixel-based sizes depend on the output pixel size.
"""

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

DEFAULT_OUTPUT_SIZE = (512, 512)   # was config.OUTPUT_SIZE in the first project (max H, W in pixels)


# ----------------------------------------------------------------------
# Scan conversion
# ----------------------------------------------------------------------
def scan_convert(bmode, x_axis, z_axis, output_size=None, z_range=None, keep_aspect=True):
    """Resample the (depth x line) B-mode onto display pixels; see module docstring."""
    bmode = np.asarray(bmode, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    z_axis = np.asarray(z_axis, dtype=float)
    if bmode.shape != (z_axis.size, x_axis.size):
        raise ValueError(f"bmode shape {bmode.shape} != (len(z_axis), len(x_axis)) = {(z_axis.size, x_axis.size)}")
    h_max, w_max = DEFAULT_OUTPUT_SIZE if output_size is None else output_size
    # v2: the depth range is chosen per phantom by the beamformer, so the default is the whole z_axis
    z0, z1 = (z_axis[0], z_axis[-1]) if z_range is None else z_range
    z0, z1 = max(z0, z_axis[0]), min(z1, z_axis[-1])
    x0, x1 = x_axis[0], x_axis[-1]

    if keep_aspect:
        pixel = max((z1 - z0) / (h_max - 1), (x1 - x0) / (w_max - 1))
        h = int(np.floor((z1 - z0) / pixel + 1e-9)) + 1
        w = int(np.floor((x1 - x0) / pixel + 1e-9)) + 1
    else:
        h, w = h_max, w_max
    z_out = np.linspace(z0, z1, h)
    x_out = np.linspace(x0, x1, w)

    interp = RegularGridInterpolator((z_axis, x_axis), bmode, method="linear")
    zz, xx = np.meshgrid(z_out, x_out, indexing="ij")
    image = interp(np.stack([zz.ravel(), xx.ravel()], axis=1)).reshape(h, w)
    return np.clip(image, 0.0, 1.0), x_out, z_out


# ----------------------------------------------------------------------
# Despeckling: T7 notebook's guided filter (Section 4), unchanged
# ----------------------------------------------------------------------
def _finish(out, img, scan):
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out[~scan] = img[~scan]
    return out


def guided(img, scan, r=4, eps=0.01, guide=None):
    """Notebook's guided filter: uint8 img, bool scan mask -> uint8 (guide=None -> self-guided)."""
    I = img.astype(np.float32) / 255; G = I if guide is None else guide.astype(np.float32) / 255
    box = lambda x: cv2.blur(x, (2 * r + 1, 2 * r + 1))
    mI, mG = box(I), box(G); cov = box(G * I) - mG * mI; var = box(G * G) - mG * mG
    a = cov / (var + eps); b = mI - a * mG
    return _finish(255 * (box(a) * G + box(b)), img, scan)


def to_uint8(image):
    """[0, 1] float -> uint8 grey levels (the notebook's working format)."""
    return np.clip(np.round(255 * np.asarray(image, dtype=float)), 0, 255).astype(np.uint8)


def postprocess(image, method="guided", r=4, eps=0.001, gamma=1.0):
    """Despeckle a scan-converted [0, 1] image, then gamma-map; returns float [0, 1]. See module docstring."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if method == "none":
        out = image.copy()
    elif method == "guided":
        img = to_uint8(image)
        out = guided(img, np.ones(img.shape, bool), r=r, eps=eps).astype(float) / 255
    else:
        raise ValueError(f"method must be 'guided' or 'none', got {method!r}")
    return out if gamma == 1.0 else out ** gamma


# ----------------------------------------------------------------------
# Quality metrics (notebook Section 3 definitions, on uint8 images)
# ----------------------------------------------------------------------
RING = 10
_kernel = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def speckle_index(img, scan_mask=None, k=7):
    """Mean of local std / local mean (k x k) inside the scan area; lower = less speckle."""
    f = img.astype(np.float32)
    scan_mask = np.ones(f.shape, bool) if scan_mask is None else scan_mask
    mu = cv2.blur(f, (k, k)); sd = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - mu * mu, 0))
    valid = (cv2.erode(scan_mask.astype(np.uint8), _kernel(k)) > 0) & (mu > 1)
    return float(np.mean(sd[valid] / mu[valid]))


def lesion_ring(lesion_mask):
    """Background ring: lesion dilated by RING px, minus the lesion."""
    return (cv2.dilate(lesion_mask.astype(np.uint8), _kernel(2 * RING + 1)) > 0) & ~lesion_mask


def cnr(img, lesion_mask):
    """|mu_lesion - mu_ring| / sqrt(var_lesion + var_ring); ring = lesion dilated by RING px."""
    a, b = img[lesion_mask].astype(np.float64), img[lesion_ring(lesion_mask)].astype(np.float64)
    return abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-9)


def gcnr(img, lesion_mask):
    """1 - overlap of lesion and ring grey-level histograms (0 = identical, 1 = separable)."""
    a, b = img[lesion_mask], img[lesion_ring(lesion_mask)]
    ha = np.bincount(a, minlength=256) / a.size; hb = np.bincount(b, minlength=256) / b.size
    return 1 - np.minimum(ha, hb).sum()


def edge_sharpness(img, lesion_mask):
    """Mean Sobel gradient magnitude on a 3-px band across the lesion border."""
    f = img.astype(np.float32)
    g = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    m = lesion_mask.astype(np.uint8)
    band = cv2.dilate(m, _kernel(3)) != cv2.erode(m, _kernel(3))
    return float(g[band].mean())
