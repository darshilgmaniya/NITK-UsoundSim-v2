"""
pipeline.py - the whole simulator in one call (integration step 3)
==================================================================

    runner 1-5   integration/02_full_relay   phantom -> ... -> beamformed_rf (depth, lines)
    runner 6     teams/T6_Image_Reconstruction  beamformed_rf -> conditioned RF (DC removal + band-pass)
    runner 7     teams/T7_B_mode  RF -> bmode in [0, 1]       (envelope + log compression)
    runner 8     teams/T8_Post_Image_Processing  bmode -> scan-converted image -> despeckled final image

Runners 1-5 are *not* re-implemented: ``run_full`` is called as it is. This file only adds runners 6, 7 and 8
and the bookkeeping that joins them, so nothing upstream can drift.

``source`` accepts everything ``run_full`` / ``run_relay`` accept:
    "points"                          nine wire targets (demo)
    "abdomen", "kidney", ...          a built-in phantom preset
    "my_points.csv"                   x_mm,z_mm,amp   or   x_mm,z_mm,tissue
    "phantom.png" + mapping=...       an image (label map / intensity / colour) and its mapping file
    an (N, 3) array                   x_mm, z_mm, amp
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import _paths  # noqa: F401
from usoundsim_relay import RelayConfig
from usoundsim_full import FullResult, run_full
from receiver_beamforming import RxConfig
from image_reconstruction import ReconConfig, image_reconstruction
from bmode_formation import bmode_formation
from post_processing_stage import postprocess, scan_convert


@dataclass
class PostConfig:
    """Options of runners 7 and 8 (B-mode dynamic range, scan conversion, despeckle)."""
    dynamic_range_db: Optional[float] = None   # None -> same as RxConfig.dynamic_range_db
    despeckle: str = "guided"                  # 'guided' | 'none'
    guided_r: int = 4                          # window radius [pixels]
    guided_eps: float = 0.001
    gamma: float = 1.0
    output_size: tuple = (512, 512)            # largest (height, width) of the final image [pixels]
    keep_aspect: bool = True                   # square pixels

    def validate(self) -> None:
        if self.despeckle not in ("guided", "none"):
            raise ValueError("despeckle must be 'guided' or 'none'")
        if self.dynamic_range_db is not None and self.dynamic_range_db <= 0:
            raise ValueError("dynamic_range_db must be > 0")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")


@dataclass
class PipelineResult:
    full: FullResult                 # runners 1-5 (relay + beamformer)
    recon_rf: np.ndarray             # (depth, lines) conditioned RF         runner 6 output (T6)
    bmode: np.ndarray                # (depth, lines) in [0, 1]            runner 7 output (T7)
    image: np.ndarray                # (H, W) scan-converted, before despeckling
    final: np.ndarray                # (H, W) final despeckled image in [0, 1]    runner 8 output (T8)
    x_img: np.ndarray                # (W,) lateral axis of image/final [m]
    z_img: np.ndarray                # (H,) depth axis of image/final [m]
    files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def extent_mm(self):
        """imshow extent [left, right, bottom, top] in mm."""
        return [self.x_img[0] * 1e3, self.x_img[-1] * 1e3, self.z_img[-1] * 1e3, self.z_img[0] * 1e3]


def run_pipeline(source, rc: Optional[RelayConfig] = None, rx: Optional[RxConfig] = None,
                 post: Optional[PostConfig] = None, recon: Optional[ReconConfig] = None, *, mapping=None, spacing_mm=None,
                 background: str = "generic_soft", units: str = "mm", outdir: Optional[str] = None,
                 verbose: bool = True, **ti_overrides) -> PipelineResult:
    """Phantom / points / image -> ... -> final despeckled B-mode image."""
    rx = rx or RxConfig()
    post = post or PostConfig()
    post.validate()
    say = print if verbose else (lambda *a, **k: None)

    full = run_full(source, rc, rx, mapping=mapping, spacing_mm=spacing_mm, background=background,
                    units=units, outdir=outdir, verbose=verbose, **ti_overrides)
    bf = full.bf
    timings = dict(full.timings)

    t0 = time.time()
    rcfg = recon or ReconConfig()
    rec = image_reconstruction(bf.beamformed_rf, bf.z_axis, c=full.relay.rc.c0, f0_hz=full.relay.rc.f0_hz, config=rcfg)
    timings["runner6_recon"] = time.time() - t0
    say(f"[run 6 ] Image recon     : rf {rec.rf.shape}; " + ("; ".join(rec.notes) if rcfg.enabled else "disabled")
        + f"   ({timings['runner6_recon']:.1f}s)")

    t0 = time.time()
    dr = post.dynamic_range_db if post.dynamic_range_db is not None else rx.dynamic_range_db
    bmode = bmode_formation(rec.rf, dr)
    timings["runner7_bmode"] = time.time() - t0
    say(f"[run 7 ] B-mode          : bmode {bmode.shape} in [0, 1], dynamic range {dr:g} dB   ({timings['runner7_bmode']:.1f}s)")

    t0 = time.time()
    image, x_img, z_img = scan_convert(bmode, bf.x_lines, bf.z_axis, output_size=post.output_size,
                                       keep_aspect=post.keep_aspect)
    final = postprocess(image, method=post.despeckle, r=post.guided_r, eps=post.guided_eps, gamma=post.gamma)
    timings["runner8_post"] = time.time() - t0
    say(f"[run 8 ] Scan conv + post: image {image.shape} ({(x_img[1] - x_img[0]) * 1e3:.3f} mm/px), "
        f"despeckle={post.despeckle}   ({timings['runner8_post']:.1f}s)")

    res = PipelineResult(full, rec.rf, bmode, image, final, x_img, z_img, dict(full.files), timings)
    if outdir:
        res.files.update(save_pipeline(outdir, res))
    return res


def save_pipeline(outdir: str, res: PipelineResult) -> Dict[str, str]:
    """Write 06_recon_rf.npz, 07_bmode_outputs.npz and 08_final_image.npz (+ final_image.png written by the plotting helper)."""
    os.makedirs(outdir, exist_ok=True)
    f5 = os.path.join(outdir, "06_recon_rf.npz")
    np.savez_compressed(f5, recon_rf=res.recon_rf, z_axis=res.full.bf.z_axis, x_lines=res.full.bf.x_lines)
    f6 = os.path.join(outdir, "07_bmode_outputs.npz")
    np.savez_compressed(f6, bmode=res.bmode, z_axis=res.full.bf.z_axis, x_lines=res.full.bf.x_lines)
    f7 = os.path.join(outdir, "08_final_image.npz")
    np.savez_compressed(f7, image=res.image, final=res.final, x_img=res.x_img, z_img=res.z_img)
    return {"06_recon_rf": f5, "07_bmode": f6, "08_final_image": f7}
