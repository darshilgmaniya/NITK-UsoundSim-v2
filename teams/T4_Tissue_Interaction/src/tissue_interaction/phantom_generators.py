"""Synthetic phantoms for every body-region preset (used for demos and tests).

Everything the user can load from a file can also be produced here, so the
whole pipeline can be tested without any input image.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .phantom import Phantom
from .tissue_library import get_tissue


class PhantomBuilder:
    """Tiny constructive-geometry helper.  All coordinates in millimetres,
    x centred on 0, z = depth from the skin surface."""

    def __init__(self, width_mm, depth_mm, spacing_mm=0.25, background="generic_soft", name="phantom"):
        self.s = spacing_mm
        nx = int(round(width_mm / spacing_mm))
        nz = int(round(depth_mm / spacing_mm))
        x = (np.arange(nx) + 0.5 - nx / 2) * spacing_mm
        z = (np.arange(nz) + 0.5) * spacing_mm
        self.X, self.Z = np.meshgrid(x, z)
        self.names: List[str] = [background]
        self.idx = np.zeros((nz, nx), np.int16)
        self.name = name

    def _id(self, tissue: str) -> int:
        t = get_tissue(tissue).name
        if t not in self.names:
            self.names.append(t)
        return self.names.index(t)

    def layer(self, z0, z1, tissue, wobble_mm=0.0, period_mm=35.0, phase=0.0):
        w = wobble_mm * np.sin(2 * np.pi * self.X / period_mm + phase)
        self.idx[(self.Z >= z0 + w) & (self.Z < z1 + w)] = self._id(tissue)
        return self

    def rect(self, x0, x1, z0, z1, tissue):
        self.idx[(self.X >= x0) & (self.X < x1) & (self.Z >= z0) & (self.Z < z1)] = self._id(tissue)
        return self

    def _ell(self, cx, cz, rx, rz, angle_deg):
        a = np.deg2rad(angle_deg)
        xr = (self.X - cx) * np.cos(a) + (self.Z - cz) * np.sin(a)
        zr = -(self.X - cx) * np.sin(a) + (self.Z - cz) * np.cos(a)
        return (xr / rx) ** 2 + (zr / rz) ** 2

    def ellipse(self, cx, cz, rx, rz, tissue, angle_deg=0.0):
        self.idx[self._ell(cx, cz, rx, rz, angle_deg) <= 1.0] = self._id(tissue)
        return self

    def ring(self, cx, cz, rx, rz, thick, tissue, angle_deg=0.0):
        outer = self._ell(cx, cz, rx, rz, angle_deg) <= 1.0
        inner = self._ell(cx, cz, max(rx - thick, 1e-3), max(rz - thick, 1e-3), angle_deg) <= 1.0
        self.idx[outer & ~inner] = self._id(tissue)
        return self

    def vessel(self, cx, cz, rx, rz, angle_deg=0.0, lumen="blood", wall=0.8):
        self.ellipse(cx, cz, rx + wall, rz + wall, "vessel_wall", angle_deg)
        return self.ellipse(cx, cz, rx, rz, lumen, angle_deg)

    def body_wall(self, skin=1.5, fat=10.0, muscle=8.0, wobble=0.6):
        self.layer(0, skin, "skin")
        self.layer(skin, skin + fat, "fat", wobble_mm=wobble)
        self.layer(skin + fat, skin + fat + muscle, "muscle")
        return skin + fat + muscle

    def build(self) -> Phantom:
        return Phantom(self.idx, [get_tissue(n) for n in self.names],
                       self.s * 1e-3, self.s * 1e-3, name=self.name)


# ----------------------------------------------------------------------------
def make_abdomen(spacing_mm=0.25, anomalies=True) -> Phantom:
    b = PhantomBuilder(120, 120, spacing_mm, "liver", "abdomen")
    zw = b.body_wall(1.5, 12, 10)
    b.layer(zw, 120, "liver")
    b.ellipse(-35, 62, 9, 17, "bile", 20)                       # gallbladder
    b.vessel(-5, 58, 2.2, 24, -12)                              # hepatic vein
    b.vessel(12, 72, 2.8, 17, 70)                               # portal vein
    b.ellipse(-52, 28, 3.2, 2.6, "bone")                        # rib
    b.ellipse(38, 96, 18, 12, "kidney_cortex", -20)             # kidney
    b.ellipse(38, 96, 9, 4.5, "kidney_sinus", -20)
    for k in range(6):
        a = np.deg2rad(-20 + 60 * k)
        b.ellipse(38 + 11 * np.cos(a), 96 + 7 * np.sin(a), 3.0, 2.2, "kidney_medulla", np.rad2deg(a))
    if anomalies:
        b.ellipse(25, 50, 7, 6, "tumor_hypoechoic")
        b.ellipse(-18, 92, 5, 5, "tumor_hyperechoic")
        b.ellipse(8, 100, 6, 6, "cyst_fluid")
        b.ellipse(52, 33, 4.5, 2.5, "bowel_gas")
        b.ellipse(-42, 98, 3, 3, "contrast_agent")
    return b.build()


def make_kidney(spacing_mm=0.25, anomalies=True) -> Phantom:
    b = PhantomBuilder(100, 120, spacing_mm, "fat", "kidney")
    zw = b.body_wall(1.5, 14, 12)
    b.layer(zw, 120, "fat")
    b.ellipse(0, 75, 24, 38, "kidney_cortex", 10)
    b.ellipse(0, 75, 10, 18, "kidney_sinus", 10)
    for k in range(7):
        a = np.deg2rad(10 + 50 * k)
        b.ellipse(14 * np.cos(a) * 1.0, 75 + 24 * np.sin(a), 3.5, 5.0, "kidney_medulla", np.rad2deg(a))
    if anomalies:
        b.ellipse(-14, 62, 4, 4, "cyst_fluid")
        b.ellipse(4, 85, 2.2, 2.2, "calcification")
        b.ellipse(15, 55, 5, 5, "tumor_hypoechoic")
    return b.build()


def make_thyroid(spacing_mm=0.1, anomalies=True) -> Phantom:
    b = PhantomBuilder(40, 40, spacing_mm, "muscle", "thyroid")
    b.layer(0, 0.8, "skin")
    b.layer(0.8, 3.0, "fat", wobble_mm=0.2)
    b.layer(3.0, 8.0, "muscle")
    b.layer(8.0, 40, "muscle")
    b.ellipse(-9, 17, 7.5, 10, "thyroid", 8)
    b.ellipse(9, 17, 7.5, 10, "thyroid", -8)
    b.rect(-3, 3, 13, 18, "thyroid")                              # isthmus
    b.ellipse(0, 31, 8, 7, "air")                                 # trachea
    b.vessel(-17, 18, 3.2, 3.2, 0)
    b.vessel(17, 18, 3.2, 3.2, 0)
    if anomalies:
        b.ellipse(-9, 16, 2.6, 2.2, "tumor_hypoechoic")
        b.ellipse(9, 20, 1.8, 1.8, "cyst_fluid")
    return b.build()


def make_breast(spacing_mm=0.1, anomalies=True) -> Phantom:
    b = PhantomBuilder(45, 50, spacing_mm, "breast_fat", "breast")
    b.layer(0, 1.0, "skin")
    b.ellipse(0, 16, 20, 8, "breast_glandular")
    b.ellipse(-12, 20, 8, 3.5, "breast_glandular", 15)
    b.layer(28, 34, "muscle", wobble_mm=0.4)
    b.layer(34, 50, "lung", wobble_mm=0.3)
    b.ellipse(-12, 38, 3.5, 3.0, "bone")
    b.ellipse(12, 38, 3.5, 3.0, "bone")
    if anomalies:
        b.ellipse(4, 16, 4.0, 2.8, "tumor_hypoechoic", 20)
        b.ellipse(-10, 9, 2.5, 2.0, "cyst_fluid")
    return b.build()


def make_cardiac(spacing_mm=0.3, contrast_in_lv=False) -> Phantom:
    b = PhantomBuilder(110, 140, spacing_mm, "generic_soft", "cardiac")
    b.layer(0, 1.5, "skin")
    b.layer(1.5, 6, "fat")
    b.layer(6, 12, "muscle")
    b.ellipse(-30, 16, 4, 3, "bone")
    b.ellipse(30, 16, 4, 3, "bone")
    b.ellipse(-52, 85, 9, 42, "lung")
    b.ellipse(52, 85, 9, 42, "lung")
    b.ellipse(0, 78, 38, 52, "myocardium")
    b.ellipse(-13, 88, 12, 30, "blood", 8)                        # LV
    b.ellipse(16, 84, 11, 26, "blood", -8)                        # RV
    b.ellipse(-14, 45, 13, 9, "blood")                            # LA
    b.ellipse(16, 45, 12, 9, "blood")                             # RA
    if contrast_in_lv:
        b.ellipse(-13, 88, 9, 26, "contrast_agent", 8)
    return b.build()


def make_carotid(spacing_mm=0.1, anomalies=True) -> Phantom:
    b = PhantomBuilder(38, 35, spacing_mm, "muscle", "carotid")
    b.layer(0, 0.8, "skin")
    b.layer(0.8, 3.0, "fat")
    b.layer(3, 35, "muscle")
    b.vessel(-3, 14, 3.2, 3.2, 0, wall=0.9)
    b.vessel(11, 12, 4.2, 3.0, 0)                                 # jugular
    if anomalies:
        b.ellipse(-3, 17.2, 2.6, 1.0, "tumor_hypoechoic")        # soft plaque
        b.ellipse(-3, 11.0, 1.4, 0.7, "calcification")           # calcified plaque
    return b.build()


def make_msk(spacing_mm=0.1, anomalies=True) -> Phantom:
    b = PhantomBuilder(40, 45, spacing_mm, "muscle", "msk")
    b.layer(0, 0.9, "skin")
    b.layer(0.9, 4.0, "fat", wobble_mm=0.2)
    b.layer(4, 18, "muscle")
    b.layer(18, 22, "tendon", wobble_mm=0.8, period_mm=40)
    b.layer(22, 27, "muscle")
    b.layer(27, 45, "muscle")
    b.layer(31, 34, "bone", wobble_mm=1.5, period_mm=45)
    b.layer(30, 31, "cartilage", wobble_mm=1.5, period_mm=45)
    if anomalies:
        b.ellipse(0, 13, 3.5, 2.2, "abscess")
    return b.build()


def make_obstetric(spacing_mm=0.3, anomalies=True) -> Phantom:
    b = PhantomBuilder(120, 150, spacing_mm, "uterus", "obstetric")
    zw = b.body_wall(1.5, 15, 10)
    b.layer(zw, 150, "uterus")
    b.ellipse(0, 88, 48, 52, "amniotic_fluid")
    b.ellipse(-8, 82, 21, 19, "fetal_soft", 15)                   # head
    b.ring(-8, 82, 21, 19, 2.0, "bone", 15)                       # skull
    b.ellipse(-8, 82, 17.5, 15.5, "brain_gray", 15)
    b.ellipse(-8, 82, 3, 9, "cyst_fluid", 15)
    b.ellipse(22, 105, 17, 11, "fetal_soft", -25)                 # body
    if anomalies:
        b.ellipse(-42, 120, 5, 5, "cyst_fluid")
    return b.build()


def make_brain(spacing_mm=0.25, anomalies=True) -> Phantom:
    b = PhantomBuilder(70, 90, spacing_mm, "brain_gray", "brain")
    b.layer(0, 1.0, "skin")
    b.layer(1.0, 3.0, "brain_gray")
    b.ellipse(0, 50, 28, 32, "brain_white")
    b.ellipse(-8, 42, 3.5, 11, "cyst_fluid", 15)
    b.ellipse(8, 42, 3.5, 11, "cyst_fluid", -15)
    b.vessel(0, 28, 1.3, 1.3, 0, wall=0.5)
    b.ring(0, 52, 34, 40, 3.5, "bone")
    if anomalies:
        b.ellipse(14, 60, 4, 4, "tumor_hyperechoic")
    return b.build()


def make_lung(spacing_mm=0.1, anomalies=True) -> Phantom:
    b = PhantomBuilder(45, 50, spacing_mm, "lung", "lung")
    b.layer(0, 1.0, "skin")
    b.layer(1.0, 6.0, "fat", wobble_mm=0.3)
    b.layer(6, 15, "muscle")
    b.ellipse(-12, 19, 4.5, 4, "bone")
    b.ellipse(12, 19, 4.5, 4, "bone")
    b.layer(15, 50, "lung")
    b.ellipse(-12, 19, 4.5, 4, "bone")
    b.ellipse(12, 19, 4.5, 4, "bone")
    if anomalies:
        b.ellipse(0, 24, 4, 3.0, "tumor_hypoechoic")
    return b.build()


def make_prostate(spacing_mm=0.2, anomalies=True) -> Phantom:
    b = PhantomBuilder(60, 70, spacing_mm, "generic_soft", "prostate")
    zw = b.body_wall(1.5, 8, 6)
    b.ellipse(0, 28, 22, 13, "urine")
    b.ellipse(0, 52, 16, 11, "prostate")
    b.ellipse(0, 66, 8, 3, "bowel_gas")
    if anomalies:
        b.ellipse(6, 54, 4, 3, "tumor_hypoechoic")
    return b.build()


def make_generic(spacing_mm=0.15, anomalies=True) -> Phantom:
    """Contrast-target phantom: one inclusion of every important contrast class."""
    b = PhantomBuilder(60, 70, spacing_mm, "liver", "generic")
    b.layer(0, 70, "liver")
    xs = [-22, -11, 0, 11, 22]
    for x, t in zip(xs, ["water", "tumor_hypoechoic", "tumor_hyperechoic", "contrast_agent", "blood"]):
        b.ellipse(x, 18, 4.5, 4.5, t)
    for x, t in zip(xs[:3], ["bowel_gas", "bone", "calcification"]):
        b.ellipse(x - 5 if t != "bone" else x, 44 if t != "bone" else 40, 4.0, 3.0, t)
    b.ellipse(14, 50, 5, 5, "fat")
    b.ellipse(24, 40, 3.0, 3.0, "tumor_hypoechoic")
    return b.build()


GENERATORS: Dict[str, Callable[..., Phantom]] = {
    "abdomen": make_abdomen, "kidney": make_kidney, "thyroid": make_thyroid, "breast": make_breast,
    "cardiac": make_cardiac, "carotid": make_carotid, "msk": make_msk, "obstetric": make_obstetric,
    "brain": make_brain, "lung": make_lung, "prostate": make_prostate, "generic": make_generic,
}


def make_phantom(body_part: str, spacing_mm: float | None = None, **kw) -> Phantom:
    key = body_part.strip().lower()
    if key not in GENERATORS:
        raise KeyError(f"no generator for '{body_part}'. Choose from {sorted(GENERATORS)}")
    if spacing_mm is not None:
        kw["spacing_mm"] = spacing_mm
    return GENERATORS[key](**kw)


def make_point_targets(depths_mm=(15, 25, 35, 45, 55), lateral_mm=(-8.0, 0.0, 8.0), amp=1.0):
    """Wire/point-target scatterer list (x_m, z_m, amp) - the classic resolution test for beamformers."""
    pts = [(x * 1e-3, z * 1e-3, amp) for z in depths_mm for x in lateral_mm]
    return np.array(pts)
