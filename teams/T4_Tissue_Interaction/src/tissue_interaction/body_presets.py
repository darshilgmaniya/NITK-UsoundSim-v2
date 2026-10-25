"""Body-region presets + typical clinical probe classes.

Probe classes are *inspired by* the Philips transducer families (C5-1 curvilinear,
L12-5 linear, S5-1 phased sector) but the numbers are class-typical, NOT vendor
data.  Replace them with whatever the Transducer team finalises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .config import ProbeConfig, TissueInteractionConfig


@dataclass(frozen=True)
class ProbeClass:
    name: str
    geometry: str
    f0_hz: float
    apex_z_m: float
    note: str


PROBE_CLASSES: Dict[str, ProbeClass] = {
    "curvilinear_abdominal": ProbeClass("C5-1-like curvilinear", "sector", 3.0e6, -0.050,
                                        "deep abdomen / obstetric, ~1-5 MHz"),
    "linear_small_parts": ProbeClass("L12-5-like linear", "linear", 8.0e6, 0.0,
                                     "thyroid, breast, MSK, vascular, ~5-12 MHz"),
    "phased_cardiac": ProbeClass("S5-1-like phased sector", "sector", 2.5e6, 0.0,
                                 "cardiac / intercostal windows, ~1-5 MHz"),
    "linear_general": ProbeClass("general linear", "linear", 5.0e6, 0.0, "generic 5 MHz linear"),
}


@dataclass(frozen=True)
class BodyPreset:
    name: str
    description: str
    probe: str
    depth_mm: float
    width_mm: float
    legend: Dict[int, str] = field(default_factory=dict)   # default label-id -> tissue


def _legend(*names):
    return {i: n for i, n in enumerate(names)}


BODY_PRESETS: Dict[str, BodyPreset] = {p.name: p for p in [
    BodyPreset("abdomen", "liver / gallbladder / vessels / kidney / rib / bowel gas",
               "curvilinear_abdominal", 120, 120,
               _legend("coupling_gel", "skin", "fat", "muscle", "liver", "bile", "blood", "kidney_cortex",
                       "kidney_medulla", "kidney_sinus", "tumor_hypoechoic", "tumor_hyperechoic",
                       "cyst_fluid", "bowel_gas", "bone", "vessel_wall", "contrast_agent", "spleen", "pancreas")),
    BodyPreset("kidney", "kidney with cortex/medulla/sinus, stone and cyst", "curvilinear_abdominal", 120, 100,
               _legend("coupling_gel", "skin", "fat", "muscle", "kidney_cortex", "kidney_medulla",
                       "kidney_sinus", "calcification", "cyst_fluid", "liver")),
    BodyPreset("thyroid", "thyroid lobes, trachea (air), carotid arteries, nodule", "linear_small_parts", 40, 40,
               _legend("coupling_gel", "skin", "fat", "muscle", "thyroid", "air", "blood", "vessel_wall",
                       "tumor_hypoechoic", "cyst_fluid")),
    BodyPreset("breast", "fat/glandular tissue, mass, cyst, rib and lung", "linear_small_parts", 50, 45,
               _legend("coupling_gel", "skin", "breast_fat", "breast_glandular", "tumor_hypoechoic",
                       "cyst_fluid", "muscle", "bone", "lung")),
    BodyPreset("cardiac", "four heart chambers, myocardium, chest wall, lung", "phased_cardiac", 140, 110,
               _legend("coupling_gel", "skin", "fat", "muscle", "bone", "myocardium", "blood", "lung",
                       "contrast_agent")),
    BodyPreset("carotid", "carotid artery with plaque, jugular vein", "linear_small_parts", 35, 38,
               _legend("coupling_gel", "skin", "fat", "muscle", "blood", "vessel_wall", "calcification",
                       "tumor_hypoechoic")),
    BodyPreset("msk", "muscle / tendon / cortical bone (acoustic shadow)", "linear_small_parts", 45, 40,
               _legend("coupling_gel", "skin", "fat", "muscle", "tendon", "bone", "cartilage")),
    BodyPreset("obstetric", "uterus, amniotic fluid, fetal head with skull", "curvilinear_abdominal", 150, 120,
               _legend("coupling_gel", "skin", "fat", "muscle", "uterus", "amniotic_fluid", "fetal_soft",
                       "bone", "brain_gray", "cyst_fluid")),
    BodyPreset("brain", "neonatal transfontanellar: brain, ventricles, skull", "phased_cardiac", 90, 70,
               _legend("coupling_gel", "skin", "bone", "brain_gray", "brain_white", "cyst_fluid", "blood")),
    BodyPreset("lung", "chest wall, ribs, pleura, aerated lung", "linear_small_parts", 50, 45,
               _legend("coupling_gel", "skin", "fat", "muscle", "bone", "lung", "tumor_hypoechoic")),
    BodyPreset("prostate", "bladder, prostate with lesion", "linear_general", 70, 60,
               _legend("coupling_gel", "skin", "fat", "muscle", "urine", "prostate", "tumor_hypoechoic",
                       "bowel_gas")),
    BodyPreset("generic", "contrast-target phantom: water, air, bone, tumours, contrast agent",
               "linear_general", 70, 60,
               _legend("coupling_gel", "liver", "water", "bowel_gas", "bone", "tumor_hypoechoic",
                       "tumor_hyperechoic", "contrast_agent", "blood", "calcification", "fat")),
]}


def list_body_parts():
    return sorted(BODY_PRESETS)


def get_preset(name: str) -> BodyPreset:
    key = name.strip().lower()
    if key not in BODY_PRESETS:
        raise KeyError(f"unknown body part '{name}'. Choose from {list_body_parts()}")
    return BODY_PRESETS[key]


def config_for_body_part(name: str, **overrides) -> TissueInteractionConfig:
    """Config with the probe / frequency that is typical for the body region."""
    p = get_preset(name)
    pc = PROBE_CLASSES[p.probe]
    cfg = TissueInteractionConfig(f0_hz=pc.f0_hz)
    cfg.probe = ProbeConfig(name=pc.name, geometry=pc.geometry, apex_z_m=pc.apex_z_m)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
