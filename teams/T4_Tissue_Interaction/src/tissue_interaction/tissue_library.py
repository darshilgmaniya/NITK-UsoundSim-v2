"""Acoustic tissue library.

Every entry holds the *bulk* acoustic properties the Tissue Interaction module
needs.  Values are textbook mid-range numbers (Duck 1990 "Physical Properties
of Tissue", Szabo "Diagnostic Ultrasound Imaging", Bushberg "Essential Physics
of Medical Imaging").  They are STARTING POINTS: tune them against real Philips
images / the professor's reference tables.  Add your own with
``register_tissue`` or through the JSON/YAML mapping file.

Units
-----
density            kg/m^3
speed              m/s
atten_db_cm_mhz    alpha0 in  alpha(f) = alpha0 * f_MHz**atten_exp   [dB/cm]
backscatter_db     diffuse (speckle-generating) scattering level in dB,
                   relative to normal liver parenchyma (= 0 dB)
scatter_exp        diffuse backscatter amplitude ~ (f/f0)**scatter_exp
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Union

DB_PER_NEPER = 8.685889638065035


@dataclass(frozen=True)
class Tissue:
    name: str
    density: float
    speed: float
    atten_db_cm_mhz: float
    atten_exp: float = 1.0
    backscatter_db: float = 0.0
    scatter_exp: float = 1.0
    kind: str = "soft"
    description: str = ""

    @property
    def impedance(self) -> float:
        """Acoustic impedance Z = rho*c  [Rayl = kg m^-2 s^-1]."""
        return self.density * self.speed

    @property
    def impedance_mrayl(self) -> float:
        return self.impedance / 1e6

    def attenuation_db_cm(self, f_hz: float) -> float:
        return self.atten_db_cm_mhz * (f_hz / 1e6) ** self.atten_exp

    def attenuation_np_m(self, f_hz: float) -> float:
        """Amplitude attenuation in Np/m at frequency f_hz."""
        return self.attenuation_db_cm(f_hz) * 100.0 / DB_PER_NEPER


def _t(name, rho, c, a0, y, bs, m, kind, desc=""):
    return Tissue(name, rho, c, a0, y, bs, m, kind, desc)


_BUILTIN: List[Tissue] = [
    # ---- fluids / anechoic -------------------------------------------------
    _t("water", 998, 1480, 0.0022, 2.0, -90, 2.0, "fluid", "pure water (20 C)"),
    _t("cyst_fluid", 1010, 1530, 0.02, 2.0, -90, 2.0, "fluid", "simple cyst, anechoic"),
    _t("bile", 1030, 1540, 0.1, 1.5, -90, 2.0, "fluid", "gallbladder / bile ducts"),
    _t("urine", 1010, 1550, 0.01, 2.0, -90, 2.0, "fluid", "bladder"),
    _t("amniotic_fluid", 1010, 1534, 0.01, 2.0, -90, 2.0, "fluid", "obstetric"),
    _t("coupling_gel", 1020, 1520, 0.05, 2.0, -90, 2.0, "coupling", "ultrasound gel / stand-off"),
    _t("blood", 1060, 1575, 0.14, 1.2, -40, 2.0, "fluid", "flowing blood, nearly anechoic"),
    # ---- gas ----------------------------------------------------------------
    _t("air", 1.2, 343, 50.0, 1.0, -90, 0.0, "gas", "air / gas - total reflector"),
    _t("bowel_gas", 1.2, 343, 50.0, 1.0, -90, 0.0, "gas", "gas inside bowel"),
    _t("lung", 400, 650, 40.0, 1.0, -15, 1.0, "gas", "aerated lung, strong shadowing"),
    # ---- soft organs --------------------------------------------------------
    _t("liver", 1060, 1590, 0.5, 1.1, 0.0, 1.0, "soft", "reference echogenicity (0 dB)"),
    _t("spleen", 1060, 1560, 0.54, 1.1, 2.0, 1.0, "soft"),
    _t("kidney_cortex", 1050, 1560, 0.7, 1.1, -6.0, 1.0, "soft"),
    _t("kidney_medulla", 1050, 1555, 0.7, 1.1, -20.0, 1.0, "soft", "renal pyramids, hypoechoic"),
    _t("kidney_sinus", 1000, 1470, 0.6, 1.0, 6.0, 1.0, "soft", "renal sinus fat, hyperechoic"),
    _t("pancreas", 1045, 1560, 0.6, 1.1, 1.0, 1.0, "soft"),
    _t("thyroid", 1050, 1550, 0.8, 1.1, 4.0, 1.0, "soft"),
    _t("myocardium", 1060, 1576, 0.6, 1.1, -6.0, 1.0, "soft", "heart muscle"),
    _t("prostate", 1045, 1561, 0.7, 1.1, -3.0, 1.0, "soft"),
    _t("uterus", 1050, 1560, 0.6, 1.1, -3.0, 1.0, "soft"),
    _t("fetal_soft", 1040, 1550, 0.4, 1.1, -4.0, 1.0, "soft", "fetal soft tissue"),
    _t("brain_gray", 1040, 1560, 0.6, 1.2, -12.0, 1.0, "soft"),
    _t("brain_white", 1040, 1550, 0.8, 1.2, -9.0, 1.0, "soft"),
    _t("breast_glandular", 1040, 1520, 0.75, 1.1, 2.0, 1.0, "soft"),
    _t("breast_fat", 930, 1470, 0.6, 1.0, -14.0, 1.0, "soft"),
    _t("vessel_wall", 1070, 1580, 0.9, 1.1, 4.0, 1.0, "soft", "bright artery/vein wall"),
    _t("generic_soft", 1050, 1540, 0.5, 1.0, 0.0, 1.0, "soft", "fallback soft tissue"),
    # ---- musculoskeletal / superficial -------------------------------------
    _t("skin", 1100, 1600, 1.0, 1.0, 3.0, 1.0, "soft"),
    _t("fat", 950, 1450, 0.6, 1.0, -12.0, 1.0, "soft", "subcutaneous fat, hypoechoic"),
    _t("muscle", 1090, 1580, 0.75, 1.1, -8.0, 1.0, "soft"),
    _t("tendon", 1120, 1750, 4.7, 1.0, 3.0, 1.0, "soft", "fibrillar, bright"),
    _t("cartilage", 1100, 1665, 1.8, 1.0, -25.0, 1.0, "soft"),
    _t("bone", 1900, 3500, 13.0, 1.0, -10.0, 1.0, "bone", "cortical bone, strong reflector + shadow"),
    # ---- pathology / anomalies / agents ------------------------------------
    _t("tumor_hypoechoic", 1070, 1550, 0.9, 1.2, -10.0, 1.0, "lesion", "solid malignant-like, hypoechoic"),
    _t("tumor_hyperechoic", 1060, 1580, 0.7, 1.1, 6.0, 1.0, "lesion", "hyperechoic lesion (e.g. hemangioma)"),
    _t("abscess", 1030, 1540, 0.3, 1.2, -25.0, 1.0, "lesion", "complex fluid collection"),
    _t("calcification", 2000, 3500, 10.0, 1.0, 10.0, 1.0, "bone", "stone / calcification, shadowing"),
    _t("contrast_agent", 1055, 1500, 1.0, 1.0, -10.0, 2.0, "contrast", "microbubble-enhanced blood"),
]

TISSUES: Dict[str, Tissue] = {t.name: t for t in _BUILTIN}

_ALIASES = {
    "gel": "coupling_gel", "gallbladder": "bile", "bladder": "urine", "cyst": "cyst_fluid",
    "tumor": "tumor_hypoechoic", "tumour": "tumor_hypoechoic", "gas": "bowel_gas",
    "heart": "myocardium", "kidney": "kidney_cortex", "brain": "brain_gray",
    "breast": "breast_glandular", "stone": "calcification", "microbubbles": "contrast_agent",
    "lesion": "tumor_hypoechoic", "fluid": "cyst_fluid", "soft": "generic_soft",
    "soft_tissue": "generic_soft", "background": "generic_soft", "vessel": "blood",
}


def register_tissue(tissue: Tissue) -> Tissue:
    TISSUES[tissue.name] = tissue
    return tissue


def tissue_from_dict(d: dict) -> Tissue:
    """Build a tissue from a user dict.  ``base`` may name an existing tissue to
    inherit missing values from; otherwise density/speed/atten are required."""
    d = dict(d)
    base = d.pop("base", None)
    if base is not None:
        t = get_tissue(base)
        name = d.pop("name", t.name)
        return replace(t, name=name, **d)
    missing = [k for k in ("name", "density", "speed", "atten_db_cm_mhz") if k not in d]
    if missing:
        raise ValueError(f"custom tissue {d} is missing required keys {missing} "
                         "(or give a 'base' tissue to inherit from)")
    return Tissue(**d)


def get_tissue(spec: Union[str, dict, Tissue]) -> Tissue:
    if isinstance(spec, Tissue):
        return spec
    if isinstance(spec, dict):
        return tissue_from_dict(spec)
    key = str(spec).strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in TISSUES:
        raise KeyError(f"unknown tissue '{spec}'. Known: {sorted(TISSUES)}")
    return TISSUES[key]


def list_tissues() -> List[Tissue]:
    return sorted(TISSUES.values(), key=lambda t: (t.kind, t.name))


def markdown_table() -> str:
    rows = ["| tissue | kind | rho (kg/m^3) | c (m/s) | Z (MRayl) | alpha0 (dB/cm/MHz^y) | y | backscatter (dB re liver) | note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t in list_tissues():
        rows.append(f"| {t.name} | {t.kind} | {t.density:g} | {t.speed:g} | {t.impedance_mrayl:.3f} | "
                    f"{t.atten_db_cm_mhz:g} | {t.atten_exp:g} | {t.backscatter_db:g} | {t.description} |")
    return "\n".join(rows)
