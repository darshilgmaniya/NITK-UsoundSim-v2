"""Configuration objects for the Tissue Interaction module."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Optional


@dataclass
class ProbeConfig:
    """Only the geometry facts the Tissue Interaction module needs.

    linear  : parallel vertical beams, origin of the incident wave for a point
              (x, z) is (x, 0).
    sector  : radial beams from an apex (phased / curvilinear arrays).  For a
              curvilinear probe put the apex behind the face (apex_z_m < 0).
    """
    name: str = "generic_linear"
    geometry: str = "linear"
    apex_x_m: float = 0.0
    apex_z_m: float = 0.0
    # receive array (used to build the channel data handed to Receiver Beamforming)
    n_elements: int = 64
    pitch_m: float = 3.0e-4
    element_x_m: Optional[list] = None     # explicit element positions override n_elements/pitch
    element_z_m: Optional[list] = None     # (default 0 = flat array on the skin surface)

    def element_positions(self):
        """(xz[n,2], normals[n,2]) of the receive elements.
        flat arrays lie on z=0 centred on x=0; a sector probe with the apex behind the
        face (apex_z_m < 0) is treated as a curvilinear array on that arc."""
        import numpy as np
        if self.element_x_m is not None:
            ex = np.asarray(self.element_x_m, float)
            ez = np.zeros_like(ex) if self.element_z_m is None else np.asarray(self.element_z_m, float)
            return np.stack([ex, ez], 1), np.tile([0.0, 1.0], (ex.size, 1))
        n = int(self.n_elements)
        if self.geometry == "sector" and self.apex_z_m < 0:
            R = -self.apex_z_m
            th = (np.arange(n) - (n - 1) / 2) * (self.pitch_m / R)
            xz = np.stack([self.apex_x_m + R * np.sin(th), self.apex_z_m + R * np.cos(th)], 1)
            nrm = np.stack([np.sin(th), np.cos(th)], 1)
            return xz, nrm
        ex = (np.arange(n) - (n - 1) / 2) * self.pitch_m
        return np.stack([ex, np.zeros(n)], 1), np.tile([0.0, 1.0], (n, 1))

    def origins_and_dirs(self, x, z):
        """Return (origin[n,2], unit_dir[n,2]) of the incident wave at points."""
        import numpy as np
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        if self.geometry == "linear":
            o = np.stack([x, np.zeros_like(x)], 1)
            d = np.stack([np.zeros_like(x), np.ones_like(x)], 1)
        elif self.geometry == "sector":
            o = np.tile([self.apex_x_m, self.apex_z_m], (x.size, 1)).astype(float)
            v = np.stack([x - self.apex_x_m, z - self.apex_z_m], 1)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            d = v / np.maximum(n, 1e-12)
        else:
            raise ValueError(f"unknown probe geometry '{self.geometry}'")
        return o, d


@dataclass
class TissueInteractionConfig:
    # --- acoustic / imaging parameters (get the real ones from Transducer team)
    f0_hz: float = 3.5e6
    c0: float = 1540.0
    fs_hz: float = 40e6                # only used for the built-in mock incident wave
    fractional_bandwidth: float = 0.6
    f_number: float = 2.0
    pulse_cycles: float = 2.0
    # --- scatterer generation -------------------------------------------------
    scatterers_per_cell: float = 6.0   # speckle realism (>=10 is ideal, costs memory)
    density_per_mm2: Optional[float] = None   # None -> derived from scatterers_per_cell
    max_scatterers: int = 30000          # budget for DIFFUSE scatterers (interface ones are added on top)
    diffuse_ref_reflectivity: float = 0.01   # rms reflectivity per resolution cell, liver
    interface_gain: float = 1.0
    interface_spacing_over_lambda: float = 0.5
    specular_theta0_deg: float = 10.0
    specular_floor: float = 0.10
    texture_strength: float = 0.25
    texture_corr_mm: float = 3.0
    prune_db: float = -60.0            # diffuse tissues below this are treated anechoic
    refine_step_over_lambda: float = 0.25
    max_refined_pixels: int = 6_000_000
    smooth_refine: bool = True
    # --- path effects ---------------------------------------------------------
    # 'full' : apply two-way attenuation (+ frequency downshift) and interface
    #          transmission loss (shadowing) on the way to/from each scatterer
    # 'none' : the Acoustic Propagation module already did all of that
    path_model: str = "full"
    # --- return path to the receive elements (channel data) -------------------
    spreading_exponent: float = 0.5    # 0.5 = 2-D cylindrical, 1.0 = 3-D spherical
    spreading_ref_m: float = 0.01
    element_directivity: str = "cosine"   # 'cosine' | 'none'
    seed: int = 0
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    @property
    def wavelength(self) -> float:
        return self.c0 / self.f0_hz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TissueInteractionConfig":
        d = dict(d or {})
        probe = d.pop("probe", None)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if probe:
            cfg.probe = ProbeConfig(**probe)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "TissueInteractionConfig":
        import yaml
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))
