"""Reader for the Acoustic Propagation team's forward_propagation() output  (+ an emulator for testing).

forward_propagation() gives the incident mechanical pressure that arrives AT EACH SCATTERER from the
64-element TX array, versus time:

    incident_pressure              (n_scatterers, n_samples)   row i  <->  scatterer_xyz[i]
    incident_pressure_row_labels   (n_scatterers,)
    scatterer_xyz                  (n_scatterers, 3)  [+ scatterer_xyz_columns / _units]
    t_axis                         (n_samples,)  seconds           (name auto-detected, see below)
    pulse_t, pulse_p               the TX pulse used
    element_x                      TX element geometry [m]
    (optional) scatterer_amp       reflectivity of each scatterer

Everything the Tissue Interaction module needs (scatterer positions, incident waveforms, centre
frequency, sampling rate, array geometry) is read from this single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_UNIT = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6}
_T_KEYS = ("t_axis", "incident_t_axis", "incident_pressure_t_axis", "t", "time", "time_axis")
_AMP_KEYS = ("scatterer_amp", "scatterer_amplitude", "scatterer_amplitudes", "scatterer_reflectivity", "amp")


@dataclass
class APForward:
    incident: np.ndarray                 # (n_scat, n_samples)
    t_axis: np.ndarray                   # s
    xyz: np.ndarray                      # (n_scat, 3) m
    row_labels: np.ndarray
    amp: Optional[np.ndarray] = None
    element_x: Optional[np.ndarray] = None     # m
    element_z: Optional[np.ndarray] = None
    pulse_t: Optional[np.ndarray] = None
    pulse_p: Optional[np.ndarray] = None
    c: Optional[float] = None
    extra_keys: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def fs_hz(self) -> float:
        return 1.0 / float(self.t_axis[1] - self.t_axis[0])

    def summary(self) -> str:
        s = (f"AP forward file: {self.xyz.shape[0]} scatterers x {self.incident.shape[1]} samples, "
             f"fs={self.fs_hz/1e6:.1f} MHz, t=[{self.t_axis[0]*1e6:.1f},{self.t_axis[-1]*1e6:.1f}] us")
        if self.element_x is not None:
            s += f", {self.element_x.size} TX elements"
        if self.pulse_p is not None:
            s += f", pulse f0~{estimate_f0(self.pulse_t, self.pulse_p)/1e6:.2f} MHz"
        if self.amp is None:
            s += "  (no scatterer amplitude in file -> default amp)"
        return s


def estimate_f0(pulse_t, pulse_p) -> float:
    """Dominant frequency of the TX pulse (zero-padded FFT peak, DC excluded)."""
    t = np.asarray(pulse_t, float); p = np.asarray(pulse_p, float)
    dt = float(t[1] - t[0])
    n = 1 << int(np.ceil(np.log2(max(len(p), 8) * 16)))
    sp = np.abs(np.fft.rfft(p - p.mean(), n=n))
    f = np.fft.rfftfreq(n, dt)
    sp[0] = 0
    return float(f[sp.argmax()])


def _scalar(z, key, default=None):
    if key in z.files:
        v = z[key]
        return v.item() if v.ndim == 0 else v
    return default


def load_ap_forward(path, *, incident_key: Optional[str] = None, t_key: Optional[str] = None,
                    fs_hz: Optional[float] = None) -> APForward:
    z = np.load(str(path), allow_pickle=False)
    notes = []
    ik = incident_key or ("incident_pressure" if "incident_pressure" in z.files else None)
    if ik is None:
        raise KeyError(f"no 'incident_pressure' in {z.files}; pass incident_key='...'")
    inc = np.asarray(z[ik], float)
    if inc.ndim != 2:
        raise ValueError(f"incident_pressure must be 2-D (n_scatterers, n_samples); got {inc.shape}")
    if not np.isfinite(inc).all():
        raise ValueError("incident_pressure contains NaN/inf")
    if "scatterer_xyz" not in z.files:
        raise KeyError(f"no 'scatterer_xyz' in {z.files}")
    xyz = np.asarray(z["scatterer_xyz"], float)
    cols = [str(c).lower() for c in z["scatterer_xyz_columns"]] if "scatterer_xyz_columns" in z.files else None
    unit = str(_scalar(z, "scatterer_xyz_units", "m")).strip()
    if "scatterer_xyz_units" not in z.files:
        notes.append("scatterer_xyz_units missing -> assumed metres")
    sc = _UNIT.get(unit, 1.0)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    if xyz.shape[1] == 2:
        xyz = np.stack([xyz[:, 0], np.zeros(len(xyz)), xyz[:, 1]], 1)
    elif cols and len(cols) == xyz.shape[1] and set("xyz") <= set(cols):
        xyz = xyz[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
    xyz = xyz * sc
    if xyz.shape[0] != inc.shape[0]:
        raise ValueError(f"row mismatch: incident_pressure has {inc.shape[0]} rows but scatterer_xyz has "
                         f"{xyz.shape[0]}. Row i of incident_pressure must belong to scatterer_xyz[i].")
    if xyz[:, 2].min() < 0:
        raise ValueError("scatterer depth z < 0: z = 0 must be the probe face and depth positive downward")
    labels = (np.asarray([str(s) for s in z["incident_pressure_row_labels"]])
              if "incident_pressure_row_labels" in z.files else np.array([f"scatterer_{i}" for i in range(len(xyz))]))
    tk = t_key or next((k for k in _T_KEYS if k in z.files), None)
    if tk is not None:
        t = np.asarray(z[tk], float)
        if t.size != inc.shape[1]:
            raise ValueError(f"time axis '{tk}' has {t.size} samples, incident_pressure has {inc.shape[1]}")
    else:
        pt = np.asarray(z["pulse_t"], float) if "pulse_t" in z.files else None
        if fs_hz is None and pt is not None and pt.size > 1:
            fs_hz = 1.0 / float(pt[1] - pt[0])
            notes.append("no t_axis in file: sampling rate taken from pulse_t, time zero = first sample")
        if fs_hz is None:
            raise KeyError(f"no time axis found among {_T_KEYS} and no pulse_t; pass t_key='...' or fs_hz=...")
        t = np.arange(inc.shape[1]) / fs_hz
    amp = next((np.asarray(z[k], float) for k in _AMP_KEYS if k in z.files), None)
    ex = ez = None
    if "element_x" in z.files:
        eu = _UNIT.get(str(_scalar(z, "element_x_units", "m")).strip(), 1.0)
        ex = np.asarray(z["element_x"], float).ravel() * eu
        ez = (np.asarray(z["element_z"], float).ravel() * eu) if "element_z" in z.files else np.zeros_like(ex)
        if "element_x_units" not in z.files:
            notes.append("element_x_units missing -> assumed metres")
    pt = np.asarray(z["pulse_t"], float).ravel() if "pulse_t" in z.files else None
    pp = np.asarray(z["pulse_p"], float).ravel() if "pulse_p" in z.files else None
    c = next((float(_scalar(z, k)) for k in ("sound_speed", "c0", "c") if k in z.files and z[k].ndim == 0), None)
    used = {ik, "scatterer_xyz", "scatterer_xyz_columns", "scatterer_xyz_units", "incident_pressure_row_labels",
            "pulse_t", "pulse_p", "element_x", "element_x_units", "element_z", tk, *_AMP_KEYS}
    return APForward(inc, t, xyz, labels, amp, ex, ez, pt, pp, c,
                     [k for k in z.files if k not in used], notes)


# ---------------------------------------------------------------------------
def emulate_ap_forward(path, scatterer_xyz, *, n_el=64, pitch=3.0e-4, f0=3.5e6, fs=40e6, c=1540.0,
                       fractional_bandwidth=0.6, focus=None, amp=None, spreading_ref=0.01):
    """STAND-IN for forward_propagation() with the same keys - for testing only.
    All elements fire (optionally focused at focus=(x, z) metres); the incident pressure at each
    scatterer is the sum of the delayed, spread pulses from the 64 TX elements."""
    from .interaction import gaussian_pulse
    xyz = np.asarray(scatterer_xyz, float)
    ex = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    sigma_f = fractional_bandwidth * f0 / (2 * np.sqrt(2 * np.log(2)))
    sigma_t = 1 / (2 * np.pi * sigma_f)
    pt = np.arange(-4 * sigma_t, 4 * sigma_t, 1 / fs)
    pp = gaussian_pulse(pt, 0.0, f0, fractional_bandwidth)
    if focus is None:
        tx_delay = np.zeros(n_el)
    else:
        rf_ = np.hypot(focus[0] - ex, focus[1])
        tx_delay = (rf_.max() - rf_) / c
    r = np.hypot(xyz[:, [0]] - ex[None], xyz[:, [2]])
    arr = r / c + tx_delay[None]
    n = int(np.ceil((arr.max() + 6 * sigma_t) * fs))
    t = np.arange(n) / fs
    inc = np.zeros((len(xyz), n))
    for i in range(len(xyz)):
        for e in range(n_el):
            inc[i] += gaussian_pulse(t, arr[i, e], f0, fractional_bandwidth) \
                      * (spreading_ref / max(r[i, e], spreading_ref)) ** 0.5 / n_el
    d = dict(description=np.array("EMULATED forward_propagation() output (test stand-in)"),
             t_axis=t, t_axis_units=np.array("s"), incident_pressure=inc,
             incident_pressure_row_labels=np.array([f"emulated_scatterer_{i}" for i in range(len(xyz))]),
             scatterer_xyz=xyz, scatterer_xyz_columns=np.array(["x", "y", "z"]), scatterer_xyz_units=np.array("m"),
             pulse_t=pt, pulse_p=pp, element_x=ex, element_x_units=np.array("m"))
    if amp is not None:
        d["scatterer_amp"] = np.asarray(amp, float)
    np.savez(path, **d)
    return path
