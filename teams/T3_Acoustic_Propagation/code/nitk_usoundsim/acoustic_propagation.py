"""
acoustic_propagation.py
========================

Acoustic Propagation module for NITK-UsoundSim (Phase 1).

Implements a SIMPLE, FIELD-II-STYLE LINEAR SPATIAL IMPULSE RESPONSE MODEL
for an N_ELEMENTS-element linear array (128 in the integrated pipeline;
set in config.py). This is an educational simplification of
the real spatial-impulse-response formalism used by Field II, not a
full acoustic-field simulator (no element directivity/apodization,
no diffraction-limited beam profile, no nonlinear/pseudospectral
propagation). Those belong to a later, more realistic propagation model.

Signal flow (this module owns everything from the TX pulse up to the
mechanical pressure arriving back at the N_ELEMENTS receive elements)::

    N_ELEMENTS (128) TX elements
         |
    forward_propagation()        -> incident mechanical wave at scatterer
         |
    mock_tissue_interaction()    -> reflected mechanical wave  (TEMPORARY,
         |                           swappable stand-in for the real
         |                           Tissue Interaction module)
    return_propagation()         -> pressure[N_ELEMENTS RX elements, time]

The data stays a MECHANICAL ACOUSTIC PRESSURE WAVEFORM throughout - this
module never touches RF/electrical channel data (that is a later stage,
e.g. transducer receive sensitivity + electronics, owned by another
module).

Units
-----
- Distances: metres.
- Time: seconds.
- Frequency: Hz internally; converted to MHz only where ALPHA_0's
  dB/MHz/cm convention requires it.
- ALPHA_0 is a dB/MHz/cm *amplitude* attenuation coefficient (see
  calculate_attenuation for the exact dB -> linear-amplitude conversion).

All physical constants come from ``nitk_usoundsim.config`` and are never
redefined here.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from nitk_usoundsim.config import (
    ALPHA_0,
    C,
    F0,
    FS,
    N_ELEMENTS,
    PHANTOM_POINTS,
    PITCH,
)

# ----------------------------------------------------------------------
# Assumption: transmitted pulse shape
# ----------------------------------------------------------------------
# The project brief leaves the exact pulse duration/envelope to be chosen
# reasonably. We use a Gaussian-modulated sinusoid
#
#     p(t) = sin(2*pi*F0*t) * exp(-t^2 / (2*sigma^2))
#
# with sigma set so the envelope spans approximately N_CYCLES periods of
# the F0 carrier (sigma = N_CYCLES / (2*F0)). N_CYCLES = 3 is a typical
# choice for a diagnostic-imaging-like transmit pulse: short enough for
# reasonable axial resolution, long enough to have a well-defined centre
# frequency. This is documented here as the single assumption governing
# pulse shape; change N_CYCLES to explore other pulse lengths.
N_CYCLES = 3


def _pulse_shape(t: np.ndarray, f0: float = F0, n_cycles: int = N_CYCLES) -> np.ndarray:
    """
    Closed-form Gaussian-modulated sinusoidal pulse, centred at t = 0.

    Physical meaning: the mechanical pressure waveform emitted by a
    transducer element, evaluated at arbitrary time(s) ``t``. Because
    this is a closed-form function of time (not a fixed sample array),
    it can be evaluated directly at any propagation-delayed time
    (t - t_TX(n)) without needing to resample/interpolate — this keeps
    forward propagation numerically exact regardless of delay size.

    Parameters
    ----------
    t : array_like
        Time(s) [s] at which to evaluate the pulse, relative to the
        pulse centre (t=0 is the envelope peak).
    f0 : float
        Carrier (centre) frequency [Hz].
    n_cycles : int
        Approximate number of carrier cycles under the Gaussian envelope.

    Returns
    -------
    np.ndarray
        Pressure amplitude (arbitrary units, peak ~= 1) at each time in t.
    """
    sigma = n_cycles / (2.0 * f0)
    envelope = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return np.sin(2.0 * np.pi * f0 * t) * envelope


def generate_pulse(
    fs: float = FS, f0: float = F0, n_cycles: int = N_CYCLES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a short, sampled Gaussian-modulated sinusoidal TX pulse for
    display/reference purposes (e.g. plotting "the transmitted pulse").

    Internally, forward/return propagation do NOT resample this array;
    they re-evaluate ``_pulse_shape`` directly at whatever delayed time
    is needed. This function exists to give a concrete, sampled
    ``p(t)`` waveform matching the brief's example::

        p(t) = sin(2*pi*F0*t) * Gaussian_envelope(t)

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz] (defaults to the global FS).
    f0 : float
        Carrier frequency [Hz] (defaults to the global F0).
    n_cycles : int
        Number of carrier cycles under the envelope (see N_CYCLES).

    Returns
    -------
    t : np.ndarray
        Time axis [s], centred at 0.
    p : np.ndarray
        Sampled pulse amplitude at each time in ``t``.
    """
    sigma = n_cycles / (2.0 * f0)
    t_half_width = 4.0 * sigma  # generous support: envelope is ~e^-8 there
    t = np.arange(-t_half_width, t_half_width, 1.0 / fs)
    p = _pulse_shape(t, f0=f0, n_cycles=n_cycles)
    return t, p


def create_linear_array(
    n_elements: int = N_ELEMENTS, pitch: float = PITCH
) -> np.ndarray:
    """
    Compute the lateral (x) position of every element of a linear array,
    centred on x = 0 (the array's geometric centre defines the imaging
    origin, consistent with the symmetric X_MIN/X_MAX FOV in config.py).

    Parameters
    ----------
    n_elements : int
        Number of elements (defaults to the global N_ELEMENTS).
    pitch : float
        Centre-to-centre element spacing [m] (defaults to the global PITCH).

    Returns
    -------
    np.ndarray, shape (n_elements,)
        x-position [m] of each element, element 0 at the most negative x.
    """
    indices = np.arange(n_elements) - (n_elements - 1) / 2.0
    return indices * pitch


def calculate_tx_distance(
    element_x: np.ndarray, x_s: float, z_s: float
) -> np.ndarray:
    """
    One-way distance from transmit element(s) to a scatterer at (x_s, z_s).

        d_TX(n) = sqrt((x_s - x_n)^2 + z_s^2)

    Parameters
    ----------
    element_x : array_like
        TX element x-position(s) [m].
    x_s, z_s : float
        Scatterer lateral / axial (depth) position [m].

    Returns
    -------
    np.ndarray
        Distance [m] from each TX element to the scatterer.
    """
    element_x = np.asarray(element_x, dtype=float)
    return np.sqrt((x_s - element_x) ** 2 + z_s ** 2)


def calculate_rx_distance(
    element_x: np.ndarray, x_s: float, z_s: float
) -> np.ndarray:
    """
    One-way distance from a scatterer at (x_s, z_s) to receive element(s).

        d_RX(m) = sqrt((x_s - x_m)^2 + z_s^2)

    Geometrically identical to calculate_tx_distance (the array is
    reciprocal - same elements act as TX and RX here); kept as a
    separate, explicitly-named function so TX/RX paths stay independently
    readable/testable and so RX-specific effects (e.g. receive-aperture
    truncation, element directivity) can be added later without touching
    the TX path.

    Parameters
    ----------
    element_x : array_like
        RX element x-position(s) [m].
    x_s, z_s : float
        Scatterer lateral / axial (depth) position [m].

    Returns
    -------
    np.ndarray
        Distance [m] from the scatterer to each RX element.
    """
    element_x = np.asarray(element_x, dtype=float)
    return np.sqrt((x_s - element_x) ** 2 + z_s ** 2)


def calculate_propagation_delay(distance: np.ndarray, c: float = C) -> np.ndarray:
    """
    Convert a one-way propagation distance to a propagation time.

        t = d / C

    Parameters
    ----------
    distance : array_like
        Propagation distance [m].
    c : float
        Speed of sound [m/s] (defaults to the global C).

    Returns
    -------
    np.ndarray
        Propagation time [s].
    """
    return np.asarray(distance, dtype=float) / c


def calculate_attenuation(
    distance: np.ndarray, f0: float = F0, alpha0: float = ALPHA_0
) -> np.ndarray:
    """
    One-way amplitude attenuation factor for a mechanical pressure wave.

    ALPHA_0 is given in dB / MHz / cm (a standard soft-tissue attenuation
    convention). To use it we must convert:

        - frequency:  Hz  -> MHz   (divide by 1e6)
        - distance:   m   -> cm    (multiply by 100)

    then the attenuation in dB is

        atten_dB = ALPHA_0 * f0_MHz * distance_cm

    and the dB -> linear AMPLITUDE conversion (not power, hence /20 and
    not /10) is

        atten_linear = 10 ** (-atten_dB / 20)

    Parameters
    ----------
    distance : array_like
        One-way propagation distance [m].
    f0 : float
        Frequency at which attenuation is evaluated [Hz] (defaults to F0).
    alpha0 : float
        Attenuation coefficient [dB/MHz/cm] (defaults to the global ALPHA_0).

    Returns
    -------
    np.ndarray
        Dimensionless linear amplitude attenuation factor in (0, 1].
    """
    distance = np.asarray(distance, dtype=float)
    f0_mhz = f0 / 1.0e6          # Hz -> MHz
    distance_cm = distance * 100.0  # m -> cm
    atten_db = alpha0 * f0_mhz * distance_cm
    atten_linear = 10.0 ** (-atten_db / 20.0)
    return atten_linear


def forward_propagation(
    scatterer: Dict[str, float],
    element_x: np.ndarray,
    t_axis: np.ndarray,
    f0: float = F0,
    c: float = C,
    alpha0: float = ALPHA_0,
    n_cycles: int = N_CYCLES,
    tau_tx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward propagation: transducer (all TX elements) -> scatterer.

    The incident pressure field AT THE SCATTERER is the linear
    superposition of the (delayed, attenuated) pulse contributions
    arriving from every one of the N_ELEMENTS transmit elements -
    exactly the "incident mechanical wave" in the module diagram.

        d_TX(n)  = sqrt((x_s - x_n)^2 + z_s^2)
        t_TX(n)  = d_TX(n) / C
        incident_wave(t) = sum_n  A_TX(n) * p(t - tau_TX(n) - t_TX(n))

    ``tau_tx`` is an explicit hook for a future transmit
    beamforming/focusing delay (t_total = tau_TX(n) + t_TX(n) + t_RX(m)).
    It defaults to all-zeros ("zero/default transmit delays"), as
    instructed - no focusing implementation is invented here.

    Parameters
    ----------
    scatterer : dict
        Must contain "x" and "z" (scatterer position [m]). "amp" is not
        used here - amplitude is applied by mock_tissue_interaction().
    element_x : array_like, shape (n_elements,)
        TX element x-positions [m] (from create_linear_array()).
    t_axis : array_like, shape (n_samples,)
        Shared simulation time axis [s] the incident wave is evaluated on.
    f0, c, alpha0, n_cycles : float / int
        Physical parameters, default to the global config values.
    tau_tx : array_like, optional
        Per-element transmit delay [s] (beamforming/focusing). Defaults
        to zeros (no focusing) if not supplied.

    Returns
    -------
    incident_wave : np.ndarray, shape (n_samples,)
        Incident mechanical pressure waveform at the scatterer, sampled
        on t_axis.
    d_tx : np.ndarray, shape (n_elements,)
        One-way TX distance for each element (returned for diagnostics/plots).
    t_tx : np.ndarray, shape (n_elements,)
        One-way TX propagation delay for each element (diagnostics/plots).
    """
    element_x = np.asarray(element_x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)

    d_tx = calculate_tx_distance(element_x, scatterer["x"], scatterer["z"])
    t_tx = calculate_propagation_delay(d_tx, c=c)
    atten_tx = calculate_attenuation(d_tx, f0=f0, alpha0=alpha0)

    if tau_tx is None:
        tau_tx = np.zeros_like(t_tx)
    else:
        tau_tx = np.asarray(tau_tx, dtype=float)

    incident_wave = np.zeros_like(t_axis)
    for n in range(element_x.size):
        delay_n = tau_tx[n] + t_tx[n]
        incident_wave += atten_tx[n] * _pulse_shape(
            t_axis - delay_n, f0=f0, n_cycles=n_cycles
        )

    return incident_wave, d_tx, t_tx


def mock_tissue_interaction(
    incident_wave: np.ndarray, scatterer: Dict[str, float]
) -> np.ndarray:
    """
    TEMPORARY synthetic Tissue Interaction interface (Part 2).

    The real Tissue Interaction module (owned by another team) is not
    ready yet, so this stand-in models the scatterer as simply scaling
    the incident mechanical wave by its "amp":

        reflected_wave = incident_wave * scatterer["amp"]

    No acoustic impedance, reflection coefficients, or speckle physics
    are modelled - this is intentionally the simplest possible
    placeholder. The function's contract (incident wave in, reflected
    wave out, same length/sampling) is what return_propagation() relies
    on, so a real Tissue Interaction module can later be swapped in here
    without changing return_propagation() at all.

    Parameters
    ----------
    incident_wave : np.ndarray
        Incident mechanical pressure waveform at the scatterer (output
        of forward_propagation()).
    scatterer : dict
        Must contain "amp" (dimensionless synthetic reflectivity).

    Returns
    -------
    np.ndarray
        Reflected/scattered mechanical pressure waveform, same shape as
        incident_wave.
    """
    return incident_wave * scatterer["amp"]


def return_propagation(
    reflected_wave: np.ndarray,
    t_axis: np.ndarray,
    scatterer: Dict[str, float],
    element_x: np.ndarray,
    f0: float = F0,
    c: float = C,
    alpha0: float = ALPHA_0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return propagation: scatterer -> every receive element (Part 3).

        d_RX(m) = sqrt((x_s - x_m)^2 + z_s^2)
        t_RX(m) = d_RX(m) / C
        p_RX,m(t) = A_RX(m) * p_scattered(t - t_RX(m))

    ``reflected_wave`` is already a discretely SAMPLED array (it came out
    of mock_tissue_interaction(), not a closed-form pulse), so shifting it
    by a continuous delay t_RX(m) is done by linear interpolation onto the
    shared time axis (np.interp), rather than re-evaluating a formula.

    Parameters
    ----------
    reflected_wave : np.ndarray, shape (n_samples,)
        Reflected mechanical pressure waveform at the scatterer, sampled
        on t_axis (output of mock_tissue_interaction() or, later, a real
        Tissue Interaction module - this function does not care which).
    t_axis : np.ndarray, shape (n_samples,)
        Shared simulation time axis [s].
    scatterer : dict
        Must contain "x" and "z" (scatterer position [m]).
    element_x : array_like, shape (n_elements,)
        RX element x-positions [m].
    f0, c, alpha0 : float
        Physical parameters, default to the global config values.

    Returns
    -------
    pressure : np.ndarray, shape (n_elements, n_samples)
        Returning mechanical acoustic pressure at every receive element.
    d_rx : np.ndarray, shape (n_elements,)
        One-way RX distance for each element (diagnostics/plots).
    t_rx : np.ndarray, shape (n_elements,)
        One-way RX propagation delay for each element (diagnostics/plots).
    """
    element_x = np.asarray(element_x, dtype=float)
    t_axis = np.asarray(t_axis, dtype=float)
    reflected_wave = np.asarray(reflected_wave, dtype=float)

    d_rx = calculate_rx_distance(element_x, scatterer["x"], scatterer["z"])
    t_rx = calculate_propagation_delay(d_rx, c=c)
    atten_rx = calculate_attenuation(d_rx, f0=f0, alpha0=alpha0)

    n_elements = element_x.size
    pressure = np.zeros((n_elements, t_axis.size))
    for m in range(n_elements):
        # p_scattered(t - t_RX(m)), evaluated on the shared time axis by
        # interpolating the sampled reflected_wave. Samples that would
        # fall outside the simulated time window are treated as zero.
        shifted = np.interp(
            t_axis - t_rx[m], t_axis, reflected_wave, left=0.0, right=0.0
        )
        pressure[m, :] = atten_rx[m] * shifted

    return pressure, d_rx, t_rx


def simulate_acoustic_propagation(
    phantom_points: List[Dict[str, float]] = None,
    n_elements: int = N_ELEMENTS,
    pitch: float = PITCH,
    f0: float = F0,
    fs: float = FS,
    c: float = C,
    alpha0: float = ALPHA_0,
    n_cycles: int = N_CYCLES,
    time_margin_fraction: float = 0.15,
) -> Dict[str, object]:
    """
    Run the complete Phase 1 acoustic propagation pipeline:

        TX elements -> forward_propagation -> mock_tissue_interaction
                     -> return_propagation -> pressure[N_ELEMENTS, time]

    for every synthetic scatterer in ``phantom_points``, combining
    multiple scatterers' echoes by linear superposition at each receive
    element.

    Parameters
    ----------
    phantom_points : list of dict, optional
        Synthetic scatterers, each with "x", "z", "amp". Defaults to the
        global PHANTOM_POINTS.
    n_elements, pitch, f0, fs, c, alpha0, n_cycles : physical parameters
        Default to the global config values.
    time_margin_fraction : float
        Extra fraction of the max round-trip time added as head/tail
        margin on the simulation time axis, so delayed pulses are not
        clipped.

    Returns
    -------
    dict with keys:
        "t_axis"        : np.ndarray, shape (n_samples,) - shared time axis [s]
        "pressure"      : np.ndarray, shape (n_elements, n_samples) - FINAL
                           module output: received mechanical pressure at
                           every RX element vs time, ALL scatterers summed
                           (pressure[receive_element, time], per the spec)
        "received_pressure" : same array as "pressure" (clearer alias)
        "incident_pressure" : np.ndarray, shape (n_scatterers, n_samples) -
                           forward_propagation()'s incident wave AT EACH
                           SCATTERER'S LOCATION vs time, one row per
                           phantom point (row i <-> phantom_points[i])
        "reflected_pressure" : np.ndarray, shape (n_scatterers, n_samples) -
                           mock_tissue_interaction()'s reflected wave AT
                           EACH SCATTERER'S LOCATION vs time, same row order
        "element_x"     : np.ndarray, shape (n_elements,) - array geometry
        "pulse_t"       : np.ndarray - TX pulse time axis (for plotting)
        "pulse_p"       : np.ndarray - TX pulse waveform (for plotting)
        "per_scatterer" : list of dict, one per phantom point, each with
                           "scatterer", "d_tx", "t_tx", "d_rx", "t_rx",
                           "incident_wave", "reflected_wave" (both at that
                           scatterer's location vs time), and "pressure"
                           (that scatterer's own RX contribution, useful
                           for validating linear superposition)
    """
    if phantom_points is None:
        phantom_points = PHANTOM_POINTS

    element_x = create_linear_array(n_elements=n_elements, pitch=pitch)
    pulse_t, pulse_p = generate_pulse(fs=fs, f0=f0, n_cycles=n_cycles)

    # Size the shared time axis from the largest total (TX+RX) one-way
    # delay across all scatterers/elements, so no delayed pulse is clipped.
    max_total_delay = 0.0
    for scatterer in phantom_points:
        d_tx = calculate_tx_distance(element_x, scatterer["x"], scatterer["z"])
        d_rx = calculate_rx_distance(element_x, scatterer["x"], scatterer["z"])
        t_tx = calculate_propagation_delay(d_tx, c=c)
        t_rx = calculate_propagation_delay(d_rx, c=c)
        max_total_delay = max(max_total_delay, float(np.max(t_tx) + np.max(t_rx)))

    pulse_half_width = pulse_t[-1]  # generate_pulse() is symmetric about 0
    margin = time_margin_fraction * max_total_delay + pulse_half_width
    t_start = -pulse_half_width
    t_end = max_total_delay + margin
    t_axis = np.arange(t_start, t_end, 1.0 / fs)

    pressure_total = np.zeros((n_elements, t_axis.size))
    per_scatterer = []
    incident_list = []   # one incident-wave row per scatterer (per-LOCATION pressure vs time)
    reflected_list = []  # one reflected-wave row per scatterer

    for scatterer in phantom_points:
        incident_wave, d_tx, t_tx = forward_propagation(
            scatterer, element_x, t_axis, f0=f0, c=c, alpha0=alpha0, n_cycles=n_cycles
        )
        reflected_wave = mock_tissue_interaction(incident_wave, scatterer)
        pressure_s, d_rx, t_rx = return_propagation(
            reflected_wave, t_axis, scatterer, element_x, f0=f0, c=c, alpha0=alpha0
        )

        pressure_total += pressure_s  # linear superposition across scatterers
        incident_list.append(incident_wave)
        reflected_list.append(reflected_wave)

        per_scatterer.append(
            {
                "scatterer": scatterer,
                "d_tx": d_tx,
                "t_tx": t_tx,
                "d_rx": d_rx,
                "t_rx": t_rx,
                "incident_wave": incident_wave,    # pressure at THIS scatterer vs time
                "reflected_wave": reflected_wave,  # pressure at THIS scatterer vs time
                "pressure": pressure_s,            # this scatterer's own RX contribution
            }
        )

    return {
        "t_axis": t_axis,
        # Received mechanical pressure at all N_ELEMENTS RX elements vs time -- the
        # module's final output, pressure[receive_element, time].
        "pressure": pressure_total,
        "received_pressure": pressure_total,  # alias, same array, clearer name
        # Incident/reflected pressure AT EACH SCATTERER LOCATION vs time,
        # stacked into one array per stage: shape (n_scatterers, n_time).
        # Row i corresponds to phantom_points[i] / per_scatterer[i].
        "incident_pressure": np.stack(incident_list, axis=0),
        "reflected_pressure": np.stack(reflected_list, axis=0),
        "element_x": element_x,
        "pulse_t": pulse_t,
        "pulse_p": pulse_p,
        "per_scatterer": per_scatterer,
    }


# ======================================================================
# BATCH (vectorised) VERSIONS - added for the integrated relay pipeline
# ======================================================================
# forward_propagation() / return_propagation() handle ONE scatterer per
# call.  A real tissue phantom has thousands of scatterers, so these two
# functions do the SAME maths for all scatterers at once.  They are
# checked against the originals in tests/test_batch_equivalence.py.
def forward_propagation_many(
    scatterer_x: np.ndarray,
    scatterer_z: np.ndarray,
    element_x: np.ndarray,
    t_axis: np.ndarray,
    f0: float = F0,
    c: float = C,
    alpha0: float = ALPHA_0,
    n_cycles: int = N_CYCLES,
    tau_tx: Optional[np.ndarray] = None,
    dtype=np.float32,
    chunk: int = 128,
    support_sigma: float = 5.0,
) -> np.ndarray:
    """
    Incident pressure at MANY scatterers (PART 1, vectorised).

        incident[i, t] = sum_n A_TX(i,n) * p(t - tau_TX(n) - t_TX(i,n))

    Each element's pulse is only evaluated inside +-support_sigma Gaussian
    sigmas around its arrival time (the pulse is ~e^-12 of its peak
    beyond 5 sigma), so cost is N_scat x N_elem x ~120 samples instead
    of N_scat x N_elem x N_samples.

    Returns
    -------
    incident : (n_scatterers, n_samples) array of ``dtype``; row i <-> scatterer i.
    """
    sx = np.asarray(scatterer_x, float).ravel()
    sz = np.asarray(scatterer_z, float).ravel()
    ex = np.asarray(element_x, float).ravel()
    t_axis = np.asarray(t_axis, float)
    n_s, n_t = sx.size, t_axis.size
    dt = float(t_axis[1] - t_axis[0])
    t0 = float(t_axis[0])
    tau = np.zeros_like(ex) if tau_tx is None else np.asarray(tau_tx, float)
    sigma = n_cycles / (2.0 * f0)
    half = int(np.ceil(support_sigma * sigma / dt)) + 1
    offs = np.arange(-half, half + 1)
    out = np.zeros((n_s, n_t), dtype)
    for s in range(0, n_s, chunk):
        sl = slice(s, min(s + chunk, n_s))
        d = np.sqrt((sx[sl, None] - ex[None]) ** 2 + sz[sl, None] ** 2)   # (n, M)
        delay = d / c + tau[None]
        a = calculate_attenuation(d, f0=f0, alpha0=alpha0)
        k0 = np.rint((delay - t0) / dt).astype(np.int64)
        k = k0[..., None] + offs                                           # (n, M, W)
        val = a[..., None] * _pulse_shape(t0 + k * dt - delay[..., None], f0=f0, n_cycles=n_cycles)
        row = np.arange(d.shape[0])[:, None, None] * n_t
        ok = (k >= 0) & (k < n_t)
        flat = (row + k)[ok]
        acc = np.bincount(flat, weights=val[ok], minlength=d.shape[0] * n_t)
        out[sl] = acc.reshape(d.shape[0], n_t)
    return out


def return_propagation_many(
    reflected: np.ndarray,
    t_axis: np.ndarray,
    scatterer_x: np.ndarray,
    scatterer_z: np.ndarray,
    element_x: np.ndarray,
    f0: float = F0,
    c: float = C,
    alpha0: float = ALPHA_0,
    chunk: int = 256,
    rel_threshold: float = 1e-6,
) -> np.ndarray:
    """
    Return propagation for MANY scatterers (PART 2, vectorised):

        pressure[m, t] = sum_i A_RX(i,m) * reflected_i(t - t_RX(i,m))

    Same maths as return_propagation() summed over scatterers (linear
    interpolation for the fractional delay), but each scatterer's
    waveform is trimmed to its non-zero stretch and scatter-added, so
    cost scales with the waveform length, not the full time axis.

    Returns
    -------
    pressure : (n_elements, n_samples) float64 - the module's FINAL output.
    """
    reflected = np.asarray(reflected)
    sx = np.asarray(scatterer_x, float).ravel()
    sz = np.asarray(scatterer_z, float).ravel()
    ex = np.asarray(element_x, float).ravel()
    t_axis = np.asarray(t_axis, float)
    n_s, n_t = reflected.shape
    if sx.size != n_s or sz.size != n_s:
        raise ValueError("reflected has %d rows but %d scatterer positions were given" % (n_s, sx.size))
    dt = float(t_axis[1] - t_axis[0])
    n_m = ex.size
    d = np.sqrt((sx[:, None] - ex[None]) ** 2 + sz[:, None] ** 2)           # (N, M)
    pos = d / c / dt                                                         # delay in samples
    amp = calculate_attenuation(d, f0=f0, alpha0=alpha0)
    n_out = n_t + int(np.ceil(pos.max())) + 3
    out = np.zeros((n_m, n_out))
    peak = np.abs(reflected).max(axis=1)
    active = np.where(peak > 1e-12 * max(peak.max(), 1e-300))[0]
    for s in range(0, active.size, chunk):
        idx = active[s:s + chunk]
        y = reflected[idx].astype(np.float64)
        above = np.abs(y) > (rel_threshold * peak[idx])[:, None]
        first = above.argmax(1)
        last = n_t - 1 - above[:, ::-1].argmax(1)
        w = int((last - first).max()) + 1
        cols = first[:, None] + np.arange(w)[None, :]
        valid = (cols <= last[:, None]) & (cols < n_t)
        seg = np.where(valid, np.take_along_axis(y, np.clip(cols, 0, n_t - 1), 1), 0.0)
        for m in range(n_m):
            p = pos[idx, m]
            n0 = np.floor(p).astype(np.int64)
            fr = p - n0
            wts = seg * amp[idx, m][:, None]
            base = cols + n0[:, None]
            # out[k] = (1-fr) R[k-n0] + fr R[k-n0-1]   (== np.interp with zero fill)
            out[m] += np.bincount(base.ravel(), weights=(wts * (1 - fr)[:, None]).ravel(), minlength=n_out)[:n_out]
            out[m] += np.bincount((base + 1).ravel(), weights=(wts * fr[:, None]).ravel(), minlength=n_out)[:n_out]
    return out[:, :n_t]
