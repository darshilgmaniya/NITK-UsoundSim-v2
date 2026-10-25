# Physics and Mathematical Model

Symbols: N scatterers (index i), M = 128 elements (index n or m), c = 1540 m/s, f0 = 5 MHz, lambda = c/f0 = 0.308 mm, p = lambda/2 = 0.154 mm pitch, fs = 40 MHz, dt = 1/fs = 25 ns.
All quantities are 2-D (x lateral, z depth); pressure is in arbitrary units.

## 1. Geometry
Elements lie on z = 0, centred on x = 0:

    x_n = (n - (M-1)/2) * p ,   n = 0 .. M-1

For M = 128: x from -9.78 mm to +9.78 mm, aperture D = (M-1) p = 19.56 mm (64 elements: 9.70 mm). A scatterer i sits at (x_i, z_i) (y = 0).
Distance from element n to scatterer i (the same for transmit and receive):

    d(i,n) = sqrt( (x_i - x_n)^2 + z_i^2 )        t(i,n) = d(i,n) / c

## 2. Transmit pulse
Gaussian-modulated sine, evaluated in closed form at any delay (no resampling):

    p(t) = sin(2 pi f0 t) * exp( -t^2 / (2 sigma^2) ),     sigma = n_cycles / (2 f0) = 0.3 us   (n_cycles = 3)

Amplitude spectrum: Gaussian at f0 with sigma_f = 1/(2 pi sigma) = 0.53 MHz, so the -6 dB bandwidth is about 1.25 MHz (25 % fractional). Round-trip envelope FWHM in time is 2.355 sigma = 0.71 us, which is 0.54 mm in depth. This matches the measured axial FWHM of 0.53 to 0.55 mm (V3 in the verification report) and is why axial resolution does not change with element count.
Sampling at fs = 40 MHz is 8 x f0; the pulse content ends near f0 + 4 sigma_f = 7.1 MHz, well below Nyquist (20 MHz).

## 3. Runner 2 - forward propagation (Acoustic Propagation part 1)
Amplitude factor for one-way distance d:

    A(d) = 10 ^ ( - alpha0 * f_MHz * d_cm / 20 )        alpha0 = 0.5 dB/MHz/cm  ->  2.5 dB/cm at 5 MHz

Pressure arriving at scatterer i (all elements fire with optional delay tau_n; default 0):

    incident_i(t) = sum over n of  A(d(i,n)) * p( t - tau_n - t(i,n) )

Not modelled in this part: 1/r spreading, element directivity, tissue heterogeneity (uniform c). With `attenuation_owner="tissue"` alpha0 is set to 0 here and attenuation is applied by Tissue Interaction instead.

## 4. Runner 3 - tissue interaction (stage 2)
For scatterer i with signed reflectivity a_i (negative for tissue-to-gas, which flips polarity):

    reflected_i(t) = a_i * T_i * IFFT{ FFT{incident_i}(f) * H_i(f) }

    H_i(f) = (f/f0)^(m_i)  *  exp( - L_i * (f/f0)^(y_i) )

| Symbol | Meaning | Source |
|---|---|---|
| a_i | intrinsic reflectivity (interface: Fresnel at oblique incidence with specular directivity; diffuse: tissue backscatter level with texture) | stage 1 |
| m_i | backscatter frequency exponent | tissue table |
| L_i | two-way path attenuation at f0, in nepers, integrated along the straight path through the phantom | stage 1 |
| y_i | effective attenuation exponent along the path | stage 1 |
| T_i | two-way interface transmission factor (shadowing behind bone, gas, lung) | stage 1 |
Terms with L_i and T_i apply only for `path_model="full"`. The FFT length is the next power of two at or above 2 x (number of samples), so there is no wrap-around. Details of the scatterer model: `teams/T4_Tissue_Interaction/docs/PHYSICS.md`.

Interface scatterers (kind 1) produce specular echoes at tissue boundaries, spaced lambda/2 along the boundary; diffuse scatterers (kind 0) are random sub-wavelength reflectors that create speckle.

## 5. Runner 4 - return propagation (Acoustic Propagation part 2)
Each scatterer re-radiates its reflected waveform; element m receives the delayed, attenuated sum:

    pressure_m(t) = sum over i of  A(d(i,m)) * reflected_i( t - t(i,m) )

The fractional-sample delay is applied by linear interpolation. The result `received_pressure[m, t]` has shape (M, T) and is the hand-off to Receiver Beamforming.

## 6. Time axis
Time zero is the centre of the transmitted pulse at an element with zero transmit delay. The axis starts at -w with w = 4 sigma = 1.2 us (so the pulse start is included) and ends at

    t_end = (1 + 0.15) * max over i of [ max_n (t(i,n) + tau_n) + max_m t(i,m) ] + 2 w

A 15 % margin covers late arrivals. T = (t_end + w) * fs samples (for example 2,166 for the nine wire targets, about 4,900 for the generic phantom, 8,400 for the abdomen).

## 7. How the batch functions compute this fast
`forward_propagation_many`: each element's pulse is only non-negligible within +-5 sigma of its arrival (e^-12.5 of peak). With sigma = 12 samples this is a 123-sample window, so the work is N x M x 123 evaluations instead of N x M x T. Results are accumulated with `numpy.bincount`.
`return_propagation_many`: each reflected waveform is trimmed to the stretch above 1e-6 of its own peak, then, for every element, scatter-added at integer shift n0 = floor(delay/dt) with weights (1 - f) and f where f = delay/dt - n0. This is identical to `numpy.interp` with zero fill. Cost is N x M x (trimmed length).
Both are checked against the original loops (relative error 6.5e-7 and 6.8e-7).

## 8. What 64 to 128 elements changes
| Quantity | 64 el | 128 el | Note |
|---|---|---|---|
| Aperture D | 9.70 mm | 19.56 mm | doubles |
| Lateral FWHM at z = 20 mm | 0.78 mm | 0.40 mm | about lambda z / D scaling (0.31 mm ideal) |
| Axial FWHM | 0.5 to 0.57 mm | 0.53 to 0.55 mm | unchanged: set by the pulse |
| Echo amplitude (nine wire targets) | received peak 7.3, rms 0.65 | received peak 12.3, rms 1.05 | 1.7x and 1.6x; arbitrary units, not normalised; the factor depends on geometry because unfocused arrivals do not add in phase |
| Compute cost (AP parts) | 1x | about 1.8x | linear in M |
| Grating lobes | none | none | pitch lambda/2 |
The array is now slightly wider than the +-10 mm field of view in `config.py` (+-9.78 mm elements), so edge scatterers are seen at wide angles. AP part 2 has no element directivity; TI stage 3 includes a cosine directivity (see limitations).

## 9. Units and conventions
| Quantity | Unit | Note |
|---|---|---|
| position | m | columns x, y, z; y = 0 |
| time | s | shared `t_axis`, uniform spacing 1/fs |
| pressure | arbitrary | linear in scatterer strength (verified) |
| attenuation coefficient | dB/MHz/cm | converted with 20 log10 (amplitude) |
