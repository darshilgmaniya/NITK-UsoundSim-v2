# Physics model of the Tissue Interaction module

Notation: rho density, c speed of sound, Z = rho c impedance, f0 centre frequency, lambda = c0/f0.

## 1. Tissue -> acoustic maps
Every pixel of the user phantom is mapped to a tissue from the library (or a custom tissue). The module derives
rho, c, Z, attenuation alpha(f) = alpha0 f_MHz^y [dB/cm] and a backscatter level (dB relative to liver).
Low-resolution phantoms are up-sampled (smooth one-hot interpolation + argmax) to lambda/4 so that boundaries
are placed accurately.

## 2. Interface scatterers (specular reflection)
A scatterer is placed every lambda/2 along every boundary between two tissues. With incidence angle theta1
(between the beam and the local interface normal, estimated from the gradient of the smoothed log-impedance map)
Snell's law gives sin(theta2) = (c2/c1) sin(theta1) and the pressure reflection coefficient

    R(theta1) = (Z2 cos(theta1) - Z1 cos(theta2)) / (Z2 cos(theta1) + Z1 cos(theta2))

Beyond the critical angle the reflection is total (|R| = 1). The sign of R is kept: soft tissue -> gas gives a
negative R, i.e. an inverted echo. Because a rough boundary only returns energy towards the probe when it is
roughly perpendicular to the beam, R is multiplied by a directivity

    D(theta1) = floor + (1 - floor) exp(-(theta1 / theta0)^2)

(theta0 = 10 deg, floor = 0.1 by default). The scatterer amplitude is  R D lambda_s / (lambda F#),
so the points that fall inside one beam width add coherently to about R D.

## 3. Diffuse scatterers (speckle)
Sub-wavelength scatterers are drawn uniformly (default 6 per resolution cell). Amplitudes are Gaussian with
standard deviation  r_ref 10^(backscatter_dB/20) / sqrt(n_cell)  so the speckle statistics do not depend on the
number of scatterers. Tissues below `prune_db` (water, cysts, bile, gas) get no diffuse scatterers (anechoic).
A smooth random field modulates the level (texture_strength) to mimic organ-scale echogenicity changes.
Backscatter amplitude follows (f/f0)^m (m = 1 soft tissue, 2 for Rayleigh scatterers such as blood / microbubbles).

## 4. Path effects (path_model = full)
For every scatterer the straight path from the transducer is integrated through the phantom:

* two-way attenuation  L = 2 sum alpha_k ds   (Np at f0) and effective exponent y_eff, applied as
  exp(-L (f/f0)^y_eff) in the frequency domain -> the spectrum is down-shifted with depth;
* interface transmission loss  prod (1 - R_k^2)  over all boundaries crossed -> acoustic shadows behind bone, gas, lung;
* one-way time of flight  sum ds / c_k  (used by the mock incident wave).

## 5. Reflected waveform

    reflected_i(t) = amp_i * Tx_i * IFFT{ FFT{incident_i}(f) * (f/f0)^m_i * exp(-L_i (f/f0)^y_i) }

## Known simplifications (be ready to explain these at the review)
* Straight rays: no refraction, no multiple scattering, no reverberation. Only the first-order (single) scattering is modelled.
* 2-D scalar model; the scatterer file already carries a `y` column so a 3-D phantom can be added later.
* Linear acoustics: no harmonic imaging, no microbubble non-linearity (contrast agent is modelled by raised linear backscatter).
* Specular + diffuse split is phenomenological, not a full-wave solution (k-Wave / Field II are the reference for that).
* Library values are textbook mid-range numbers, not Philips calibration data.
