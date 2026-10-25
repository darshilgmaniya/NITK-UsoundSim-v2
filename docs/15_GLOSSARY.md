# Glossary

| Term | Meaning |
|---|---|
| Acoustic Propagation (AP) | Module that moves the pulse from the array to the scatterers (forward) and the echoes back (return). |
| Alpha0 | Attenuation coefficient in dB per MHz per cm. |
| Aperture | Width of the active array: (elements - 1) x pitch, 19.56 mm for 128 elements. |
| Apodisation | Weighting of channels (for example Hann) to lower side lobes. |
| Attenuation owner | The module that applies attenuation: Tissue Interaction (default) or Acoustic Propagation. |
| B-mode | Brightness image: envelope of beamformed echoes in decibels. |
| Baton | The array handed from one runner to the next. |
| Beamforming | Combining element signals with delays to focus; delay-and-sum is the simplest form. |
| Contract (interface contract) | Agreed names, shapes, units and row order of the hand-over arrays. |
| Delay-and-sum (DAS) | Beamforming by shifting each channel by the travel time to a pixel and adding. |
| Diffuse scatterer | Tiny random reflector that creates speckle (kind 0). |
| Dynamic focus | Receive focus recomputed for every depth. |
| Echo / RF data | The pressure signal recorded at each element, before beamforming. |
| Envelope | Magnitude of the analytic signal (Hilbert transform). |
| f0 | Centre frequency of the pulse, 5 MHz. |
| FWHM | Full width at half maximum (-6 dB in amplitude), used for resolution. |
| fs | Sampling frequency, 40 MHz. |
| Grating lobes | Extra beams caused by pitch larger than lambda/2 (absent here). |
| Incident pressure | Pressure wave arriving at a scatterer. |
| Interface scatterer | Specular reflector on a tissue boundary (kind 1). |
| Lambda | Wavelength, c / f0 = 0.308 mm. |
| Lateral / axial | Across the beam (x) / along the beam (z, depth). |
| Pitch | Distance between element centres, lambda/2 = 0.154 mm. |
| Path model | Tissue Interaction option: `full` (tissue attenuation, shadows) or `none`. |
| Reflected pressure | Pressure re-radiated by a scatterer after interaction with tissue. |
| Relay | The chain of four runners from phantom to received pressure. |
| Runner | One step of the relay (TI stage 1, AP part 1, TI stage 2, AP part 2). |
| Scatterer | A point where the wave interacts with tissue; row of the (N, T) arrays. |
| Shadowing | Reduced echo behind strongly reflecting or attenuating structures such as bone or gas. |
| Speckle | Granular texture from interference of many diffuse scatterers. |
| Stage 3 | Tissue Interaction's own channel-data model (alternative echo output). |
| t_axis | Shared time axis [s]; time zero = pulse centre at an element with no delay. |
| tau_tx | Per-element transmit delay used for focusing or steering. |
| Tissue Interaction (TI) | Module that turns a phantom into scatterers and converts incident into reflected pressure. |
| Wire target | Single point reflector used to measure resolution. |
