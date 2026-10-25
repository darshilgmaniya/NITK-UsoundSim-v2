# Acoustic hand-off
Load `outputs/tx_beamforming_package.npz` and use `element_x_m`, `tx_delays_s`, and `tx_weights` directly. Each has shape `(128,)`. Do not create a second element geometry inside Acoustic Propagation. Pass `tx_delays_s` as the per-element transmit delay vector and apply `tx_weights[n]` to element n if apodization is supported.
