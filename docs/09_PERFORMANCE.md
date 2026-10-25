# Performance and Scaling

Measured with `integration/01_ap_ti_relay/verification/run_verification.py` (section V7), each case in a fresh Python process. Phantom: `generic`, TI stage 3 disabled. Machine: a shared Linux sandbox container, single run per case; absolute times vary with your CPU, the ratios are what to rely on.

## 1. Measured results
| `max_scatterers` | Scatterers N | Samples T | Elements | TI stage 1 | AP part 1 | TI stage 2 | AP part 2 | Total | Peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 2,083 | 4,832 | 128 | 0.5 s | 1.7 s | 1.5 s | 4.8 s | 8.4 s | 1.04 GB |
| 2,000 | 3,586 | 4,916 | 128 | 0.6 s | 3.1 s | 2.9 s | 6.5 s | 13.3 s | 1.14 GB |
| 8,000 | 9,592 | 4,938 | 128 | 1.1 s | 8.7 s | 7.1 s | 16.1 s | 32.9 s | 1.58 GB |
| 2,000 | 3,586 | 4,780 | **64** | 0.6 s | 1.7 s | 2.6 s | 3.9 s | 8.8 s | 1.14 GB |

Larger case measured once during development (abdomen preset, 12,000 diffuse scatterers): N = 19,740, T = 8,401, about 14 s AP part 1, 20 s TI stage 2, 69 s AP part 2, 39 s TI stage 3.

![scaling](verification/performance_scaling.png)

## 2. How cost scales
Speed-up of the batch functions over the original one-scatterer loops (40 scatterers, 128 elements, 40 mm deep axis): forward 0.35 s to 0.03 s (10x), return 0.11 s to 0.02 s (6x).

| Runner | Cost model | Evidence |
|---|---|---|
| TI stage 1 | about linear in N | 0.5 s to 1.1 s |
| AP part 1 | N x M x 123 pulse evaluations | 0.8 to 0.9 ms per scatterer; halves with 64 elements (1.7 s vs 3.1 s) |
| TI stage 2 | N x T log T (FFT), independent of M | 2.6 s vs 2.9 s for 64 vs 128 elements |
| AP part 2 | N x M x trimmed waveform length | 1.7 to 2.3 ms per scatterer; 3.9 s vs 6.5 s for 64 vs 128 elements |
Rule of thumb: total seconds = 3.4 to 4 ms x N for 128 elements on this machine. Doubling elements roughly doubles the two AP parts only.

## 3. Memory
Dominant arrays (float32): `incident` and `reflected`, each 4 x N x T bytes. N = 9,592, T = 4,938: 190 MB each. Peak RAM also holds float64 temporaries inside the FFT chunk (2,048 rows) and the return step, plus Python and library overhead (about 1 GB baseline). Fitted to the three 128-element runs above: peak RAM about 0.9 GB + 14 bytes x N x T.

## 4. Reducing time and memory
| Action | Effect |
|---|---|
| Lower `--max-scatterers` | fewer scatterers; speckle statistics degrade (a warning is printed) |
| `also_ti_channel=False` | skips TI stage 3 (about a quarter of the abdomen run) |
| Crop the phantom to the aperture width | scatterers far outside +-10 mm contribute little and cost time |
| Smaller depth | T and cost fall linearly with depth |
| Use float32 end to end | already the default for the (N,T) arrays |
| Pass `out=np.lib.format.open_memmap(...)` to `compute_reflected_pressure` | keeps the reflected array on disk (TI feature, not wired into the CLI) |

## 5. Known bottlenecks and ideas (not implemented)
- AP part 2 loops over the 128 elements in Python; moving the inner step to numba or a C extension would give a large speed-up.
- Chunks run on one CPU core; processes per element group would parallelise it.
- Storage of `incident` and `reflected` can be avoided by streaming per chunk of scatterers (the hand-off to Receiver BF only needs the (128, T) result).
