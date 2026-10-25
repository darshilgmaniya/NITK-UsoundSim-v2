# Integration 01 - Acoustic Propagation + Tissue Interaction relay

Joins [`teams/T3_Acoustic_Propagation`](../../teams/T3_Acoustic_Propagation) (128 elements) and
[`teams/T4_Tissue_Interaction`](../../teams/T4_Tissue_Interaction) as four runners. Each hands a labelled baton to the next.

| # | Runner | Baton it produces |
|---|---|---|
| 1 | Tissue Interaction, stage 1 | `scatterer_xyz` (N, 3) |
| 2 | Acoustic Propagation, part 1 | `incident_pressure` (N, T) |
| 3 | Tissue Interaction, stage 2 | `reflected_pressure_input` (N, T) |
| 4 | Acoustic Propagation, part 2 | `received_pressure` (128, T) -> **Receiver Beamforming** |

## Run
```bash
cd integration/01_ap_ti_relay
python run_relay.py --source points                         # 9 wire targets, about 5 s
python run_relay.py --source generic --max-scatterers 12000 # contrast phantom, about 1 min
python run_relay.py --source abdomen --max-scatterers 12000 # about 3 min
python run_relay.py --source my_points.csv                  # x_mm,z_mm,tissue  or  x_mm,z_mm,amp
python run_relay.py --source phantom.png --mapping map.json --spacing-mm 0.2
```
Or open `relay_demo.ipynb` (Colab-ready). From Python:
```python
import _paths
from usoundsim_relay import run_relay, RelayConfig
res = run_relay("generic", RelayConfig(attenuation_owner="tissue"))
res.received_pressure.shape   # (128, T)
```

## Files written to `outputs/<source>/`
| File | Written by | Content |
|---|---|---|
| `01_tissue_scatterers.npz` | runner 1 | scatterer positions, amplitudes, labels |
| `02_part1_forward_propagation_outputs.npz` | runner 2 | `incident_pressure`, `t_axis`, pulse, element positions |
| `03_tissue_interaction_output.npz` | runner 3 | `reflected_pressure_input` |
| `04_part2_return_propagation_outputs.npz` | runner 4 | full record with `received_pressure` |
| **`receiver_bf_input.npz`** | runner 4 | **what the Receiver BF team loads** (`received_pressure`, `t_axis`, `element_x`, ...) |
| `alt_ti_stage3_channel_data.npz` | extra | alternative echo data (tissue-aware delays) |
| `relay_overview.png` | script | one-page picture of the whole relay |

The repository ships only the `points` run and the overview pictures of `generic` and `abdomen`; the large `.npz` files regenerate with the commands above.

## Results
![points](../../docs/images/relay_points.png)

## Options (`RelayConfig`)
| Option | Default | Meaning |
|---|---|---|
| `n_elements`, `pitch_m`, `f0_hz`, `fs_hz`, `c0` | from AP `config.py` (128, lambda/2, 5 MHz, 40 MHz, 1540) | array and medium |
| `attenuation_owner` | `"tissue"` | `"tissue"`: TI attenuates, AP alpha0 = 0. `"ap"`: AP uniform 0.5 dB/MHz/cm, TI off |
| `tau_tx` | `None` | per-element transmit delays [s] |
| `max_scatterers` | 8000 | diffuse-scatterer budget (speed and memory) |
| `also_ti_channel` | `True` | also compute the TI stage 3 alternative |

## Tests (`python -m pytest -q`, 7 tests)
128 elements end to end, contract shapes and row order, point targets focus within 0.25 mm of their true position, TI stage 3 agrees on target position, linear amplitude scaling, attenuation modes, file round trip.

## More
| Folder / file | Content |
|---|---|
| `examples/load_and_beamform.py` | tested example for the Receiver Beamforming team |
| `verification/run_verification.py` | reproduces every number in the verification and performance reports (about 3 min) |
| `relay_demo.ipynb` | step-by-step notebook, Colab-ready |
| [`docs/`](../../docs/README.md) | complete documentation: 15 documents, migration guide, PDF technical report |
