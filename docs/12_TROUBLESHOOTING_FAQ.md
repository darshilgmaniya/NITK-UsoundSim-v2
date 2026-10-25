# Troubleshooting and FAQ

## Troubleshooting
| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nitk_usoundsim` or `tissue_interaction` | modules not on the path | run from `integration/01_ap_ti_relay/` (the scripts import `_paths.py`); in your own script `sys.path.insert(0, "<repo>/integration/01_ap_ti_relay"); import _paths` |
| `ModuleNotFoundError: usoundsim_relay` | running from another folder | same as above |
| `MemoryError` or the process is killed | too many scatterers or too deep a phantom | lower `--max-scatterers`, set `also_ti_channel=False`, crop the phantom; see [09_PERFORMANCE.md](09_PERFORMANCE.md) |
| Warning `max_scatterers=... limits speckle density` | scatterer budget below what the phantom needs | accept for quick runs; raise `--max-scatterers` for final images |
| Run takes minutes | large N or deep phantom | abdomen at 12,000 scatterers is about 3 min; use `generic` or `points` to test |
| `received_pressure` rows do not match my element list | you assumed a different element order | use `element_x` from the file; it increases left to right |
| Targets appear shifted in depth by a constant | forgot that time zero is the pulse centre and `t_axis` starts negative | use `t_axis`, never the sample index, when converting to depth |
| Image has faint arcs below each target | unfocused all-element transmit modelled as z/c | expected; see [10_RECEIVER_BF_HANDOFF.md](10_RECEIVER_BF_HANDOFF.md), section 3 |
| Echoes at depth are weak or absent behind bone or gas | shadowing by Tissue Interaction (`attenuation_owner="tissue"`) | intended; use `"ap"` to see the unshadowed model |
| Amplitudes differ from the old 64-element results | the echo is a sum over 128 unnormalised element contributions | expected: about 1.6x to 1.7x larger on the wire-target test (geometry dependent); compare shapes, not absolute values |
| `ValueError: attenuation_owner must be 'tissue' or 'ap'` | typo | use exactly `"tissue"` or `"ap"` |
| `ValueError: incident must have shape (N, n_samples)` | scatterer list and incident array out of step | do not reorder or filter one without the other |
| Point file loads but depths are 1000x too large or small | wrong units | pass `units="m"` or `"mm"` to match the file |
| Notebook cannot import | Colab started in the wrong folder | `%cd NITK-UsoundSim/integration/01_ap_ti_relay` before running cell 1 |
| Tests in Tissue Interaction fail with `No module named tissue_interaction` | missing `PYTHONPATH=src` | `cd teams/T4_Tissue_Interaction && PYTHONPATH=src python -m pytest -q` |
| `check_arr.py` in the AP module fails | it reads a file the demo writes | run `demo_acoustic_propagation.py` first |

## FAQ
**Why 128 elements?** Doubling the aperture roughly halves the lateral beam width (verified: 1.7x to 1.9x). Details: [MIGRATION_64_TO_128.md](MIGRATION_64_TO_128.md).

**Can I use another element count?** Yes: `RelayConfig(n_elements=96)` or `--n-elements 96`, or edit `N_ELEMENTS` in `config.py`. Pitch stays lambda/2, so the aperture scales.

**Why is the pitch 0.154 mm and not 0.30 mm?** `PITCH = LAMBDA/2` at 5 MHz in the Team 3 config. A real L12-4-like probe has a larger pitch; change `PITCH` in `config.py` (an unsteered array shows grating lobes once the pitch exceeds lambda).

**Which echo file should I use?** `receiver_bf_input.npz`. The TI stage 3 file is an alternative with its own physics; never combine.

**Is the output in pascals?** No, arbitrary units, linear in scatterer strength.

**Does the relay add noise?** No. Add noise after loading if you need it.

**Is transmit focused?** No, unfocused by default. Provide `tau_tx` for focusing.

**Where is attenuation applied?** By Tissue Interaction by default; see [02_USER_GUIDE.md](02_USER_GUIDE.md) section 5.

**Why do AP part 2 and TI stage 3 files have different lengths?** Stage 3 uses tissue-specific speed of sound and therefore longer times of flight, so its time axis is longer.

**Can I run only part of the chain?** Call the module functions directly: `forward_propagation_many`, `compute_reflected_pressure`, `return_propagation_many` (see [05_API_REFERENCE.md](05_API_REFERENCE.md)).

**Does it run on a GPU?** No, NumPy on CPU.

**How do I cite or credit?** Cite the repository and the teams listed in the root README (Team 3 Acoustic Propagation, Subtask 04 Tissue Interaction).
