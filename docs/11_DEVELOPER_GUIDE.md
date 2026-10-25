# Developer Guide

## 1. Repository map
```
teams/T3_Acoustic_Propagation/code/nitk_usoundsim/   config.py, acoustic_propagation.py, demo_acoustic_propagation.py, tests
teams/T4_Tissue_Interaction/src/tissue_interaction/  phantom*, scatterers.py, interaction.py, channel_data.py, outputs.py, ...
integration/01_ap_ti_relay/usoundsim_relay/            relay.py (the four runners), verify.py
integration/01_ap_ti_relay/                            run_relay.py, _paths.py, conftest.py, tests/, verification/, examples/, relay_demo.ipynb
docs/                                                  this documentation
```

## 2. Layer rules
1. A module never imports another module.
2. The integration owns data hand-over only: no physics, no unit conversion beyond what a contract requires.
3. `config.py` of Acoustic Propagation is the single source of truth for the array.
4. A change to a module goes in that module's folder, with its own tests, and an entry in `CHANGELOG.md`.

## 3. Setup for development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Run suites (also what CI runs):
```bash
(cd teams/T3_Acoustic_Propagation/code && python -m pytest -q nitk_usoundsim)
(cd teams/T4_Tissue_Interaction && PYTHONPATH=src python -m pytest -q)
(cd integration/01_ap_ti_relay && python -m pytest -q)
python integration/01_ap_ti_relay/verification/run_verification.py   # regenerates docs/verification (about 3 min)
```

## 4. How the relay is built (`relay.py`)
`run_relay` runs four blocks in order; each block times itself (`timings`), logs the arrays it hands over (`_baton`), and optionally writes its file.
1. `_runner1_scatterers(source, cfg, ...)`: dispatches by source type (`Phantom`, preset name, points, image) to Tissue Interaction.
2. `_time_axis(xyz, ex, rc, half_width)`: one shared time axis for all scatterers.
3. `ap.forward_propagation_many` then `compute_reflected_pressure` then `ap.return_propagation_many`.
4. optional `compute_channel_data` (alternative).

## 5. Coding conventions
| Topic | Rule |
|---|---|
| Units | SI in code (m, s); millimetres only at user-facing edges |
| Arrays | (N, T) per scatterer, (M, T) per element; row order is a contract |
| Dtypes | float32 for large (N,T) arrays, float64 for axes and positions |
| Docstrings | state shapes and units of every array argument |
| Randomness | pass a seed; never rely on global state |
| Tests | each public function: shape, a physical check, an edge case |
| Style | PEP 8, type hints on public functions, no wildcard imports |

## 6. Adding the next module (next runner)
1. Put the team's project in `teams/Tn_name/` unchanged. It must run and test alone.
2. Agree the baton: arrays, shapes, units. Add to [06_DATA_DICTIONARY.md](06_DATA_DICTIONARY.md) (or a new data dictionary for the new integration).
3. Create `integration/NN_a_b_relay/` from `01_ap_ti_relay` as a template: extend `_paths.py`, add the runner to the relay, log its baton with `_baton`, add timing.
4. Tests: shapes and row order; one physical check (for example focus position); one file round trip.
5. Update CI (`.github/workflows/ci.yml`), the module table in the root `README.md`, `CHANGELOG.md`.
Planned order: Transmit Beamforming, then the AP/TI relay (done), then Receiver Beamforming, then image formation.

## 7. Testing strategy
| Level | Where | Example |
|---|---|---|
| Unit | module test folders | pulse shape, distances, delays |
| Equivalence | `test_batch_equivalence.py` | fast code equals reference code |
| Integration | `tests/test_relay.py` | contract shapes, row order, focusing |
| Validation | `verification/run_verification.py` | resolution, linearity, attenuation ownership |
When you change maths, add a test that fails on the old behaviour first.

## 8. Releasing
1. All three suites green; `run_verification.py` results unchanged or explained.
2. Update `CHANGELOG.md` and the version note in the root README.
3. Tag `vX.Y.Z`; attach a zip of the repository.

## 9. Git workflow
Branch per change (`feature/<name>`, `fix/<name>`), pull request with the test output, one reviewer from the affected module's team. Commit style: `module: short imperative summary`.
