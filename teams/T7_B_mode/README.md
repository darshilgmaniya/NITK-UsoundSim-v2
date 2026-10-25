# Module 04 - B-mode formation (runner 6)

`bmode_formation(beamformed_rf, dynamic_range_db=None) -> bmode`

Envelope (Hilbert along depth), normalise to the frame peak, 20*log10, clip to the dynamic range, map to [0, 1].
Input `(n_depths, n_lines)` beamformed RF (depth on axis 0); output has the same shape.

Source: the first project's `teams/T6_B_mode/bmode_formation.py`. Only change: `config.DYNAMIC_RANGE_DB` became the
module constant `DEFAULT_DYNAMIC_RANGE_DB = 50`. The module docstring has the equations, the assumptions and the limitations.

Test: `python -m pytest -q tests` (3 tests): range and shape, a 20 dB step is recovered as 20 dB, zero input and bad arguments.
Use: `PYTHONPATH=src python -c "from bmode_formation import bmode_formation"`.
