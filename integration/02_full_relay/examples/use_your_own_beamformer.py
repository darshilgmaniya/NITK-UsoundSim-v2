"""Plug the Receiver BF team's OWN ``das_beamform`` (receive_beamforming.py) into the relay.

    python examples/use_your_own_beamformer.py path/to/folder_with_receive_beamforming_py

Any function with this signature works:
    das_beamform(raw_rf, x_elements, x_lines, config, apply_focus=True, t_axis=None,
                 apodization="hann", f_number=1.0)  ->  (beamformed_rf (n_depth, n_lines), z_axis)
`raw_rf` arrives as (n_elements, n_samples) [one record shared by every line = unfocused transmit].
If your function needs (n_elements, n_samples, n_lines), wrap it as shown below.
"""
import importlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _paths  # noqa: F401
import numpy as np
from usoundsim_full import run_full, demo_points
from receiver_beamforming import RxConfig

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
    team = importlib.import_module("receive_beamforming").das_beamform     # <- your real file
else:
    from receiver_beamforming import das_beamform as team                  # stand-in so the example runs


def wrapped(raw, xe, xl, cfg, apply_focus=True, t_axis=None, apodization="hann", f_number=1.0):
    if raw.ndim == 2:                                  # relay layout -> team layout (same record for every line)
        raw = np.repeat(raw[:, :, None], len(xl), axis=2)
    return team(raw, xe, xl, cfg, apply_focus=apply_focus, t_axis=t_axis, apodization=apodization, f_number=f_number)


res = run_full(demo_points(), rx=RxConfig(beamformer=wrapped, n_lines=61), verbose=False)
print("beamformed_rf", res.beamformed_rf.shape, "-> image peak at z index", int(np.argmax(res.bf.envelope.max(1))))
