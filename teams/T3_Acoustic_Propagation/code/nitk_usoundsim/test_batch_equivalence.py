"""The vectorised AP functions must reproduce the original single-scatterer ones."""
import numpy as np
from nitk_usoundsim import config, acoustic_propagation as ap

SC = [dict(x=0.0, z=0.012, amp=1.0), dict(x=0.004, z=0.025, amp=0.6), dict(x=-0.006, z=0.031, amp=1.0)]


def _axis(el):
    pt, _ = ap.generate_pulse()
    mx = 0.0
    for s in SC:
        tt = ap.calculate_propagation_delay(ap.calculate_tx_distance(el, s["x"], s["z"]))
        mx = max(mx, 2 * tt.max())
    return np.arange(-pt[-1], mx * 1.15 + pt[-1], 1 / config.FS)


def test_forward_many_matches_loop():
    el = ap.create_linear_array()
    t = _axis(el)
    ref = np.stack([ap.forward_propagation(s, el, t)[0] for s in SC])
    new = ap.forward_propagation_many([s["x"] for s in SC], [s["z"] for s in SC], el, t, dtype=np.float64)
    assert new.shape == ref.shape == (3, t.size)
    assert np.abs(new - ref).max() < 1e-4 * np.abs(ref).max()


def test_return_many_matches_loop_and_sums_scatterers():
    el = ap.create_linear_array()
    t = _axis(el)
    inc = np.stack([ap.forward_propagation(s, el, t)[0] * s["amp"] for s in SC])
    ref = sum(ap.return_propagation(inc[i], t, SC[i], el)[0] for i in range(3))
    new = ap.return_propagation_many(inc, t, [s["x"] for s in SC], [s["z"] for s in SC], el)
    assert new.shape == (config.N_ELEMENTS, t.size)
    assert np.abs(new - ref).max() < 1e-4 * np.abs(ref).max()


def test_element_count_is_128():
    assert config.N_ELEMENTS == 128 and ap.create_linear_array().size == 128
