"""End-to-end checks of the AP -> TI -> AP relay (128 elements)."""
import numpy as np
import pytest
from scipy import signal

from usoundsim_relay import RelayConfig, run_relay
from nitk_usoundsim import config

PTS = np.array([[0.0, 20.0, 1.0], [5.0, 30.0, 1.0], [-6.0, 12.0, 0.5]])   # x_mm, z_mm, amp


@pytest.fixture(scope="module")
def res():
    return run_relay(PTS, RelayConfig(), units="mm", verbose=False)


def test_128_elements_everywhere(res):
    assert res.received_pressure.shape[0] == 128 == config.N_ELEMENTS
    assert res.element_x.size == 128 and res.cfg.probe.n_elements == 128
    assert res.ti_channel.shape[0] == 128


def test_row_order_and_shapes_are_the_contract(res):
    assert res.incident.shape == res.reflected.shape == (3, res.t_axis.size)
    assert np.allclose(res.scatterers.xyz[:, [0, 2]], PTS[:, :2] * 1e-3)
    assert res.received_pressure.shape == (128, res.t_axis.size)


def test_targets_focus_at_true_coordinates(res):
    """Delay-and-sum of the relay output must put every point target where it was placed."""
    from usoundsim_relay.verify import locate
    for x_mm, z_mm, _ in PTS:
        xb, zb = locate(res.received_pressure, res.t_axis, res.element_x, x_mm * 1e-3, z_mm * 1e-3)
        assert abs(xb - x_mm * 1e-3) < 0.25e-3 and abs(zb - z_mm * 1e-3) < 0.25e-3


def test_tissue_aware_alternative_agrees_on_target_position(res):
    from usoundsim_relay.verify import locate
    xb, zb = locate(res.ti_channel, res.ti_channel_t, res.element_x, 0.0, 0.020)
    assert abs(xb) < 0.3e-3 and abs(zb - 0.020) < 0.3e-3


def test_only_one_attenuation_owner():
    a = run_relay(PTS, RelayConfig(attenuation_owner="ap", also_ti_channel=False), units="mm", verbose=False)
    b = run_relay(PTS, RelayConfig(attenuation_owner="tissue", also_ti_channel=False), units="mm", verbose=False)
    assert a.cfg.path_model == "none" and b.cfg.path_model == "full"
    assert np.isfinite(a.received_pressure).all() and np.isfinite(b.received_pressure).all()


def test_amplitude_scales_linearly():
    a = run_relay(PTS, RelayConfig(also_ti_channel=False), units="mm", verbose=False).received_pressure
    p3 = PTS.copy(); p3[:, 2] *= 3
    b = run_relay(p3, RelayConfig(also_ti_channel=False), units="mm", verbose=False).received_pressure
    assert np.allclose(b, 3 * a, rtol=1e-4, atol=1e-6 * np.abs(a).max())


def test_files_written(tmp_path):
    r = run_relay(PTS, RelayConfig(also_ti_channel=False), units="mm", outdir=str(tmp_path), verbose=False)
    d = np.load(tmp_path / "receiver_bf_input.npz")
    assert d["received_pressure"].shape[0] == 128 and d["element_x"].size == 128
    for k in ("01_scatterers", "02_part1", "03_reflected", "04_part2"):
        assert k in r.files
