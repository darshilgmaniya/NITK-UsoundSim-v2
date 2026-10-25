"""End-to-end checks of the complete relay: phantom -> TI -> AP -> TI -> AP -> Receiver BF -> image."""
import os

import numpy as np
import pytest

import _paths  # noqa: F401
from usoundsim_relay import RelayConfig
from usoundsim_full import demo_points, run_from_handoff, run_full
import receiver_beamforming as rb

PTS = np.array([[0.0, 20.0, 1.0], [5.0, 30.0, 1.0], [-6.0, 12.0, 0.5]])   # x_mm, z_mm, amp


@pytest.fixture(scope="module")
def full():
    return run_full(PTS, RelayConfig(), units="mm", verbose=False)


def test_five_runners_chain_with_expected_shapes(full):
    r, b = full.relay, full.bf
    assert r.received_pressure.shape[0] == 128
    assert b.input.rx.shape == r.received_pressure.shape                       # runner 4 -> 5 baton unchanged
    assert b.beamformed_rf.shape == (b.z_axis.size, b.x_lines.size)
    assert set(full.timings) >= {"runner1_ti_stage1", "runner2_ap_part1", "runner3_ti_stage2",
                                 "runner4_ap_part2", "runner5_receiver_bf"}


def test_baton_into_beamformer_is_identical_to_relay_output(full):
    assert np.array_equal(full.bf.input.rx, full.relay.received_pressure.astype(np.float64))
    assert np.array_equal(full.bf.input.t_axis, full.relay.t_axis)


def test_image_places_every_target_at_its_true_coordinates(full):
    for x_mm, z_mm, _ in PTS:
        xb, zb = rb.locate_peak(full.bf, x_mm * 1e-3, z_mm * 1e-3)
        assert abs(xb - x_mm * 1e-3) < 0.15e-3 and abs(zb - z_mm * 1e-3) < 0.1e-3


def test_amplitude_scales_linearly_through_the_whole_chain():
    a = run_full(PTS, RelayConfig(), units="mm", verbose=False)
    b = run_full(PTS * np.array([1, 1, 2]), RelayConfig(), units="mm", verbose=False)
    ratio = b.bf.beamformed_rf.max() / a.bf.beamformed_rf.max()
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_hand_off_file_round_trip(tmp_path):
    res = run_full("points", outdir=str(tmp_path), verbose=False)
    for k in ("receiver_bf_input", "05_beamformed"):
        assert os.path.exists(res.files[k])
    again = run_from_handoff(res.files["receiver_bf_input"], verbose=False)
    # the hand-off file stores float32, so agreement is to single precision, not bit-exact
    scale = np.abs(res.beamformed_rf).max()
    assert np.abs(again.beamformed_rf - res.beamformed_rf).max() < 1e-5 * scale
    d = np.load(res.files["05_beamformed"])
    assert d["beamformed_rf"].shape == res.beamformed_rf.shape
    assert {"z_axis", "x_lines", "envelope_preview", "bmode_db_preview"} <= set(d.files)


def test_csv_points_source(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text("x_mm,z_mm,amp\n0,15,1\n3,25,0.8\n")
    res = run_full(str(p), verbose=False)
    xb, zb = rb.locate_peak(res.bf, 3e-3, 25e-3)
    assert abs(xb - 3e-3) < 0.2e-3 and abs(zb - 25e-3) < 0.1e-3


@pytest.mark.parametrize("n_el", [64, 96, 128])
def test_any_element_count_flows_through(n_el):
    res = run_full("points", RelayConfig(n_elements=n_el), verbose=False)
    assert res.bf.input.n_elements == n_el and res.relay.received_pressure.shape[0] == n_el
    xb, zb = rb.locate_peak(res.bf, 0.0, 20e-3)
    assert abs(xb) < 0.25e-3 and abs(zb - 20e-3) < 0.1e-3


def test_ap_attenuation_owner_also_works():
    res = run_full("points", RelayConfig(attenuation_owner="ap"), verbose=False)
    assert np.isfinite(res.bf.beamformed_rf).all() and res.bf.beamformed_rf.std() > 0


def test_single_scatterer_and_shallow_target():
    r1 = run_full(np.array([[0.0, 10.0, 1.0]]), units="mm", verbose=False)
    xb, zb = rb.locate_peak(r1.bf, 0.0, 10e-3)
    assert abs(xb) < 0.2e-3 and abs(zb - 10e-3) < 0.1e-3
    r2 = run_full(np.array([[0.0, 3.0, 1.0]]), units="mm", verbose=False)
    assert np.isfinite(r2.bf.beamformed_rf).all()


def test_focused_transmit_delays_reach_the_beamformer():
    """Per-element TX delays given to the relay must automatically switch the beamformer to tx_mode='nearest'."""
    n, pitch, c = 128, RelayConfig().pitch_m, RelayConfig().c0
    ex = (np.arange(n) - (n - 1) / 2) * pitch
    zf = 20e-3
    tau = (np.hypot(ex, zf).max() - np.hypot(ex, zf)) / c                        # focus on axis at 20 mm
    res = run_full(np.array([[0.0, 20.0, 1.0]]), RelayConfig(tau_tx=tau), units="mm", verbose=False)
    assert res.bf.cfg.tx_mode == "nearest" and np.allclose(res.bf.cfg.tx_delays, tau)
    xb, zb = rb.locate_peak(res.bf, 0.0, 20e-3, half_z=2e-3)
    assert abs(xb) < 0.2e-3 and abs(zb - 20e-3) < 0.15e-3


def test_bad_receiver_options_fail_before_the_slow_part():
    with pytest.raises(ValueError, match="apodization"):
        run_full("points", rx=rb.RxConfig(apodization="nope"), verbose=False)


def test_user_supplied_beamformer_plugs_in():
    seen = {}

    def mine(raw, xe, xl, cfg, apply_focus=True, t_axis=None, apodization="hann", f_number=1.0):
        seen["shape"] = raw.shape
        return rb.das_beamform(raw, xe, xl, cfg, apply_focus, t_axis, apodization, f_number)

    res = run_full("points", rx=rb.RxConfig(beamformer=mine, n_lines=41), verbose=False)
    assert seen["shape"][0] == 128 and res.beamformed_rf.shape[1] == 41
