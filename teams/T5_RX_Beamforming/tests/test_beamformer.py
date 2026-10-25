"""Tests for the Receiver Beamforming module (synthetic channel data only; no other module needed)."""
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

import receiver_beamforming as rb

C, FS, F0 = 1540.0, 40e6, 5e6
N_EL = 64
PITCH = C / F0 / 2


def pulse(t, f0=F0, n_cycles=3):
    sigma = n_cycles / (2.5 * f0)
    return np.sin(2 * np.pi * f0 * t) * np.exp(-0.5 * (t / sigma) ** 2)


def make_channels(targets, n_el=N_EL, fs=FS, tmax=60e-6, amps=None):
    """Plane-wave transmit (t_tx = z/c); echo of each point target seen by each element."""
    ex = (np.arange(n_el) - (n_el - 1) / 2) * PITCH
    t = np.arange(-1.2e-6, tmax, 1 / fs)
    rx = np.zeros((n_el, t.size))
    for k, (x, z) in enumerate(targets):
        a = 1.0 if amps is None else amps[k]
        tau = z / C + np.hypot(x - ex, z) / C
        rx += a * pulse(t[None, :] - tau[:, None])
    return rx, t, ex


@pytest.fixture(scope="module")
def three_targets():
    tg = [(-2e-3, 12e-3), (0.0, 20e-3), (2.5e-3, 28e-3)]
    rx, t, ex = make_channels(tg)
    inp = rb.make_receiver_input(rx, t, ex, fs_hz=FS, c0=C, f0_hz=F0,
                                 scatterer_xyz=np.array([[x, 0, z] for x, z in tg]))
    return tg, inp


def test_point_targets_focus_at_true_position(three_targets):
    tg, inp = three_targets
    res = rb.beamform(inp)
    for x, z in tg:
        xb, zb = rb.locate_peak(res, x, z)
        assert abs(xb - x) < 0.12e-3 and abs(zb - z) < 0.05e-3


@pytest.mark.parametrize("apod", rb.APODIZATIONS)
def test_every_apodization_runs_and_focuses(three_targets, apod):
    tg, inp = three_targets
    res = rb.beamform(inp, rb.RxConfig(apodization=apod))
    xb, zb = rb.locate_peak(res, *tg[1])
    assert abs(xb - tg[1][0]) < 0.2e-3 and abs(zb - tg[1][1]) < 0.06e-3


def test_full_aperture_and_none_fnumber(three_targets):
    tg, inp = three_targets
    res = rb.beamform(inp, rb.RxConfig(f_number=None))
    xb, zb = rb.locate_peak(res, *tg[1])
    assert abs(xb - tg[1][0]) < 0.2e-3


def test_linear_in_amplitude():
    tg = [(0.0, 15e-3)]
    rx, t, ex = make_channels(tg)
    x_lines = np.linspace(-1e-3, 1e-3, 9)
    cfg = rb.BeamformConfig(SAMPLING_FREQUENCY=FS, SPEED_OF_SOUND=C, MAX_DEPTH=0.025, N_ELEMENTS=N_EL)
    a, _ = rb.das_beamform(rx, ex, x_lines, cfg, t_axis=t)
    b, _ = rb.das_beamform(3.0 * rx, ex, x_lines, cfg, t_axis=t)
    assert np.allclose(b, 3.0 * a, rtol=1e-9, atol=1e-12)


def test_3d_layout_equals_shared_2d_layout():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    x_lines = np.linspace(-2e-3, 2e-3, 5)
    cfg = rb.BeamformConfig(SAMPLING_FREQUENCY=FS, SPEED_OF_SOUND=C, MAX_DEPTH=0.025)
    a, za = rb.das_beamform(rx, ex, x_lines, cfg, t_axis=t)
    rx3 = np.repeat(rx[:, :, None], x_lines.size, axis=2)
    b, zb = rb.das_beamform(rx3, ex, x_lines, cfg, t_axis=t)
    assert np.allclose(a, b) and np.allclose(za, zb)


def test_accepts_team_style_config_object():
    """The Receiver BF team's LiverConfig has get_depth_axis() as a method and different extras."""
    @dataclass
    class LiverConfigLike:
        SPEED_OF_SOUND: float = C
        SAMPLING_FREQUENCY: float = FS
        MAX_DEPTH: float = 0.030
        def get_depth_axis(self):
            return np.arange(0.0, self.MAX_DEPTH, self.SPEED_OF_SOUND / (2 * self.SAMPLING_FREQUENCY))
    rx, t, ex = make_channels([(0.0, 15e-3)])
    out, z = rb.das_beamform(rx, ex, np.array([0.0]), LiverConfigLike(), apply_focus=True, t_axis=t,
                             apodization="hann", f_number=1.0)
    assert out.shape == (z.size, 1) and z[0] == 0.0 and abs(z[1] - C / (2 * FS)) < 1e-12
    assert abs(z[np.argmax(np.abs(out[:, 0]))] - 15e-3) < 0.3e-3


def test_nearest_tx_equals_vertical_for_plane_wave():
    rx, t, ex = make_channels([(0.0, 20e-3)])
    cfg = rb.BeamformConfig(SAMPLING_FREQUENCY=FS, SPEED_OF_SOUND=C, MAX_DEPTH=0.03)
    xl = np.array([0.0])
    a, _ = rb.das_beamform(rx, ex, xl, cfg, t_axis=t, tx_mode="vertical")
    b, _ = rb.das_beamform(rx, ex, xl, cfg, t_axis=t, tx_mode="nearest")
    i = np.argmax(np.abs(a[:, 0]))
    assert abs(a[i, 0] - b[i, 0]) / abs(a[i, 0]) < 0.05


def test_tx_delays_shift_arrival_consistently():
    """A transmit delay of d on every element delays all echoes by d; 'nearest' with tx_delays=d must undo it."""
    rx0, t, ex = make_channels([(0.0, 20e-3)])
    d = 2e-6
    rx_d, _, _ = make_channels([(0.0, 20e-3)], tmax=62e-6)
    rx_d = np.roll(rx_d, int(round(d * FS)), axis=1)          # delay by 2 us
    cfg = rb.BeamformConfig(SAMPLING_FREQUENCY=FS, SPEED_OF_SOUND=C, MAX_DEPTH=0.03)
    t2 = np.arange(-1.2e-6, 62e-6, 1 / FS)
    img, z = rb.das_beamform(rx_d, ex, np.array([0.0]), cfg, t_axis=t2, tx_delays=np.full(N_EL, d))
    assert abs(z[np.argmax(np.abs(img[:, 0]))] - 20e-3) < 0.3e-3


def test_apodization_window_properties():
    u = np.linspace(-1.5, 1.5, 301)
    for name in rb.APODIZATIONS:
        w = rb.apodization_window(name, u)
        assert np.all(w[np.abs(u) > 1] == 0) and np.all(w >= -1e-12) and w.max() <= 1 + 1e-12
        assert np.allclose(w, w[::-1])                         # symmetric
    assert rb.apodization_window("hann", np.array([0.0]))[0] == pytest.approx(1.0)
    assert rb.apodization_window("hann", np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_error_messages():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    cfg = rb.BeamformConfig(SAMPLING_FREQUENCY=FS, SPEED_OF_SOUND=C, MAX_DEPTH=0.02)
    xl = np.array([0.0])
    with pytest.raises(ValueError, match="channels"):
        rb.das_beamform(rx[:-1], ex, xl, cfg, t_axis=t)
    with pytest.raises(ValueError, match="t_axis"):
        rb.das_beamform(rx, ex, xl, cfg, t_axis=t[:-1])
    with pytest.raises(ValueError, match="t_axis is required"):
        rb.das_beamform(rx, ex, xl, cfg)
    with pytest.raises(ValueError, match="NaN"):
        bad = rx.copy(); bad[0, 0] = np.nan
        rb.das_beamform(bad, ex, xl, cfg, t_axis=t)
    with pytest.raises(ValueError, match="apodization"):
        rb.das_beamform(rx, ex, xl, cfg, t_axis=t, apodization="triangle")
    with pytest.raises(ValueError, match="strictly increasing"):
        rb.das_beamform(rx, ex, xl, cfg, t_axis=t[::-1].copy())


def test_transposed_input_is_fixed_with_warning():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inp = rb.make_receiver_input(rx.T, t, ex, fs_hz=FS)
    assert inp.rx.shape == rx.shape and any("transposed" in str(x.message) for x in w)


def test_unsorted_elements_are_sorted_with_their_channels():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    p = np.random.default_rng(1).permutation(N_EL)
    inp = rb.make_receiver_input(rx[p], t, ex[p], fs_hz=FS)
    assert np.allclose(inp.element_x, ex) and np.allclose(inp.rx, rx)


def test_auto_fov_without_scatterer_truth():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    res = rb.beamform(rb.make_receiver_input(rx, t, ex, fs_hz=FS, c0=C))
    assert res.notes and res.beamformed_rf.shape == (res.z_axis.size, res.x_lines.size)
    assert res.x_lines[0] >= ex[0] - 1e-9 and res.x_lines[-1] <= ex[-1] + 1e-9


def test_user_fov_and_line_count():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    inp = rb.make_receiver_input(rx, t, ex, fs_hz=FS, c0=C)
    res = rb.beamform(inp, rb.RxConfig(x_range_m=(-2e-3, 2e-3), z_range_m=(5e-3, 25e-3), n_lines=21))
    assert res.x_lines.size == 21 and res.z_axis[0] >= 5e-3 - 1e-9 and res.z_axis[-1] <= 25e-3 + 1e-9
    deep = rb.beamform(inp, rb.RxConfig(z_range_m=(5e-3, 5.0)))        # deeper than the record: clipped, still valid
    assert deep.z_axis[-1] <= 0.5 * C * inp.t_axis[-1] + 1e-9
    with pytest.raises(ValueError, match="z_range"):
        rb.beamform(inp, rb.RxConfig(z_range_m=(0.2, 0.1)))
    with pytest.raises(ValueError, match="x_range"):
        rb.beamform(inp, rb.RxConfig(x_range_m=(1e-3, -1e-3)))


def test_pluggable_beamformer_is_used():
    rx, t, ex = make_channels([(0.0, 15e-3)])
    inp = rb.make_receiver_input(rx, t, ex, fs_hz=FS, c0=C)
    calls = {}

    def mine(raw, xe, xl, cfg, apply_focus=True, t_axis=None, apodization="hann", f_number=1.0):
        calls["n"] = calls.get("n", 0) + 1
        return rb.das_beamform(raw, xe, xl, cfg, apply_focus, t_axis, apodization, f_number)

    res = rb.beamform(inp, rb.RxConfig(beamformer=mine, n_lines=11, x_range_m=(-1e-3, 1e-3)))
    assert calls["n"] == 1 and res.beamformed_rf.shape[1] == 11


def test_bad_options_rejected():
    for kw in (dict(apodization="x"), dict(tx_mode="x"), dict(f_number=-1), dict(line_spacing_m=0),
               dict(n_lines=0), dict(dynamic_range_db=0)):
        with pytest.raises(ValueError):
            rb.RxConfig(**kw).validate()


def test_bmode_preview_and_save_roundtrip(tmp_path, three_targets):
    tg, inp = three_targets
    res = rb.beamform(inp, rb.RxConfig(dynamic_range_db=40))
    assert res.bmode_db.max() == pytest.approx(0.0) and res.bmode_db.min() >= -40 - 1e-9
    assert res.image_u8.dtype == np.uint8 and res.image_u8.max() == 255
    p = rb.save_beamformed(str(tmp_path / "bf.npz"), res)
    d = np.load(p)
    assert d["beamformed_rf"].shape == res.beamformed_rf.shape
    assert d["z_axis"].shape[0] == res.z_axis.size and d["x_lines"].shape[0] == res.x_lines.size
    assert float(d["dynamic_range_db"]) == 40.0


def test_load_from_npz_roundtrip(tmp_path, three_targets):
    tg, inp = three_targets
    p = tmp_path / "rx.npz"
    np.savez(p, received_pressure=inp.rx, t_axis=inp.t_axis, element_x=inp.element_x,
             sampling_rate_hz=FS, c0_nominal_m_s=C, f0_hz=F0, scatterer_xyz=inp.scatterer_xyz)
    got = rb.load_receiver_input(str(p))
    assert got.rx.shape == inp.rx.shape and got.fs_hz == FS
    with pytest.raises(FileNotFoundError):
        rb.load_receiver_input(str(tmp_path / "nope.npz"))
    np.savez(tmp_path / "bad.npz", received_pressure=inp.rx)
    with pytest.raises(KeyError, match="t_axis"):
        rb.load_receiver_input(str(tmp_path / "bad.npz"))


def test_note_when_scatterers_lie_outside_the_aperture():
    tg = [(0.0, 15e-3), (9e-3, 15e-3)]                        # second target is outside +-(63.5*pitch) = +-4.9 mm
    rx, t, ex = make_channels(tg)
    inp = rb.make_receiver_input(rx, t, ex, fs_hz=FS, c0=C,
                                 scatterer_xyz=np.array([[x, 0, z] for x, z in tg]))
    res = rb.beamform(inp)
    assert any("outside the array" in n for n in res.notes)
    res2 = rb.beamform(inp, rb.RxConfig(x_range_m=(-2e-3, 10e-3)))      # user override: no automatic note
    assert not any("outside the array" in n for n in res2.notes)
