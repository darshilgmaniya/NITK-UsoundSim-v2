import numpy as np
import pytest
from scipy import signal

import tissue_interaction as ti

XYZ = np.array([[0.0, 0.0, 0.020], [0.005, 0.0, 0.035], [-0.006, 0.0, 0.050]])


@pytest.fixture(scope="module")
def ap_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("ap") / "forward.npz"
    ti.emulate_ap_forward(str(p), XYZ, f0=3.5e6, fs=40e6)
    return str(p)


def test_reader_maps_every_key(ap_file):
    ap = ti.load_ap_forward(ap_file)
    assert ap.incident.shape[0] == 3 and ap.element_x.size == 64
    assert np.allclose(ap.xyz, XYZ) and len(ap.row_labels) == 3
    assert ti.estimate_f0(ap.pulse_t, ap.pulse_p) == pytest.approx(3.5e6, rel=0.03)


def test_row_i_in_equals_row_i_out_and_labels_pass_through(ap_file):
    res = ti.run_from_ap(ap_file, verbose=False)
    assert res.reflected.shape[0] == 3 and np.allclose(res.scatterers.xyz, XYZ)
    assert list(res.scatterers.row_labels()) == list(ti.load_ap_forward(ap_file).row_labels)
    assert res.channel_rf.shape[0] == 64


def test_echo_arrives_incident_time_plus_return_time(ap_file):
    ap = ti.load_ap_forward(ap_file)
    res = ti.run_from_ap(ap, verbose=False)
    t_in = ap.t_axis[np.abs(signal.hilbert(ap.incident[0])).argmax()]
    env = np.abs(signal.hilbert(res.channel_rf[32]))
    t_rx = res.t_rx[env[: int(0.45 * env.size)].argmax()]
    assert (t_rx - t_in) == pytest.approx(0.020 / 1540, abs=0.1e-6)


def test_receive_elements_follow_ap_element_x(ap_file):
    ap = ti.load_ap_forward(ap_file)
    res = ti.run_from_ap(ap, verbose=False)
    assert np.allclose(res.element_xz[:, 0], ap.element_x)


def test_amplitude_scales_linearly(ap_file):
    a = ti.run_from_ap(ap_file, default_amp=1.0, verbose=False).channel_rf
    b = ti.run_from_ap(ap_file, default_amp=3.0, verbose=False).channel_rf
    assert np.allclose(b, 3 * a, rtol=1e-4, atol=1e-9)


def test_phantom_adds_shadow_to_scatterers_behind_bone(ap_file):
    b = ti.PhantomBuilder(40, 70, 0.2, "generic_soft", "bone_slab")
    b.layer(27, 31, "bone")
    free = ti.run_from_ap(ap_file, verbose=False)
    sh = ti.run_from_ap(ap_file, phantom=b.build(), verbose=False)
    e_free = (free.reflected ** 2).sum(1); e_sh = (sh.reflected ** 2).sum(1)
    assert e_sh[0] == pytest.approx(e_free[0], rel=0.2)       # in front of the bone: unchanged
    assert e_sh[1] < 0.05 * e_free[1] and e_sh[2] < 0.05 * e_free[2]


def test_bad_files_are_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, incident_pressure=np.zeros((4, 50)), scatterer_xyz=np.zeros((3, 3)), t_axis=np.arange(50) * 1e-8)
    with pytest.raises(ValueError, match="row mismatch"):
        ti.load_ap_forward(str(p))
    np.savez(p, incident_pressure=np.zeros((3, 50)), scatterer_xyz=np.ones((3, 3)))
    with pytest.raises(KeyError):
        ti.load_ap_forward(str(p))
