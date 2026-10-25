import numpy as np
import pytest
from scipy import signal

import tissue_interaction as ti
from tissue_interaction.channel_data import compute_channel_data


def _focus_error(res, x0, z0, c=1540.0):
    an = signal.hilbert(res.channel_rf.astype(np.float64), axis=1)
    fs = 1 / (res.t_rx[1] - res.t_rx[0]); ex, ez = res.element_xz[:, 0], res.element_xz[:, 1]
    xs = np.arange(x0 - 2e-3, x0 + 2e-3, 0.1e-3); zs = np.arange(z0 - 1.5e-3, z0 + 1.5e-3, 0.05e-3)
    X, Z = np.meshgrid(xs, zs)
    t = Z[..., None] / c + np.hypot(X[..., None] - ex, Z[..., None] - ez) / c
    pos = (t - res.t_rx[0]) * fs
    n0 = np.clip(np.floor(pos).astype(int), 0, an.shape[1] - 2); fr = pos - n0; e = np.arange(len(ex))
    img = np.abs((an[e, n0] * (1 - fr) + an[e, n0 + 1] * fr).sum(-1))
    iz, ix = np.unravel_index(img.argmax(), img.shape)
    return xs[ix] - x0, zs[iz] - z0


def _cfg():
    cfg = ti.config_for_body_part("generic"); cfg.f0_hz, cfg.fs_hz = 5e6, 60e6
    return cfg


def test_point_targets_focus_at_true_positions():
    pts = ti.make_point_targets(depths_mm=(20, 40), lateral_mm=(-5.0, 4.0))
    res = ti.run_points(pts * np.array([1e3, 1e3, 1]), _cfg(), units="mm", verbose=False)
    for x, z, _ in pts:
        ex, ez = _focus_error(res, x, z)
        assert abs(ex) < 0.15e-3 and abs(ez) < 0.1e-3


def test_scatterer_list_roundtrips_coordinates_and_units():
    pc = ti.load_points(np.array([[10.0, 20.0, 1.0]]), units="mm")      # x,z,amp in mm
    assert pc.x[0] == pytest.approx(0.01) and pc.z[0] == pytest.approx(0.02) and pc.mode == "scatterers"
    sc = ti.scatterers_from_points(pc, _cfg())
    assert sc.xyz[0, 2] == pytest.approx(0.02) and sc.tof[0] == pytest.approx(0.02 / 1540, rel=1e-3)


def test_csv_tissue_cloud_is_rasterised_and_physics_runs(tmp_path):
    p = tmp_path / "pts.csv"
    rows = ["x_mm,z_mm,tissue"]
    for z in np.arange(0.25, 30, 0.5):
        for x in np.arange(-14.75, 15, 0.5):
            t = "water" if (x ** 2 + (z - 15) ** 2) < 25 else "liver"
            rows.append(f"{x},{z},{t}")
    p.write_text("\n".join(rows))
    pc = ti.load_points(str(p))
    assert pc.mode == "tissue_cloud"
    ph = ti.tissue_cloud_to_phantom(pc)
    assert ph.dx == pytest.approx(0.5e-3, rel=0.02) and set(ph.tissue_names()) >= {"water", "liver"}
    cfg = _cfg(); cfg.max_scatterers = 4000
    res = ti.run_points(str(p), cfg, verbose=False, qa_preview=False)
    assert res.channel_rf.shape[0] == 64 and np.isfinite(res.channel_rf).all()


def test_per_point_properties_without_tissue_names():
    d = dict(x_mm=[0, 1, 2, 3], z_mm=[5, 5, 5, 5], density=[1000, 1000, 1900, 1900],
             speed=[1500, 1500, 3500, 3500], atten_db_cm_mhz=[0.1, 0.1, 13, 13])
    pc = ti.load_points(d)
    ph = ti.tissue_cloud_to_phantom(pc, spacing_mm=1.0)
    assert len(ph.tissue_list) >= 2


def test_bad_points_rejected():
    with pytest.raises(ValueError):
        ti.load_points({"x_mm": [0], "z_mm": [-3], "amp": [1]})            # negative depth
    with pytest.raises(ValueError):
        ti.load_points({"x_mm": [0], "z_mm": [3]})                          # nothing to simulate


def test_element_count_and_custom_positions():
    cfg = _cfg(); cfg.probe.n_elements = 32
    assert cfg.probe.element_positions()[0].shape == (32, 2)
    cfg.probe.element_x_m = [-1e-3, 0.0, 1e-3]
    assert cfg.probe.element_positions()[0].shape == (3, 2)
    cur = ti.config_for_body_part("abdomen").probe.element_positions()[0]
    assert cur[:, 1].min() > -0.0501 and cur[0, 1] < cur[len(cur) // 2, 1] + 1e-9   # arc, edges curve away


def test_channel_energy_is_symmetric_for_centred_target():
    pts = np.array([[0.0, 0.03, 1.0]])
    res = ti.run_points(pts, _cfg(), units="m", verbose=False)
    e = (res.channel_rf.astype(float) ** 2).sum(1)
    assert np.allclose(e, e[::-1], rtol=1e-3)
