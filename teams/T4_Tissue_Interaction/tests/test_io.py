import json
import numpy as np
import pytest

import tissue_interaction as ti
from tissue_interaction.outputs import (load_incident, load_scatterers, save_reflected, save_scatterers)
from tissue_interaction.interaction import compute_reflected_pressure, make_mock_incident


def test_label_image_roundtrip(tmp_path):
    from PIL import Image
    a = np.zeros((40, 50), np.uint8); a[10:30, 10:40] = 1
    p = tmp_path / "ph.png"; Image.fromarray(a, "L").save(p)
    m = {"labels": {"0": "water", "1": "liver"}}
    ph = ti.load_phantom(str(p), mapping=m, spacing_mm=0.2)
    assert ph.tissue_names() == ["water", "liver"] and ph.dx == pytest.approx(2e-4)


def test_unmapped_label_raises_unless_loose():
    a = np.zeros((10, 10), np.int32); a[2:5, 2:5] = 7
    m = {"labels": {"0": "water"}}
    with pytest.raises(KeyError):
        ti.load_phantom(a, mapping=m, spacing_mm=0.2)
    ph = ti.load_phantom(a, mapping=m, spacing_mm=0.2, strict=False)
    assert "generic_soft" in ph.tissue_names()


def test_scale_is_mandatory():
    with pytest.raises(ValueError):
        ti.load_phantom(np.zeros((8, 8), np.uint8), mapping={"labels": {"0": "water"}})


def test_custom_tissue_from_mapping():
    a = np.zeros((10, 10), np.uint8); a[3:6, 3:6] = 1
    m = {"labels": {"0": "liver", "1": {"name": "my_lesion", "density": 1075, "speed": 1555,
                                        "atten_db_cm_mhz": 0.9, "backscatter_db": -8}}}
    ph = ti.load_phantom(a, mapping=m, spacing_mm=0.2)
    t = [x for x in ph.tissue_list if x.name == "my_lesion"][0]
    assert t.speed == 1555 and t.backscatter_db == -8


def test_color_and_intensity_modes():
    rgb = np.zeros((20, 20, 3), np.uint8); rgb[5:10, 5:10] = (255, 0, 0)
    ph = ti.load_phantom(rgb, mapping={"colors": {"#000000": "fat", "#ff0000": "liver"}}, spacing_mm=0.2)
    assert set(ph.tissue_names()) == {"fat", "liver"}
    g = np.zeros((20, 20), np.uint8); g[5:10, 5:10] = 200; g[12:15, 12:15] = 100
    ph = ti.load_phantom(g, mode="intensity", mapping={"classes": ["water", "liver", "bone"]}, spacing_mm=0.2)
    assert set(ph.tissue_names()) == {"water", "liver", "bone"}


def test_two_stage_handoff_matches_single_stage(tmp_path):
    ph = ti.make_phantom("thyroid", spacing_mm=0.2)
    cfg = ti.config_for_body_part("thyroid", max_scatterers=3000)
    sc = ti.build_scatterers(ph, cfg)
    save_scatterers(str(tmp_path / "s.npz"), sc, cfg)
    sc2, cfg2 = load_scatterers(str(tmp_path / "s.npz"))
    assert np.array_equal(sc.xyz, sc2.xyz) and cfg2.f0_hz == cfg.f0_hz
    inc, t = make_mock_incident(sc, cfg)
    np.savez(tmp_path / "inc.npz", incident_pressure=inc, t_axis=t)
    inc2, t2 = load_incident(str(tmp_path / "inc.npz"))
    a = compute_reflected_pressure(sc, inc, t, cfg)
    b = compute_reflected_pressure(sc2, inc2, t2, cfg2)
    assert np.allclose(a, b)
    save_reflected(str(tmp_path / "o.npz"), sc2, b, t2, cfg2)
    z = np.load(tmp_path / "o.npz")
    for k in ("t_axis", "reflected_pressure_input", "reflected_pressure_input_row_labels",
              "scatterer_xyz", "scatterer_xyz_columns", "t_axis_units"):
        assert k in z.files
    assert z["reflected_pressure_input"].shape == (sc.n, len(t))


def test_wrong_incident_shape_is_rejected():
    ph = ti.make_phantom("thyroid", spacing_mm=0.2)
    cfg = ti.config_for_body_part("thyroid", max_scatterers=2000)
    sc = ti.build_scatterers(ph, cfg)
    with pytest.raises(ValueError):
        compute_reflected_pressure(sc, np.zeros((sc.n - 1, 100), np.float32), np.arange(100) * 1e-8, cfg)
