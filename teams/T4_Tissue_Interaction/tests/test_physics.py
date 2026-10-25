import numpy as np
import pytest

import tissue_interaction as ti
from tissue_interaction.interaction import compute_reflected_pressure, make_mock_incident
from tissue_interaction.scatterers import KIND_INTERFACE, build_scatterers


def _two_layer(top="water", bottom="bone", z_if_mm=10, depth_mm=20, spacing=0.05):
    b = ti.PhantomBuilder(10, depth_mm, spacing, top, "two_layer")
    b.layer(z_if_mm, depth_mm, bottom)
    return b.build()


def _cfg(**kw):
    cfg = ti.TissueInteractionConfig(f0_hz=5e6, scatterers_per_cell=4, **kw)
    return cfg


def test_normal_incidence_reflection_matches_fresnel():
    ph = _two_layer("water", "bone")
    cfg = _cfg(specular_floor=1.0)            # D = 1 -> isolate the Fresnel term
    sc = build_scatterers(ph, cfg)
    itf = sc.kind == KIND_INTERFACE
    assert itf.sum() > 5
    Z1, Z2 = ti.get_tissue("water").impedance, ti.get_tissue("bone").impedance
    R = (Z2 - Z1) / (Z2 + Z1)
    lam = cfg.wavelength
    expected = R * (lam * 0.5) / (lam * cfg.f_number)
    assert np.allclose(sc.amp[itf], expected, rtol=1e-3)


def test_soft_to_gas_polarity_is_inverted():
    ph = _two_layer("liver", "air")
    sc = build_scatterers(ph, _cfg())
    itf = sc.kind == KIND_INTERFACE
    assert np.all(sc.amp[itf] < 0)


def test_bone_casts_acoustic_shadow():
    ph = _two_layer("liver", "bone", z_if_mm=8, depth_mm=25)
    sc = build_scatterers(ph, _cfg())
    below = sc.xyz[:, 2] > 12e-3
    above = (sc.xyz[:, 2] < 7e-3)
    assert below.any() and above.any()
    assert sc.path_tx_loss[below].max() < 0.7
    assert sc.path_tx_loss[above].min() > 0.99
    assert sc.path_atten_np[below].mean() > sc.path_atten_np[above].mean()


def test_tof_uses_local_speed_of_sound():
    ph = _two_layer("fat", "bone", z_if_mm=10, depth_mm=20)
    sc = build_scatterers(ph, _cfg())
    s = np.argmin(np.abs(sc.xyz[:, 2] - 18e-3) + (sc.kind == KIND_INTERFACE) * 1.0)
    expected = 10e-3 / 1450 + 8e-3 / 3500
    assert abs(sc.tof[s] - expected) / expected < 0.03


def test_oblique_boundary_is_weaker_than_perpendicular():
    b = ti.PhantomBuilder(30, 30, 0.1, "liver", "disc")
    b.ellipse(0, 15, 8, 8, "water")
    sc = build_scatterers(b.build(), _cfg())
    itf = sc.kind == KIND_INTERFACE
    ang = sc.incid_angle_deg[itf]
    a = np.abs(sc.amp[itf])
    mid = (ang > 20) & (ang < 45)
    assert mid.any() and a[ang < 5].mean() > 4 * a[mid].mean()   # specular directivity


def test_anechoic_cyst_has_no_diffuse_scatterers_inside():
    b = ti.PhantomBuilder(30, 30, 0.1, "liver", "cyst")
    b.ellipse(0, 15, 6, 6, "cyst_fluid")
    ph = b.build()
    sc = build_scatterers(ph, _cfg())
    dif = sc.kind == 0
    x, z = sc.xyz[dif, 0] * 1e3, sc.xyz[dif, 2] * 1e3
    inside = (x / 5.5) ** 2 + ((z - 15) / 5.5) ** 2 < 1
    assert inside.sum() == 0


def test_output_contract_shape_and_finite():
    ph = ti.make_phantom("thyroid", spacing_mm=0.2)
    cfg = ti.config_for_body_part("thyroid", max_scatterers=5000)
    sc = build_scatterers(ph, cfg)
    inc, t = make_mock_incident(sc, cfg)
    out = compute_reflected_pressure(sc, inc, t, cfg)
    assert out.shape == (sc.n, len(t)) and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_path_model_none_has_more_deep_energy():
    ph = ti.make_phantom("msk", spacing_mm=0.2)
    base = ti.config_for_body_part("msk", max_scatterers=5000)
    sc = build_scatterers(ph, base)
    inc, t = make_mock_incident(sc, base)
    full = compute_reflected_pressure(sc, inc, t, base)
    base.path_model = "none"
    none = compute_reflected_pressure(sc, inc, t, base)
    deep = sc.xyz[:, 2] > 30e-3
    assert (none[deep] ** 2).sum() > (full[deep] ** 2).sum()


def test_deterministic_with_seed():
    ph = ti.make_phantom("thyroid", spacing_mm=0.2)
    cfg = ti.config_for_body_part("thyroid", max_scatterers=3000, seed=7)
    a, b = build_scatterers(ph, cfg), build_scatterers(ph, cfg)
    assert np.array_equal(a.xyz, b.xyz) and np.array_equal(a.amp, b.amp)


def test_contrast_agent_brighter_than_blood():
    assert ti.get_tissue("contrast_agent").backscatter_db > ti.get_tissue("blood").backscatter_db + 20


@pytest.mark.parametrize("bp", sorted(ti.GENERATORS))
def test_every_body_part_runs(bp):
    ph = ti.make_phantom(bp, spacing_mm=max(0.3, ti.make_phantom(bp).dx * 1e3))
    cfg = ti.config_for_body_part(bp, max_scatterers=4000)
    sc = build_scatterers(ph, cfg)
    assert sc.n > 100 and np.isfinite(sc.amp).all()
