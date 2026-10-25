"""End-to-end tests of integration 03: phantom in, final image out."""
import os
import numpy as np
import pytest

import _paths  # noqa: F401
from usoundsim_relay import RelayConfig
from receiver_beamforming import RxConfig
from usoundsim_full import demo_points
from usoundsim_pipeline import PostConfig, run_pipeline

ROOT = _paths.ROOT
SAMPLE = os.path.join(ROOT, "teams", "T4_Tissue_Interaction", "sample_data")


@pytest.fixture(scope="module")
def points_result():
    return run_pipeline(demo_points(), RelayConfig(also_ti_channel=False), verbose=False)


def test_shapes_and_ranges(points_result):
    r = points_result
    assert r.bmode.shape == r.full.beamformed_rf.shape
    assert r.final.shape == r.image.shape == (r.z_img.size, r.x_img.size)
    for a in (r.bmode, r.image, r.final):
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1


def test_nine_wires_at_the_right_place(points_result):
    r = points_result
    for z_mm in (10, 20, 30):
        for x_mm in (-6, 0, 6):
            zi = np.argmin(np.abs(r.z_img * 1e3 - z_mm)); xi = np.argmin(np.abs(r.x_img * 1e3 - x_mm))
            win = r.final[max(zi - 12, 0):zi + 13, max(xi - 12, 0):xi + 13]
            j, i = np.unravel_index(win.argmax(), win.shape)
            z_pk = r.z_img[max(zi - 12, 0) + j] * 1e3; x_pk = r.x_img[max(xi - 12, 0) + i] * 1e3
            assert abs(z_pk - z_mm) < 0.5 and abs(x_pk - x_mm) < 0.5, (x_mm, z_mm, x_pk, z_pk)


def test_despeckle_none_equals_scan_converted():
    r = run_pipeline(demo_points(), RelayConfig(also_ti_channel=False), post=PostConfig(despeckle="none"), verbose=False)
    assert np.array_equal(r.final, r.image)


def test_csv_phantom(tmp_path):
    f = tmp_path / "p.csv"; f.write_text("x_mm,z_mm,amp\n0,15,1\n3,25,1\n")
    r = run_pipeline(str(f), RelayConfig(also_ti_channel=False), verbose=False)
    assert r.final.shape[0] > 100 and r.final.max() > 0.9


def test_image_phantom_with_mapping_and_outputs(tmp_path):
    out = str(tmp_path / "o")
    r = run_pipeline(os.path.join(SAMPLE, "phantom_thyroid_labels.png"), RelayConfig(also_ti_channel=False, max_scatterers=3000),
                     mapping=os.path.join(SAMPLE, "phantom_thyroid_mapping.json"), spacing_mm=0.04, outdir=out, verbose=False)
    assert r.final.std() > 0
    for k in ("05_beamformed", "06_recon_rf", "07_bmode", "08_final_image"):
        assert os.path.exists(r.files[k]), k
    z = np.load(r.files["08_final_image"])
    assert np.array_equal(z["final"], r.final)


def test_bad_options_fail_early():
    with pytest.raises(ValueError):
        run_pipeline(demo_points(), post=PostConfig(despeckle="median"), verbose=False)
    with pytest.raises(ValueError):
        run_pipeline(demo_points(), rx=RxConfig(apodization="nope"), verbose=False)
