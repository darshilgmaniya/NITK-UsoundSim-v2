import os, sys, json
import numpy as np
import pytest
from PIL import Image

from usoundsim_pipeline.upload import (resolve_inputs, save_uploaded, collect_upload, sample_phantom, _Done)


@pytest.fixture
def phantom(tmp_path):
    img = np.zeros((40, 60), np.uint8); img[10:20, 10:20] = 1
    p = tmp_path / "ph.png"; Image.fromarray(img).save(p)
    m = tmp_path / "ph_mapping.json"; m.write_text(json.dumps({"spacing_mm": 0.2, "labels": {"0": "liver", "1": "cyst_fluid"}}))
    return p, m


def test_image_with_mapping(phantom):
    r = resolve_inputs([phantom[0], phantom[1]])
    assert r.is_image and r.mapping.endswith("ph_mapping.json") and r.source.endswith("ph.png")


def test_image_finds_sibling_mapping(phantom):
    assert resolve_inputs([phantom[0]]).mapping.endswith("ph_mapping.json")


def test_image_without_mapping_is_an_error(tmp_path):
    p = tmp_path / "x.png"; Image.fromarray(np.zeros((5, 5), np.uint8)).save(p)
    with pytest.raises(ValueError, match="mapping"):
        resolve_inputs([p])


def test_csv_and_npy(tmp_path):
    c = tmp_path / "pts.csv"; c.write_text("x_mm,z_mm,amp\n0,10,1\n")
    assert not resolve_inputs([c]).is_image
    n = tmp_path / "pts.npy"; np.save(n, np.array([[0, 10, 1.0], [2, 20, 0.5]]))
    r = resolve_inputs([n])
    assert r.source.endswith("_points.csv") and np.loadtxt(r.source, delimiter=",", skiprows=1).shape == (2, 3)
    bad = tmp_path / "bad.npy"; np.save(bad, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="shape"):
        resolve_inputs([bad])


def test_errors(tmp_path):
    with pytest.raises(ValueError):
        resolve_inputs([])
    with pytest.raises(FileNotFoundError):
        resolve_inputs([tmp_path / "nope.png"])
    t = tmp_path / "a.txt"; t.write_text("x")
    z = tmp_path / "b.csv"; z.write_text("x_mm,z_mm,amp\n0,1,1\n")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_inputs([t, z])


def test_widget_value_formats(tmp_path, phantom):
    ph, mp = phantom
    v8 = ({"name": "ph.png", "content": memoryview(ph.read_bytes())}, {"name": "ph_mapping.json", "content": mp.read_bytes()})
    v7 = {"ph.png": {"metadata": {}, "content": ph.read_bytes()}, "ph_mapping.json": {"metadata": {}, "content": mp.read_bytes()}}
    colab = {"ph.png": ph.read_bytes(), "ph_mapping.json": mp.read_bytes()}
    for i, v in enumerate((v8, v7, colab)):
        d = tmp_path / f"up{i}"
        r = collect_upload(_Done(v), str(d))
        assert r.is_image and os.path.isfile(r.source) and os.path.isfile(r.mapping) and str(d) in r.source
    assert collect_upload(_Done({}), str(tmp_path / "none")) is None and collect_upload(None) is None


def test_sample_phantoms_exist():
    for n in ("sample_abdomen", "sample_thyroid"):
        r = sample_phantom(n)
        assert r.is_image and os.path.isfile(r.mapping)


def test_cli_with_uploaded_files(tmp_path, monkeypatch, phantom):
    """The same path the file dialog takes: files -> resolve_inputs -> run."""
    import run_pipeline as rp
    monkeypatch.setattr(rp, "pick_files_dialog", lambda *a, **k: [str(phantom[0]), str(phantom[1])])
    res = rp.main(["--upload", "--max-scatterers", "1500", "--out", str(tmp_path / "out")])
    assert os.path.isfile(tmp_path / "out" / "final_image.png") and res.final.ndim == 2
