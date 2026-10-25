import numpy as np
import pytest
from post_processing_stage import postprocess, scan_convert, speckle_index, to_uint8


def _speckle(shape=(400, 60), seed=1):
    return np.clip(np.random.default_rng(seed).rayleigh(0.3, shape), 0, 1)


def test_scan_convert_axes_and_square_pixels():
    z = np.linspace(0, 0.04, 400); x = np.linspace(-0.01, 0.01, 60)
    img, xo, zo = scan_convert(_speckle(), x, z)
    assert img.shape == (zo.size, xo.size) and img.shape[0] <= 512
    assert (xo[1] - xo[0]) == pytest.approx(zo[1] - zo[0], rel=0.01)    # square pixels (within 1 %)
    assert zo[0] == pytest.approx(0) and zo[-1] == pytest.approx(0.04, abs=1e-4)


def test_scan_convert_default_uses_whole_z_axis():
    z = np.linspace(0.002, 0.021, 300); x = np.linspace(-0.009, 0.009, 50)
    _, _, zo = scan_convert(_speckle((300, 50)), x, z)
    assert zo[0] == pytest.approx(0.002) and zo[-1] == pytest.approx(0.021)


def test_scan_convert_shape_mismatch():
    with pytest.raises(ValueError):
        scan_convert(np.zeros((10, 5)), np.arange(4), np.arange(10))


def test_guided_filter_reduces_speckle_index():
    img = _speckle((200, 200))
    before = speckle_index(to_uint8(img)); after = speckle_index(to_uint8(postprocess(img)))
    assert after < before
    assert np.array_equal(postprocess(img, method="none"), img)


def test_postprocess_errors():
    with pytest.raises(ValueError):
        postprocess(np.zeros(5))
    with pytest.raises(ValueError):
        postprocess(np.zeros((5, 5)), gamma=0)
    with pytest.raises(ValueError):
        postprocess(np.zeros((5, 5)), method="x")
