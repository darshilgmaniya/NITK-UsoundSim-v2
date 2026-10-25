import numpy as np
import pytest
from bmode_formation import bmode_formation


def test_range_and_shape():
    rf = np.random.default_rng(0).normal(size=(300, 20))
    b = bmode_formation(rf, 40)
    assert b.shape == rf.shape and 0.0 <= b.min() and b.max() <= 1.0
    assert b.max() == pytest.approx(1.0)               # frame peak maps to 1


def test_log_compression_value():
    t = np.arange(512)
    rf = np.sin(2 * np.pi * 0.1 * t)[:, None] * np.where(t < 256, 1.0, 0.1)[:, None]   # 20 dB step
    b = bmode_formation(rf, 50)
    assert (b[100, 0] - b[400, 0]) * 50 == pytest.approx(20.0, abs=1.0)


def test_zero_input_and_errors():
    assert not bmode_formation(np.zeros((10, 3))).any()
    with pytest.raises(ValueError):
        bmode_formation(np.ones((10, 3)), 0)
    with pytest.raises(ValueError):
        bmode_formation(np.ones(10))
