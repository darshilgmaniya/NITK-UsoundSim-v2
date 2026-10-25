import numpy as np
import pytest
from image_reconstruction import image_reconstruction, ReconConfig


def _rf(f=5e6, c=1540.0, dz=1540 / 80e6):
    z = np.arange(1500) * dz
    t = 2 * z / c
    carrier = np.sin(2 * np.pi * f * t)[:, None] * np.exp(-0.5 * ((z - 15e-3) / 1e-3) ** 2)[:, None]
    return z, np.repeat(carrier, 8, axis=1)


def test_shape_and_axis_preserved():
    z, rf = _rf()
    r = image_reconstruction(rf, z)
    assert r.rf.shape == rf.shape and r.envelope.shape == rf.shape and np.array_equal(r.z_axis, z)


def test_dc_removed():
    z, rf = _rf()
    r = image_reconstruction(rf + 0.3, z, config=ReconConfig(bandpass=False))
    assert np.abs(r.rf.mean(axis=0)).max() < 1e-12


def test_inband_signal_kept_outband_removed():
    z, rf = _rf(f=5e6)
    keep = image_reconstruction(rf, z).rf
    assert np.abs(keep).max() > 0.9 * np.abs(rf).max()
    z2, rf2 = _rf(f=15e6)
    gone = image_reconstruction(rf2, z2).rf
    assert np.abs(gone).max() < 0.1 * np.abs(rf2).max()


def test_disabled_is_identity_and_errors():
    z, rf = _rf()
    assert np.allclose(image_reconstruction(rf, z, config=ReconConfig(enabled=False)).rf, rf)
    with pytest.raises(ValueError):
        image_reconstruction(rf[:, 0], z)
    with pytest.raises(ValueError):
        image_reconstruction(rf, z[:-1])
