"""
test_acoustic_propagation.py
=============================

Basic unit tests for the Phase-1 Acoustic Propagation module
(nitk_usoundsim.acoustic_propagation), covering:

  - element positions
  - wavelength and pitch
  - TX distance calculation
  - RX distance calculation
  - TX propagation delay
  - RX propagation delay
  - attenuation calculation
  - output dimensions
  - expected round-trip delay for the 20 mm phantom target
  - deeper targets -> larger delay / more attenuation (monotonicity)
  - off-axis targets -> different RX delays across elements
  - multiple-scatterer linear superposition

Run with:  python -m unittest nitk_usoundsim.test_acoustic_propagation -v
(from the usound_sim folder, so the `nitk_usoundsim` package is importable).
"""

import unittest

import numpy as np

from nitk_usoundsim import config
from nitk_usoundsim import acoustic_propagation as ap


class TestArrayGeometry(unittest.TestCase):
    def test_element_count(self):
        x = ap.create_linear_array()
        self.assertEqual(x.size, config.N_ELEMENTS)

    def test_element_spacing_equals_pitch(self):
        x = ap.create_linear_array()
        spacing = np.diff(x)
        np.testing.assert_allclose(spacing, config.PITCH, rtol=1e-12)

    def test_array_centred_on_zero(self):
        x = ap.create_linear_array()
        self.assertAlmostEqual(float(np.mean(x)), 0.0, places=12)

    def test_wavelength_and_pitch_relationship(self):
        # LAMBDA = C / F0 ; PITCH = LAMBDA / 2  (both come from config.py)
        expected_lambda = config.C / config.F0
        self.assertAlmostEqual(config.LAMBDA, expected_lambda, places=9)
        self.assertAlmostEqual(config.PITCH, expected_lambda / 2.0, places=9)


class TestDistancesAndDelays(unittest.TestCase):
    def setUp(self):
        self.x_elements = ap.create_linear_array()

    def test_tx_distance_matches_manual_formula(self):
        x_s, z_s = 0.003, 0.020
        d = ap.calculate_tx_distance(self.x_elements, x_s, z_s)
        expected = np.sqrt((x_s - self.x_elements) ** 2 + z_s ** 2)
        np.testing.assert_allclose(d, expected, rtol=1e-12)

    def test_rx_distance_matches_manual_formula(self):
        x_s, z_s = -0.004, 0.030
        d = ap.calculate_rx_distance(self.x_elements, x_s, z_s)
        expected = np.sqrt((x_s - self.x_elements) ** 2 + z_s ** 2)
        np.testing.assert_allclose(d, expected, rtol=1e-12)

    def test_tx_distance_on_axis_center_equals_depth(self):
        # For a scatterer directly on axis (x_s=0), the two central
        # elements should be almost exactly `z_s` away.
        d = ap.calculate_tx_distance(self.x_elements, 0.0, 0.020)
        self.assertAlmostEqual(d.min(), 0.020, delta=1e-4)

    def test_tx_propagation_delay(self):
        d = np.array([0.010, 0.020, 0.030])
        t = ap.calculate_propagation_delay(d, c=config.C)
        np.testing.assert_allclose(t, d / config.C, rtol=1e-12)

    def test_rx_propagation_delay(self):
        d = np.array([0.015, 0.025])
        t = ap.calculate_propagation_delay(d, c=config.C)
        np.testing.assert_allclose(t, d / config.C, rtol=1e-12)

    def test_deeper_targets_have_larger_tx_delay(self):
        d_shallow = ap.calculate_tx_distance(self.x_elements, 0.0, 0.010)
        d_deep = ap.calculate_tx_distance(self.x_elements, 0.0, 0.030)
        t_shallow = ap.calculate_propagation_delay(d_shallow, c=config.C)
        t_deep = ap.calculate_propagation_delay(d_deep, c=config.C)
        self.assertTrue(np.all(t_deep > t_shallow))

    def test_offaxis_target_gives_varying_rx_delays_across_elements(self):
        d = ap.calculate_rx_distance(self.x_elements, 0.005, 0.020)
        t = ap.calculate_propagation_delay(d, c=config.C)
        # An off-axis target must not arrive at every element at the same time.
        self.assertGreater(t.max() - t.min(), 0.0)


class TestAttenuation(unittest.TestCase):
    def test_attenuation_matches_manual_db_formula(self):
        distance_m = 0.020
        atten = ap.calculate_attenuation(distance_m, f0=config.F0, alpha0=config.ALPHA_0)
        f_mhz = config.F0 / 1.0e6
        d_cm = distance_m * 100.0
        expected_db = config.ALPHA_0 * f_mhz * d_cm
        expected_linear = 10.0 ** (-expected_db / 20.0)
        self.assertAlmostEqual(atten, expected_linear, places=12)

    def test_attenuation_at_20mm_is_5dB(self):
        # ALPHA_0=0.5 dB/MHz/cm * F0=5MHz * 2cm = 5 dB one-way.
        atten = ap.calculate_attenuation(0.020, f0=config.F0, alpha0=config.ALPHA_0)
        atten_db = -20.0 * np.log10(atten)
        self.assertAlmostEqual(atten_db, 5.0, places=6)

    def test_deeper_targets_experience_more_attenuation(self):
        atten_shallow = ap.calculate_attenuation(0.010)
        atten_deep = ap.calculate_attenuation(0.030)
        # Less remaining amplitude (smaller linear factor) means MORE attenuation.
        self.assertLess(atten_deep, atten_shallow)

    def test_attenuation_is_bounded_between_0_and_1(self):
        atten = ap.calculate_attenuation(np.array([0.0, 0.01, 0.02, 0.05]))
        self.assertTrue(np.all(atten <= 1.0))
        self.assertTrue(np.all(atten > 0.0))


class TestPulse(unittest.TestCase):
    def test_pulse_sampled_at_fs(self):
        t, p = ap.generate_pulse()
        dt = np.diff(t)
        np.testing.assert_allclose(dt, 1.0 / config.FS, rtol=1e-9)
        self.assertEqual(t.shape, p.shape)

    def test_pulse_centered_near_zero(self):
        t, p = ap.generate_pulse()
        self.assertAlmostEqual(t[np.argmax(np.abs(p))], 0.0, delta=2.0 / config.FS)


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = ap.simulate_acoustic_propagation()

    def test_output_dimensions(self):
        pressure = self.result["pressure"]
        self.assertEqual(pressure.shape[0], config.N_ELEMENTS)
        self.assertEqual(pressure.shape[1], self.result["t_axis"].size)
        self.assertGreater(pressure.shape[1], 0)

    def test_output_is_real_valued_mechanical_pressure(self):
        pressure = self.result["pressure"]
        self.assertTrue(np.isrealobj(pressure))
        self.assertTrue(np.all(np.isfinite(pressure)))

    def test_round_trip_delay_for_20mm_phantom_point(self):
        # Nominal on-axis, same-path round trip: 2 * 20e-3 / 1540 ~= 25.97 us
        entry_20mm = next(
            e for e in self.result["per_scatterer"] if abs(e["scatterer"]["z"] - 0.020) < 1e-9
        )
        round_trip_min = entry_20mm["t_tx"].min() + entry_20mm["t_rx"].min()
        expected = 2.0 * 0.020 / config.C
        self.assertAlmostEqual(round_trip_min, expected, delta=0.05e-6)  # 50 ns tolerance

    def test_deeper_phantom_points_have_larger_min_delay(self):
        entries = sorted(self.result["per_scatterer"], key=lambda e: e["scatterer"]["z"])
        min_delays = [e["t_tx"].min() for e in entries]
        self.assertTrue(np.all(np.diff(min_delays) > 0))

    def test_multiple_scatterer_linear_superposition(self):
        total = self.result["pressure"]
        manual_sum = np.zeros_like(total)
        for entry in self.result["per_scatterer"]:
            manual_sum += entry["pressure"]
        np.testing.assert_allclose(total, manual_sum, rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
