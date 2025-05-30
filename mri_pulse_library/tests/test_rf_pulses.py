# File: mri_pulse_library/tests/test_rf_pulses.py
import unittest
import numpy as np
from mri_pulse_library.rf_pulses import simple, spectral_spatial, adiabatic
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

class TestRFPulseGeneration(unittest.TestCase):

    def _check_pulse_outputs(self, rf_pulse, time_vector, min_duration=1e-4):
        self.assertIsInstance(rf_pulse, np.ndarray)
        self.assertIsInstance(time_vector, np.ndarray)
        if min_duration > 0:
            self.assertTrue(rf_pulse.size > 0, "RF pulse array is empty")
            self.assertTrue(time_vector.size > 0, "Time vector is empty")
        self.assertEqual(rf_pulse.shape, time_vector.shape, "RF pulse and time vector shape mismatch")

    def test_generate_hard_pulse(self):
        duration = 1e-3  # 1 ms
        flip_angle_deg = 90.0
        rf, time = simple.generate_hard_pulse(duration, flip_angle_deg)
        self._check_pulse_outputs(rf, time)
        # Check if all elements are approximately equal (allowing for float precision if complex in future)
        if rf.size > 0:
            self.assertTrue(np.allclose(rf, rf[0]), "Hard pulse is not constant")

        # Test zero duration
        rf_zero, time_zero = simple.generate_hard_pulse(0, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)


    def test_generate_sinc_pulse(self):
        duration = 2e-3
        bandwidth = 2000  # Hz
        flip_angle_deg = 90.0
        rf, time = simple.generate_sinc_pulse(duration, bandwidth, flip_angle_deg, n_lobes=3)
        self._check_pulse_outputs(rf, time)
         # Test zero duration
        rf_zero, time_zero = simple.generate_sinc_pulse(0, bandwidth, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)

    def test_generate_gaussian_pulse(self):
        duration = 2e-3
        bandwidth = 1000  # Hz (less critical for Gaussian shape itself, more for context)
        flip_angle_deg = 90.0
        rf, time = simple.generate_gaussian_pulse(duration, bandwidth, flip_angle_deg, sigma_factor=2.5)
        self._check_pulse_outputs(rf, time)
        if len(rf) > 0: # Peak should be at center for symmetric Gaussian
            self.assertEqual(np.argmax(np.abs(rf)), len(rf) // 2, "Gaussian pulse peak not centered")
         # Test zero duration
        rf_zero, time_zero = simple.generate_gaussian_pulse(0, bandwidth, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)

    def test_generate_spsp_pulse(self):
        duration = 10e-3
        spatial_bw = 1000
        spectral_bw = 200 # This param is noted as not directly used for shaping in current SPSP
        flip_angle_deg = 90.0
        n_subpulses = 8
        rf, time = spectral_spatial.generate_spsp_pulse(duration, spatial_bw, spectral_bw, flip_angle_deg, n_subpulses)
        self._check_pulse_outputs(rf, time)
        self.assertFalse(np.iscomplexobj(rf), "Basic SPSP pulse should be real with Gaussian spectral envelope")

    def test_generate_3d_multiband_spsp(self):
        duration = 16e-3
        spatial_bws = [1000, 1000] # Hz, for 2 slices
        slice_grad_tm = 0.01 # T/m
        slice_pos_m = [0.01, -0.01] # m, for 2 slices
        spectral_bws = [100, 100] # Hz, for 2 bands (param not strongly used in current impl)
        spectral_freqs = [-100, 100] # Hz, for 2 bands
        flip_angle_deg = 45.0
        n_subpulses = 16

        rf, time = spectral_spatial.generate_3d_multiband_spsp(
            duration, spatial_bws, slice_grad_tm, slice_pos_m,
            spectral_bws, spectral_freqs, flip_angle_deg, n_subpulses
        )
        self._check_pulse_outputs(rf, time)
        self.assertTrue(np.iscomplexobj(rf), "3D Multiband SPSP pulse should be complex")

    def test_generate_hs1_pulse(self):
        duration = 8e-3
        bandwidth_hz = 5000 # Sweep bandwidth
        # flip_angle_deg = 180 (default)
        mu = 4.9
        beta_rad_s = 800
        rf, time = adiabatic.generate_hs1_pulse(duration, bandwidth_hz, mu=mu, beta_rad_s=beta_rad_s)
        self._check_pulse_outputs(rf, time)
        self.assertTrue(np.iscomplexobj(rf), "HS1 pulse should be complex")

if __name__ == '__main__':
    unittest.main()
