# File: mri_pulse_library/tests/test_rf_pulses.py
import unittest
import numpy as np
from mri_pulse_library.rf_pulses import simple, spectral_spatial, adiabatic
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

# Imports for newly added pulse generation functions
from mri_pulse_library.rf_pulses.adiabatic.bir4_pulse import generate_bir4_pulse
from mri_pulse_library.rf_pulses.adiabatic.wurst_pulse import generate_wurst_pulse
from mri_pulse_library.rf_pulses.adiabatic.goia_wurst_pulse import generate_goia_wurst_pulse
from mri_pulse_library.rf_pulses.composite.composite_pulse import generate_composite_pulse_sequence
# Simple pulse generators are used by the enhanced composite pulse function
# Importing them here directly is not strictly necessary if only testing composite_pulse_sequence
# but good for clarity or if individual simple pulses were also tested here.
# from mri_pulse_library.rf_pulses.simple.hard_pulse import generate_hard_pulse
# from mri_pulse_library.rf_pulses.simple.sinc_pulse import generate_sinc_pulse
# from mri_pulse_library.rf_pulses.simple.gaussian_pulse import generate_gaussian_pulse


class TestRFPulseGeneration(unittest.TestCase):

    def _check_pulse_outputs(self, rf_pulse, time_vector, min_duration=1e-4):
        self.assertIsInstance(rf_pulse, np.ndarray)
        self.assertIsInstance(time_vector, np.ndarray)
        if min_duration > 0:
            self.assertTrue(rf_pulse.size > 0, "RF pulse array is empty")
            self.assertTrue(time_vector.size > 0, "Time vector is empty")
        self.assertEqual(rf_pulse.shape, time_vector.shape, "RF pulse and time vector shape mismatch")

    def test_generate_hard_pulse(self):
        duration = 1e-3
        flip_angle_deg = 90.0
        rf, time = simple.generate_hard_pulse(duration, flip_angle_deg)
        self._check_pulse_outputs(rf, time)
        if rf.size > 0:
            self.assertTrue(np.allclose(rf, rf[0]))
        rf_zero, time_zero = simple.generate_hard_pulse(0, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)

    def test_generate_sinc_pulse(self):
        duration = 2e-3
        bandwidth = 2000
        flip_angle_deg = 90.0
        rf, time = simple.generate_sinc_pulse(duration, bandwidth, flip_angle_deg, n_lobes=3)
        self._check_pulse_outputs(rf, time)
        rf_zero, time_zero = simple.generate_sinc_pulse(0, bandwidth, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)

    def test_generate_gaussian_pulse(self):
        duration = 2e-3
        bandwidth = 1000
        flip_angle_deg = 90.0
        rf, time = simple.generate_gaussian_pulse(duration, bandwidth, flip_angle_deg, sigma_factor=2.5)
        self._check_pulse_outputs(rf, time)
        if len(rf) > 0:
            self.assertEqual(np.argmax(np.abs(rf)), len(rf) // 2)
        rf_zero, time_zero = simple.generate_gaussian_pulse(0, bandwidth, flip_angle_deg)
        self._check_pulse_outputs(rf_zero, time_zero, min_duration=0)
        self.assertEqual(rf_zero.size, 0)

    def test_generate_spsp_pulse(self):
        duration = 10e-3
        spatial_bw = 1000
        spectral_bw = 200
        flip_angle_deg = 90.0
        n_subpulses = 8
        rf, time = spectral_spatial.generate_spsp_pulse(duration, spatial_bw, spectral_bw, flip_angle_deg, n_subpulses)
        self._check_pulse_outputs(rf, time)
        self.assertFalse(np.iscomplexobj(rf))

    def test_generate_3d_multiband_spsp(self):
        duration = 16e-3
        spatial_bws = [1000, 1000]
        slice_grad_tm = 0.01
        slice_pos_m = [0.01, -0.01]
        spectral_bws = [100, 100]
        spectral_freqs = [-100, 100]
        flip_angle_deg = 45.0
        n_subpulses = 16
        rf, time = spectral_spatial.generate_3d_multiband_spsp(
            duration, spatial_bws, slice_grad_tm, slice_pos_m,
            spectral_bws, spectral_freqs, flip_angle_deg, n_subpulses
        )
        self._check_pulse_outputs(rf, time)
        self.assertTrue(np.iscomplexobj(rf))

    def test_generate_hs1_pulse(self):
        duration = 8e-3
        bandwidth_hz = 5000
        mu = 4.9
        beta_rad_s = 800
        rf, time = adiabatic.generate_hs1_pulse(duration, bandwidth_hz, mu=mu, beta_rad_s=beta_rad_s)
        self._check_pulse_outputs(rf, time)
        self.assertTrue(np.iscomplexobj(rf))

    # --- BIR-4 Tests ---
    def test_generate_bir4_pulse_detailed(self):
        duration = 0.010
        bandwidth = 4000
        dt = 1e-5
        rf, t = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertIsInstance(t, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        self.assertEqual(len(rf), len(t))
        num_samples_segment = max(1, int(round((duration / 4.0) / dt)))
        expected_total_samples_bir4 = num_samples_segment * 4
        self.assertEqual(len(rf), expected_total_samples_bir4)
        if expected_total_samples_bir4 > 0:
            self.assertGreater(np.max(np.abs(rf)), 0)
            effective_duration = len(rf) * dt
            self.assertAlmostEqual(t[0], -effective_duration/2.0 + dt/2.0, delta=dt/10.0)
            self.assertAlmostEqual(t[-1], effective_duration/2.0 - dt/2.0, delta=dt/10.0)

    def test_bir4_with_peak_b1_provided(self):
        duration = 0.010
        bandwidth = 4000
        dt = 1e-5
        peak_b1_val = 15e-6
        rf, t = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=peak_b1_val)
        self.assertIsInstance(rf, np.ndarray)
        self.assertGreater(len(rf), 0)
        if len(rf) > 0:
            self.assertAlmostEqual(np.max(np.abs(rf)), peak_b1_val, delta=1e-9)

    def test_bir4_parameter_variations(self):
        duration = 0.008
        bandwidth = 3000
        dt = 1e-5
        fixed_peak_b1 = 10e-6
        rf_ref, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, beta_bir4=10.0, mu_bir4=4.9, kappa_deg=70, xi_deg=90, peak_b1_tesla=fixed_peak_b1)
        rf_kappa, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, kappa_deg=60, peak_b1_tesla=fixed_peak_b1)
        self.assertFalse(np.allclose(rf_ref, rf_kappa))
        rf_xi, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, xi_deg=45, peak_b1_tesla=fixed_peak_b1)
        self.assertFalse(np.allclose(rf_ref, rf_xi))
        rf_beta, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, beta_bir4=8.0, peak_b1_tesla=fixed_peak_b1)
        self.assertFalse(np.allclose(rf_ref, rf_beta))
        rf_mu, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, mu_bir4=6.0, peak_b1_tesla=fixed_peak_b1)
        self.assertFalse(np.allclose(rf_ref, rf_mu))
        for rf_pulse in [rf_ref, rf_kappa, rf_xi, rf_beta, rf_mu]:
            self.assertIsInstance(rf_pulse, np.ndarray)
            self.assertTrue(np.iscomplexobj(rf_pulse))
            self.assertGreater(len(rf_pulse), 0)
            self.assertEqual(len(rf_ref), len(rf_pulse))

    def test_bir4_zero_duration_detailed(self):
        rf_bir4, t_bir4 = generate_bir4_pulse(duration=0, bandwidth=1000)
        self.assertEqual(len(rf_bir4), 0)
        self.assertEqual(len(t_bir4), 0)

    def test_bir4_short_duration_multiple_of_four_segments(self):
        dt = 1e-5
        duration = 4 * dt
        rf, t = generate_bir4_pulse(duration=duration, bandwidth=1000, dt=dt)
        num_samples_segment_calc = max(1, int(round((duration / 4.0) / dt)))
        expected_total_samples = num_samples_segment_calc * 4
        self.assertEqual(len(rf), expected_total_samples)
        self.assertEqual(len(t), expected_total_samples)
        self.assertTrue(np.iscomplexobj(rf))
        if len(rf) > 0:
            self.assertGreaterEqual(np.max(np.abs(rf)), 0)

    def test_bir4_non_default_flip_angle_estimation(self):
        duration = 0.010
        bandwidth = 4000
        dt = 1e-5
        rf_90, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, flip_angle_deg=90)
        self.assertIsInstance(rf_90, np.ndarray)
        self.assertGreater(len(rf_90), 0)
        max_b1_90 = np.max(np.abs(rf_90)) if len(rf_90) > 0 else 0
        rf_180_heuristic, _ = generate_bir4_pulse(duration=duration, bandwidth=bandwidth, dt=dt, flip_angle_deg=180)
        max_b1_180_heuristic = np.max(np.abs(rf_180_heuristic)) if len(rf_180_heuristic) > 0 else 0
        if max_b1_90 > 1e-12 and max_b1_180_heuristic > 1e-12 :
             self.assertNotAlmostEqual(max_b1_90, max_b1_180_heuristic, delta=1e-9)

    # --- WURST Tests ---
    def test_generate_wurst_pulse_detailed(self):
        duration = 0.008
        bandwidth = 5000
        dt = 1e-5
        rf, t = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertIsInstance(t, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        self.assertEqual(len(rf), len(t))
        expected_samples = int(round(duration / dt))
        self.assertEqual(len(rf), expected_samples)
        if expected_samples > 0:
            self.assertGreater(np.max(np.abs(rf)), 0)
            self.assertAlmostEqual(t[0], -duration/2.0 + dt/2.0, delta=dt/10.0)
            self.assertAlmostEqual(t[-1], duration/2.0 - dt/2.0, delta=dt/10.0)
            center_idx = len(rf) // 2
            if len(rf) > 1:
                 self.assertAlmostEqual(np.abs(rf[center_idx]), np.max(np.abs(rf)), delta=np.max(np.abs(rf))*0.05)
                 self.assertAlmostEqual(np.abs(rf[0]), 0, delta=np.max(np.abs(rf))*0.1 if np.max(np.abs(rf)) > 1e-9 else 1e-9)
                 self.assertAlmostEqual(np.abs(rf[-1]), 0, delta=np.max(np.abs(rf))*0.1 if np.max(np.abs(rf)) > 1e-9 else 1e-9)

    def test_wurst_with_peak_b1_provided(self):
        duration = 0.008
        bandwidth = 5000
        dt = 1e-5
        peak_b1_val = 12e-6
        rf, t = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=peak_b1_val)
        self.assertIsInstance(rf, np.ndarray)
        self.assertGreater(len(rf), 0)
        if len(rf) > 0:
            self.assertAlmostEqual(np.max(np.abs(rf)), peak_b1_val, delta=1e-9)

    def test_wurst_b1_estimation_adiabatic(self):
        duration = 0.010
        bandwidth = 6000
        dt = 1e-5
        q_factor = 5.0
        gyromagnetic_ratio_hz_t = 42.577e6
        rf, t = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, adiabaticity_factor_Q=q_factor, gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t)
        self.assertGreater(len(rf), 0)
        sweep_rate_rad_s2 = (2 * np.pi * bandwidth) / duration
        expected_b1_max_rad_s = q_factor * np.sqrt(sweep_rate_rad_s2)
        expected_b1_max_tesla = expected_b1_max_rad_s / (gyromagnetic_ratio_hz_t * 2 * np.pi)
        if len(rf) > 0:
            self.assertAlmostEqual(np.max(np.abs(rf)), expected_b1_max_tesla, delta=expected_b1_max_tesla*0.01)

    def test_wurst_b1_estimation_fallback_integral(self):
        duration = 0.010
        bandwidth = 0
        dt = 1e-5
        flip_angle_deg = 90.0
        gyromagnetic_ratio_hz_t = 42.577e6
        rf, t = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, flip_angle_deg=flip_angle_deg, gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t)
        self.assertGreater(len(rf), 0)
        time_vec_calc = (np.arange(max(1, int(round(duration / dt)))) - (max(1, int(round(duration / dt))) - 1) / 2) * dt
        t_prime_calc = (2.0 * time_vec_calc) / duration if duration > 0 else np.zeros_like(time_vec_calc)
        t_prime_calc = np.clip(t_prime_calc, -1.0, 1.0)
        base_am_calc = 1.0 - np.abs(t_prime_calc)**20.0
        base_am_calc = np.maximum(base_am_calc, 0)
        am_envelope_calc = base_am_calc**1.0
        am_envelope_calc[np.isnan(am_envelope_calc)] = 0
        integral_abs_shape_dt = np.sum(am_envelope_calc) * dt
        gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi
        if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) > 1e-20:
            expected_b1_tesla_integral = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
            if len(rf) > 0:
                 self.assertAlmostEqual(np.max(np.abs(rf)), expected_b1_tesla_integral, delta=expected_b1_tesla_integral*0.01)
        else:
            self.assertTrue(np.max(np.abs(rf)) < 1e-6 or np.deg2rad(flip_angle_deg) == 0)

    def test_wurst_parameter_variations(self):
        duration = 0.006
        bandwidth = 4000
        dt = 1e-5
        fixed_peak_b1 = 10e-6
        rf1, _ = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=1)
        rf2, _ = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=10, phase_k=1)
        self.assertFalse(np.allclose(rf1, rf2))
        rf3, _ = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=2)
        self.assertFalse(np.allclose(rf1, rf3))
        rf4, _ = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, adiabaticity_factor_Q=5.0)
        rf5, _ = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, adiabaticity_factor_Q=8.0)
        if len(rf4) > 0 and len(rf5) > 0:
            self.assertNotAlmostEqual(np.max(np.abs(rf4)), np.max(np.abs(rf5)), delta=1e-9)
        for rf_pulse in [rf1, rf2, rf3, rf4, rf5]:
            self.assertIsInstance(rf_pulse, np.ndarray)
            self.assertTrue(np.iscomplexobj(rf_pulse))
            self.assertGreater(len(rf_pulse), 0)
            self.assertEqual(len(rf1), len(rf_pulse))

    def test_wurst_zero_duration_detailed(self):
        rf_w, t_w = generate_wurst_pulse(duration=0, bandwidth=1000)
        self.assertEqual(len(rf_w), 0)
        self.assertEqual(len(t_w), 0)

    def test_wurst_phase_profile(self):
        duration = 0.002
        bandwidth = 10000
        dt = 1e-6
        rf, t = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=10e-6)
        self.assertGreater(len(rf), 0)
        num_samples_calc = len(t)
        time_vector_calc = (np.arange(num_samples_calc) - (num_samples_calc - 1) / 2) * dt
        t_prime_from_time_vec = (2.0 * time_vector_calc) / duration if duration > dt/2 else np.zeros(num_samples_calc)
        t_prime_from_time_vec = np.clip(t_prime_from_time_vec, -1.0, 1.0)
        instantaneous_freq_hz_ideal = (bandwidth / 2.0) * t_prime_from_time_vec
        phase_rad_ideal = np.cumsum(2 * np.pi * instantaneous_freq_hz_ideal * dt)
        if num_samples_calc > 0:
            phase_rad_ideal -= phase_rad_ideal[num_samples_calc // 2]
        actual_phase = np.angle(rf)
        unwrapped_actual_phase = np.unwrap(actual_phase)
        unwrapped_ideal_phase = np.unwrap(phase_rad_ideal)
        if len(unwrapped_actual_phase) > 0 and len(unwrapped_ideal_phase) > 0 :
            unwrapped_actual_phase -= unwrapped_actual_phase[num_samples_calc // 2]
            unwrapped_ideal_phase -= unwrapped_ideal_phase[num_samples_calc // 2]
            max_phase_diff = np.max(np.abs(unwrapped_actual_phase - unwrapped_ideal_phase))
            total_phase_excursion = np.abs(unwrapped_ideal_phase[-1] - unwrapped_ideal_phase[0]) if len(unwrapped_ideal_phase) >1 else 0
            self.assertLess(max_phase_diff, max(0.1, total_phase_excursion * 0.05))

    # --- GOIA WURST Tests ---
    def test_generate_goia_wurst_pulse_detailed(self):
        duration = 0.012
        bandwidth = 3000
        slice_thickness = 0.005
        peak_gradient = 10.0
        dt = 1e-5
        rf, t, g = generate_goia_wurst_pulse(
            duration=duration, bandwidth=bandwidth,
            slice_thickness_m=slice_thickness, peak_gradient_mT_m=peak_gradient, dt=dt
        )
        self.assertIsInstance(rf, np.ndarray)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(g, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        self.assertEqual(len(rf), len(t))
        self.assertEqual(len(rf), len(g))
        expected_samples = int(round(duration / dt))
        self.assertEqual(len(rf), expected_samples)
        if expected_samples > 0:
            self.assertGreater(np.max(np.abs(rf)), 0)
            self.assertAlmostEqual(np.max(np.abs(g)), peak_gradient, delta=1e-9)
            center_idx = len(rf) // 2
            if len(rf) > 1 and np.max(np.abs(rf)) > 1e-9:
                if np.abs(rf[center_idx]) > 1e-9 :
                    self.assertAlmostEqual(np.abs(g[center_idx]) / peak_gradient,
                                           np.abs(rf[center_idx]) / np.max(np.abs(rf)),
                                           delta=0.01)
            self.assertAlmostEqual(t[0], -duration/2.0 + dt/2.0, delta=dt/10.0)
            self.assertAlmostEqual(t[-1], duration/2.0 - dt/2.0, delta=dt/10.0)

    def test_goia_wurst_with_peak_b1_provided(self):
        duration = 0.010
        bandwidth = 3000
        slice_thickness = 0.005
        peak_gradient = 10.0
        dt = 1e-5
        peak_b1_val = 10e-6
        rf, t, g = generate_goia_wurst_pulse(
            duration=duration, bandwidth=bandwidth, slice_thickness_m=slice_thickness,
            peak_gradient_mT_m=peak_gradient, dt=dt, peak_b1_tesla=peak_b1_val
        )
        self.assertIsInstance(rf, np.ndarray)
        self.assertGreater(len(rf), 0)
        if len(rf) > 0:
            self.assertAlmostEqual(np.max(np.abs(rf)), peak_b1_val, delta=1e-9)

    def test_goia_wurst_b1_estimation_adiabatic(self):
        duration = 0.012
        bandwidth = 5000
        slice_thickness = 0.003
        peak_gradient = 8.0
        dt = 1e-5
        q_factor = 6.0
        gyromagnetic_ratio_hz_t = 42.577e6
        rf, t, g = generate_goia_wurst_pulse(
            duration=duration, bandwidth=bandwidth, slice_thickness_m=slice_thickness,
            peak_gradient_mT_m=peak_gradient, dt=dt, adiabaticity_factor_Q=q_factor,
            gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t
        )
        self.assertGreater(len(rf), 0)
        sweep_rate_rad_s2 = (2 * np.pi * bandwidth) / duration
        expected_b1_max_rad_s = q_factor * np.sqrt(sweep_rate_rad_s2)
        expected_b1_max_tesla = expected_b1_max_rad_s / (gyromagnetic_ratio_hz_t * 2 * np.pi)
        if len(rf) > 0:
            self.assertAlmostEqual(np.max(np.abs(rf)), expected_b1_max_tesla, delta=expected_b1_max_tesla*0.01)

    def test_goia_wurst_b1_estimation_fallback_integral(self):
        duration = 0.010
        bandwidth = 0
        slice_thickness = 0.005
        peak_gradient = 10.0
        dt = 1e-5
        flip_angle_deg = 120.0
        gyromagnetic_ratio_hz_t = 42.577e6
        rf, t, g = generate_goia_wurst_pulse(
            duration=duration, bandwidth=bandwidth, slice_thickness_m=slice_thickness,
            peak_gradient_mT_m=peak_gradient, dt=dt, flip_angle_deg=flip_angle_deg,
            gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t
        )
        self.assertGreater(len(rf), 0)
        num_samples_calc = max(1, int(round(duration / dt)))
        time_vec_calc = (np.arange(num_samples_calc) - (num_samples_calc - 1) / 2) * dt
        t_prime_calc = np.clip((2.0 * time_vec_calc) / duration, -1.0, 1.0) if duration > 0 else np.zeros(num_samples_calc)
        power_n_default = 20.0
        phase_k_default = 1.0
        base_am_calc = 1.0 - np.abs(t_prime_calc)**power_n_default
        base_am_calc = np.maximum(base_am_calc, 0)
        am_envelope_normalized_calc = base_am_calc**phase_k_default
        if np.max(am_envelope_normalized_calc) > 1e-9:
            am_envelope_normalized_calc /= np.max(am_envelope_normalized_calc)
        else:
            am_envelope_normalized_calc = np.zeros(num_samples_calc)
        integral_abs_shape_dt = np.sum(am_envelope_normalized_calc) * dt
        gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi
        if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) > 1e-20:
            expected_b1_tesla_integral = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
            if len(rf) > 0:
                self.assertAlmostEqual(np.max(np.abs(rf)), expected_b1_tesla_integral, delta=expected_b1_tesla_integral*0.01)
        else:
            self.assertTrue(np.max(np.abs(rf)) < 1e-6 or np.deg2rad(flip_angle_deg) == 0)

    def test_goia_wurst_parameter_variations(self):
        duration = 0.010
        dt = 1e-5
        fixed_peak_b1 = 12e-6
        rf1, t1, g1 = generate_goia_wurst_pulse(duration=duration, bandwidth=3000, slice_thickness_m=0.005, peak_gradient_mT_m=10, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=1, goia_factor_C=None)
        rf2, _, _ = generate_goia_wurst_pulse(duration=duration, bandwidth=3000, slice_thickness_m=0.003, peak_gradient_mT_m=10, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=1, goia_factor_C=None)
        self.assertFalse(np.allclose(rf1, rf2))
        rf3, _, g3 = generate_goia_wurst_pulse(duration=duration, bandwidth=3000, slice_thickness_m=0.005, peak_gradient_mT_m=5, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=1, goia_factor_C=None)
        self.assertFalse(np.allclose(rf1, rf3))
        self.assertFalse(np.allclose(g1, g3))
        if len(g3)>0: self.assertAlmostEqual(np.max(np.abs(g3)), 5.0, delta=1e-9)
        rf4, _, g4 = generate_goia_wurst_pulse(duration=duration, bandwidth=3000, slice_thickness_m=0.005, peak_gradient_mT_m=10, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=10, phase_k=1, goia_factor_C=None)
        self.assertFalse(np.allclose(rf1, rf4))
        self.assertFalse(np.allclose(g1, g4))
        rf5, _, _ = generate_goia_wurst_pulse(duration=duration, bandwidth=3000, slice_thickness_m=0.005, peak_gradient_mT_m=10, dt=dt, peak_b1_tesla=fixed_peak_b1, power_n=20, phase_k=1, goia_factor_C=1e5 )
        self.assertFalse(np.allclose(rf1, rf5))
        for item in [rf1, rf2, rf3, rf4, rf5, g1, g3, g4]:
            self.assertIsInstance(item, np.ndarray)
            self.assertGreater(len(item), 0)
            self.assertEqual(len(rf1), len(item))

    def test_goia_wurst_zero_duration_detailed(self):
        rf, t, g = generate_goia_wurst_pulse(duration=0, bandwidth=1000, slice_thickness_m=0.005, peak_gradient_mT_m=10)
        self.assertEqual(len(rf), 0)
        self.assertEqual(len(t), 0)
        self.assertEqual(len(g), 0)

    def test_goia_wurst_zero_gradient(self):
        duration = 0.010
        bandwidth = 3000
        slice_thickness = 0.005
        peak_gradient_zero = 0.0
        dt = 1e-5
        rf_goia_zero_g, t_goia, g_goia = generate_goia_wurst_pulse(duration=duration, bandwidth=bandwidth, slice_thickness_m=slice_thickness, peak_gradient_mT_m=peak_gradient_zero, dt=dt, peak_b1_tesla=10e-6)
        self.assertTrue(np.allclose(g_goia, 0))
        rf_wurst, t_wurst = generate_wurst_pulse(duration=duration, bandwidth=bandwidth, dt=dt, peak_b1_tesla=10e-6, power_n=20, phase_k=1 )
        self.assertTrue(np.allclose(np.abs(rf_goia_zero_g), np.abs(rf_wurst)))
        phase_goia_zero_g = np.unwrap(np.angle(rf_goia_zero_g))
        phase_wurst = np.unwrap(np.angle(rf_wurst))
        if len(phase_goia_zero_g) > 0: phase_goia_zero_g -= phase_goia_zero_g[len(phase_goia_zero_g)//2]
        if len(phase_wurst) > 0: phase_wurst -= phase_wurst[len(phase_wurst)//2]
        self.assertTrue(np.allclose(phase_goia_zero_g, phase_wurst, atol=1e-6))

    # --- Composite Pulse Tests ---
    def test_generate_composite_pulse_sequence_enhanced(self):
        dt = 1e-5
        sub_pulse_1_duration = 0.001
        sub_pulse_2_duration = 0.0005
        example_sequence_hard = [
            {'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': sub_pulse_1_duration},
            {'pulse_type': 'hard', 'flip_angle_deg': 180, 'phase_deg': 90, 'duration_s': sub_pulse_2_duration}
        ]
        rf_hard, t_hard = generate_composite_pulse_sequence(example_sequence_hard, dt=dt)
        self.assertIsInstance(rf_hard, np.ndarray)
        self.assertIsInstance(t_hard, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf_hard))
        self.assertEqual(len(rf_hard), len(t_hard))
        expected_total_samples_hard = int(round((sub_pulse_1_duration + sub_pulse_2_duration) / dt))
        self.assertEqual(len(rf_hard), expected_total_samples_hard)
        num_samples_p1 = int(round(sub_pulse_1_duration/dt))
        if num_samples_p1 > 0:
            non_zero_p1 = rf_hard[:num_samples_p1][np.abs(rf_hard[:num_samples_p1]) > 1e-9]
            if len(non_zero_p1) > 0:
                 mean_phase_p1 = np.mean(np.angle(non_zero_p1, deg=True))
                 self.assertAlmostEqual(mean_phase_p1, 0, delta=1)
        if len(rf_hard) > num_samples_p1:
            non_zero_p2 = rf_hard[num_samples_p1:][np.abs(rf_hard[num_samples_p1:]) > 1e-9]
            if len(non_zero_p2) > 0:
                mean_phase_p2 = np.mean(np.angle(non_zero_p2, deg=True))
                self.assertAlmostEqual(mean_phase_p2, 90, delta=1)

    def test_composite_pulse_with_sinc_subpulses(self):
        dt = 1e-5
        sinc_duration = 0.002
        example_sequence_sinc = [
            {'pulse_type': 'sinc', 'flip_angle_deg': 45, 'phase_deg': 30, 'duration_s': sinc_duration, 'time_bw_product': 4},
            {'pulse_type': 'sinc', 'flip_angle_deg': 60, 'phase_deg': 120, 'duration_s': sinc_duration, 'time_bw_product': 2}
        ]
        rf, t = generate_composite_pulse_sequence(example_sequence_sinc, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        expected_samples = int(round((2 * sinc_duration) / dt))
        self.assertEqual(len(rf), expected_samples)
        if len(rf) > 0 :
            self.assertFalse(np.allclose(np.abs(rf[:int(len(rf)/2)]), np.mean(np.abs(rf[:int(len(rf)/2)]))))

    def test_composite_pulse_with_gaussian_subpulses(self):
        dt = 1e-5
        gauss_duration = 0.0015
        example_sequence_gauss = [
            {'pulse_type': 'gaussian', 'flip_angle_deg': 30, 'phase_deg': 0, 'duration_s': gauss_duration, 'time_bw_product': 2.5},
            {'pulse_type': 'gaussian', 'flip_angle_deg': 75, 'phase_deg': 180, 'duration_s': gauss_duration, 'time_bw_product': 3}
        ]
        rf, t = generate_composite_pulse_sequence(example_sequence_gauss, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        expected_samples = int(round((2 * gauss_duration) / dt))
        self.assertEqual(len(rf), expected_samples)

    def test_composite_pulse_with_mixed_subpulses(self):
        dt = 1e-5
        dur1, dur2, dur3 = 0.001, 0.002, 0.0015
        example_sequence_mixed = [
            {'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': dur1},
            {'pulse_type': 'sinc', 'flip_angle_deg': 180, 'phase_deg': 90, 'duration_s': dur2, 'time_bw_product': 4},
            {'pulse_type': 'gaussian', 'flip_angle_deg': 45, 'phase_deg': 180, 'duration_s': dur3, 'time_bw_product': 3}
        ]
        rf, t = generate_composite_pulse_sequence(example_sequence_mixed, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        expected_samples = int(round((dur1 + dur2 + dur3) / dt))
        self.assertEqual(len(rf), expected_samples)

    def test_composite_pulse_with_inter_pulse_delays(self):
        dt = 1e-5
        dur1, delay1, dur2, delay2 = 0.001, 0.0005, 0.001, 0.0002
        example_sequence_delay = [
            {'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': dur1, 'delay_s': delay1},
            {'pulse_type': 'hard', 'flip_angle_deg': 180, 'phase_deg': 90, 'duration_s': dur2, 'delay_s': delay2}
        ]
        rf, t = generate_composite_pulse_sequence(example_sequence_delay, dt=dt)
        self.assertIsInstance(rf, np.ndarray)
        self.assertTrue(np.iscomplexobj(rf))
        expected_total_duration = dur1 + delay1 + dur2 + delay2
        expected_samples = int(round(expected_total_duration / dt))
        self.assertEqual(len(rf), expected_samples)
        samples_dur1 = int(round(dur1 / dt))
        samples_delay1 = int(round(delay1 / dt))
        samples_dur2 = int(round(dur2 / dt))
        if samples_delay1 > 0:
            delay1_segment = rf[samples_dur1 : samples_dur1 + samples_delay1]
            self.assertTrue(np.allclose(delay1_segment, 0))
        if delay2 > 0:
            start_delay2 = samples_dur1 + samples_delay1 + samples_dur2
            delay2_segment = rf[start_delay2 : start_delay2 + int(round(delay2/dt))]
            self.assertTrue(np.allclose(delay2_segment, 0))

    def test_composite_pulse_invalid_type_and_edge_cases(self):
        dt = 1e-5
        example_invalid = [{'pulse_type': 'unknown', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': 0.001}]
        rf_invalid, t_invalid = generate_composite_pulse_sequence(example_invalid, dt=dt)
        self.assertEqual(len(rf_invalid), 0)
        example_zero_dur_pulse = [{'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': 0}]
        rf_zero_dur, t_zero_dur = generate_composite_pulse_sequence(example_zero_dur_pulse, dt=dt)
        self.assertEqual(len(rf_zero_dur), 0)
        dur = 0.001
        example_zero_delay = [{'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': dur, 'delay_s': 0}]
        rf_zero_delay, t_zero_delay = generate_composite_pulse_sequence(example_zero_delay, dt=dt)
        expected_samples_zero_delay = int(round(dur / dt))
        self.assertEqual(len(rf_zero_delay), expected_samples_zero_delay)
        rf_empty_list, t_empty_list = generate_composite_pulse_sequence([], dt=dt)
        self.assertEqual(len(rf_empty_list), 0)
        self.assertEqual(len(t_empty_list), 0)

    # --- General Adiabatic Zero Duration Test ---
    def test_zero_duration_adiabatic_pulses(self):
        # BIR-4, WURST, and GOIA-WURST are now tested by their own detailed zero duration tests.
        # This test is now empty as all specific adiabatic pulses covered have their own detailed tests.
        pass

if __name__ == '__main__':
    unittest.main()
