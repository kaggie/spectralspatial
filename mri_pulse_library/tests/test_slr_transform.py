import unittest
import torch
import numpy as np
import math # For math.pi
from mri_pulse_library.slr_transform import SLRTransform
from fir_designer import FIRFilterDesigner # Assumes it's at root
from mri_pulse_library.core.bloch_sim import bloch_simulate # For LTA test verification if needed

class TestSLRTransformEnhanced(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.slr = SLRTransform(verbose=False)

        # Common FIR design parameters for B-polynomial
        self.num_taps_b = 61 # Odd for Type I/II linear phase FIR
        self.fir_bands = [0, 0.1, 0.15, 0.5] # Normalized to Nyquist (0.5 = Fs/2)
        self.fir_weights = [1, 10] # Penalize stopband more

        # Common simulation parameters for LTA tests
        self.dt_s_test = 4e-6
        # Gradient needs to match RF pulse length from SLR (num_taps_b)
        # Initialize gradient in tests where RF pulse length is known
        self.gyromagnetic_ratio_hz_g_test = 4257.0


    def test_slr_refocusing_pulse_generation(self):
        # Target Mz = -1 in passband, Mz = 1 in stopband for a 180-degree refocusing pulse
        mag_ripples_se = [0.05, 0.01]  # Ripple on Mz_pass, Mz_stop
        mag_amplitudes_se = [-1.0, 1.0] # Target Mz_pass, Mz_stop
        nominal_flip_se = math.pi      # 180 degrees

        _, b_poly_amps_target_se, adj_flip_se = self.slr.magnetization_to_b_poly_specs(
            mag_ripples_se, mag_amplitudes_se, nominal_flip_se, 'se', device=self.device
        )
        self.assertAlmostEqual(b_poly_amps_target_se[0].item(), 1.0, delta=1e-3)
        self.assertAlmostEqual(b_poly_amps_target_se[1].item(), 0.0, delta=1e-3)
        self.assertAlmostEqual(adj_flip_se, nominal_flip_se, delta=1e-3)

        b_coeffs_se = FIRFilterDesigner.design_parks_mcclellan_real(
            num_taps=self.num_taps_b, bands_normalized=self.fir_bands,
            desired_amplitudes=b_poly_amps_target_se.tolist(), weights=self.fir_weights, device=self.device
        )
        B_omega_check = torch.fft.fft(b_coeffs_se, n=1024)
        max_abs_B_omega = torch.max(torch.abs(B_omega_check))
        if max_abs_B_omega.item() > 1.0 + 1e-6: # Allow small tolerance
             b_coeffs_se /= max_abs_B_omega

        rf_se = self.slr.b_poly_to_rf(b_coeffs_se, pulse_type='se', device=self.device)
        self.assertEqual(rf_se.shape, (self.num_taps_b,))
        self.assertTrue(rf_se.is_complex())
        if len(rf_se) > 0: self.assertGreater(torch.max(torch.abs(rf_se)).item(), 0)

        rf_ex_from_se_b = self.slr.b_poly_to_rf(b_coeffs_se, pulse_type='ex', device=self.device)
        self.assertFalse(torch.allclose(rf_se, rf_ex_from_se_b, atol=1e-5),
                         "RF for 'se' and 'ex' should differ for the same B-poly.")

    def test_slr_inversion_pulse_generation(self):
        mag_ripples_inv = [0.05, 0.01]
        mag_amplitudes_inv = [-1.0, 1.0]
        nominal_flip_inv = math.pi

        _, b_poly_amps_target_inv, adj_flip_inv = self.slr.magnetization_to_b_poly_specs(
            mag_ripples_inv, mag_amplitudes_inv, nominal_flip_inv, 'inv', device=self.device
        )
        self.assertAlmostEqual(b_poly_amps_target_inv[0].item(), 1.0, delta=1e-3)
        self.assertAlmostEqual(b_poly_amps_target_inv[1].item(), 0.0, delta=1e-3)
        self.assertAlmostEqual(adj_flip_inv, nominal_flip_inv, delta=1e-3)

        b_coeffs_inv = FIRFilterDesigner.design_parks_mcclellan_real(
            num_taps=self.num_taps_b, bands_normalized=self.fir_bands,
            desired_amplitudes=b_poly_amps_target_inv.tolist(), weights=self.fir_weights, device=self.device
        )
        B_omega_check_inv = torch.fft.fft(b_coeffs_inv, n=1024)
        max_abs_B_omega_inv = torch.max(torch.abs(B_omega_check_inv))
        if max_abs_B_omega_inv.item() > 1.0 + 1e-6: b_coeffs_inv /= max_abs_B_omega_inv # Allow small tolerance

        rf_inv = self.slr.b_poly_to_rf(b_coeffs_inv, pulse_type='inv', device=self.device)
        self.assertEqual(rf_inv.shape, (self.num_taps_b,))
        self.assertTrue(rf_inv.is_complex())
        if len(rf_inv)>0: self.assertGreater(torch.max(torch.abs(rf_inv)).item(), 0)

        rf_ex_from_inv_b = self.slr.b_poly_to_rf(b_coeffs_inv, pulse_type='ex', device=self.device)
        self.assertTrue(torch.allclose(rf_inv, rf_ex_from_inv_b, atol=1e-6),
                        "RF for 'inv' and 'ex' should be identical for the same B-poly.")

    def test_iterative_lta_rf_scale_basic(self):
        sta_rf_shape, _ = self.slr.design_rf_pulse_from_mag_specs(
            desired_mag_ripples=[0.01, 0.01], desired_mag_amplitudes=[1.0, 0.0],
            nominal_flip_angle_rad=math.pi/6,
            pulse_type='ex', num_taps_b_poly=self.num_taps_b,
            fir_bands_normalized=self.fir_bands, fir_desired_b_gains=[1.0,0.0],
            fir_weights=self.fir_weights, device=self.device
        )
        initial_peak_b1_gauss = 0.05
        if torch.max(torch.abs(sta_rf_shape)).item() > 1e-9 : # Avoid division by zero
             initial_rf_gauss = sta_rf_shape * (initial_peak_b1_gauss / torch.max(torch.abs(sta_rf_shape)))
        else:
             initial_rf_gauss = torch.zeros_like(sta_rf_shape)
             if initial_peak_b1_gauss > 1e-9 : # Only fail if we expected a non-zero pulse
                self.fail("Initial STA RF shape for LTA test is all zeros with non-zero target.")

        target_lta_flip_rad = math.pi / 2

        gradient_for_lta_test = torch.ones(len(initial_rf_gauss), device=self.device, dtype=torch.float32) * 0.01

        lta_rf, achieved_flip = self.slr.iterative_lta_rf_scale(
            initial_rf_gauss=initial_rf_gauss,
            target_flip_angle_rad=target_lta_flip_rad,
            gradient_waveform_gcms=gradient_for_lta_test,
            dt_s=self.dt_s_test,
            gyromagnetic_ratio_hz_g=self.gyromagnetic_ratio_hz_g_test,
            num_iterations=10,
            target_tolerance_rad=0.01,
            device=self.device.type # Pass 'cpu' or 'cuda' string
        )

        self.assertIsInstance(lta_rf, torch.Tensor)
        self.assertEqual(lta_rf.shape, initial_rf_gauss.shape)
        self.assertTrue(lta_rf.is_complex())
        self.assertAlmostEqual(achieved_flip, target_lta_flip_rad, delta=0.02,
                             msg=f"LTA achieved flip {np.rad2deg(achieved_flip):.1f} != target {np.rad2deg(target_lta_flip_rad):.1f} deg")

    def test_lta_rf_scale_clipping(self):
        initial_rf_gauss = torch.ones(self.num_taps_b, dtype=torch.complex64, device=self.device) * 0.01
        target_lta_flip_rad = math.pi
        max_b1_limit_g = 0.05
        gradient_for_lta_test = torch.ones(len(initial_rf_gauss), device=self.device, dtype=torch.float32) * 0.01

        lta_rf_clipped, achieved_flip_clipped = self.slr.iterative_lta_rf_scale(
            initial_rf_gauss=initial_rf_gauss,
            target_flip_angle_rad=target_lta_flip_rad,
            gradient_waveform_gcms=gradient_for_lta_test,
            dt_s=self.dt_s_test,
            max_b1_amplitude_gauss=max_b1_limit_g,
            num_iterations=5, device=self.device.type
        )
        if lta_rf_clipped.numel() > 0: # Check only if tensor is not empty
            self.assertLessEqual(torch.max(torch.abs(lta_rf_clipped)).item(), max_b1_limit_g + 1e-7)
        if max_b1_limit_g > 1e-9 : # Only assert this if clipping is meaningful
            self.assertLess(achieved_flip_clipped, target_lta_flip_rad,
                            "Achieved flip should be less than target if clipping occurs significantly.")

    def test_lta_rf_scale_zero_target_flip(self):
        initial_rf_gauss = torch.ones(self.num_taps_b, dtype=torch.complex64, device=self.device) * 0.01
        target_lta_flip_rad = 0.0
        gradient_for_lta_test = torch.ones(len(initial_rf_gauss), device=self.device, dtype=torch.float32) * 0.01

        lta_rf_zero, achieved_flip_zero = self.slr.iterative_lta_rf_scale(
            initial_rf_gauss=initial_rf_gauss,
            target_flip_angle_rad=target_lta_flip_rad,
            gradient_waveform_gcms=gradient_for_lta_test,
            dt_s=self.dt_s_test,
            num_iterations=5, device=self.device.type
        )
        self.assertTrue(torch.allclose(lta_rf_zero, torch.zeros_like(lta_rf_zero, device=self.device), atol=1e-5),
                        "LTA RF should be near zero if target flip angle is zero.")
        self.assertAlmostEqual(achieved_flip_zero, 0.0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
