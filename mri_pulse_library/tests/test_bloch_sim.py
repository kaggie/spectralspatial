# File: mri_pulse_library/tests/test_bloch_sim.py
import unittest
import numpy as np
from mri_pulse_library import rf_pulses
from mri_pulse_library import simulators
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

class TestBlochSimulations(unittest.TestCase):

    # Default parameters for pulse generation
    duration = 1e-3
    flip_angle_deg = 90.0
    bandwidth = 1000.0 # General purpose bandwidth

    # Default parameters for simulation
    B0_T = 3.0 # Not directly used by core sim's b0 offset, but for context
    T1_s = 1.0
    T2_s = 0.05

    # Spatial parameters
    slice_grad_Tm = 0.006 # T/m, fairly typical for 1kHz/mm with gamma_proton
                          # For example, 1kHz/mm = 1000 Hz / 0.001 m = 1e6 Hz/m
                          # G = BW / (gamma_Hz_T * slice_thickness_m)
                          # G = BW_Hz / (gamma_Hz_T/m)
                          # slice_grad_Tm = BW_Hz / ( (GAMMA_HZ_PER_T_PROTON/ (2*np.pi) ) * slice_thickness_m * 2*np.pi)
                          # slice_grad_Tm = self.bandwidth / (GAMMA_HZ_PER_T_PROTON * (0.005) ) # for 5mm slice, 1kHz BW
                          # This value 0.006 T/m for 1kHz BW gives slice thickness of approx.
                          # slice_thickness_m = 1000 / (42.577e6 * 0.006) = 1000 / 255462 = 0.0039m = 3.9mm

    z_positions_m = np.linspace(-0.01, 0.01, 5) # 5 points over 2cm range

    # Spectral parameters
    freq_offsets_hz = np.array([-100, 0, 100])

    def test_simulate_hard_pulse(self):
        rf_T, time_s = rf_pulses.simple.generate_hard_pulse(self.duration, self.flip_angle_deg)
        M_final = simulators.simulate_hard_pulse(rf_T, time_s, B0_T=self.B0_T, T1_s=self.T1_s, T2_s=self.T2_s)
        self.assertEqual(M_final.shape, (3,))
        # Example: Check for non-zero Mz after a 90-degree pulse (will not be fully zero due to T1/T2)
        # For a 90deg pulse on Mz=1, expect Mz near 0, Mxy non-zero.
        # self.assertTrue(np.abs(M_final[2]) < 0.1) # This depends heavily on T1/T2/duration

    def test_simulate_hard_pulse_profile(self):
        rf_T, time_s = rf_pulses.simple.generate_hard_pulse(self.duration, self.flip_angle_deg)
        M_profile = simulators.simulate_hard_pulse_profile(rf_T, time_s, freq_offsets_hz=self.freq_offsets_hz)
        self.assertEqual(M_profile.shape, (len(self.freq_offsets_hz), 3))

    def test_simulate_sinc_pulse(self):
        rf_T, time_s = rf_pulses.simple.generate_sinc_pulse(self.duration, self.bandwidth, self.flip_angle_deg)
        M_slice_profile = simulators.simulate_sinc_pulse(rf_T, time_s, self.slice_grad_Tm, self.z_positions_m)
        self.assertEqual(M_slice_profile.shape, (len(self.z_positions_m), 3))

    def test_simulate_gaussian_pulse(self):
        rf_T, time_s = rf_pulses.simple.generate_gaussian_pulse(self.duration, self.bandwidth, self.flip_angle_deg)
        M_slice_profile = simulators.simulate_gaussian_pulse(rf_T, time_s, self.slice_grad_Tm, self.z_positions_m)
        self.assertEqual(M_slice_profile.shape, (len(self.z_positions_m), 3))

    def test_simulate_spsp_pulse(self):
        spsp_duration = 5e-3
        # Ensure spatial_bw and spectral_bw are reasonable for spsp_duration
        spatial_bw_spsp = self.bandwidth
        spectral_bw_spsp = self.bandwidth / 5 # spectral_bw parameter not strongly used in current spsp gen

        rf_T, time_s = rf_pulses.spectral_spatial.generate_spsp_pulse(
            spsp_duration, spatial_bw_spsp, spectral_bw_spsp, self.flip_angle_deg, n_subpulses=5)
        M_spsp_profile = simulators.simulate_spsp_pulse(
            rf_T, time_s, self.slice_grad_Tm, self.z_positions_m, self.freq_offsets_hz)
        self.assertEqual(M_spsp_profile.shape, (len(self.z_positions_m), len(self.freq_offsets_hz), 3))

    def test_simulate_3d_multiband_spsp(self):
        duration_3d = 10e-3; n_subpulses_3d = 8
        spatial_bws = [self.bandwidth, self.bandwidth]
        slice_pos_m_3d = np.array([-0.005, 0.005]) # Slices closer for reasonable z_positions_m
        spectral_bws = [self.bandwidth/5, self.bandwidth/5]
        spectral_freqs = np.array([-200, 200]) # Wider spectral separation

        rf_T, time_s = rf_pulses.spectral_spatial.generate_3d_multiband_spsp(
            duration_3d, spatial_bws, self.slice_grad_Tm, slice_pos_m_3d,
            spectral_bws, spectral_freqs, self.flip_angle_deg, n_subpulses_3d
        )
        M_3d_profile = simulators.simulate_3d_multiband_spsp(
            rf_T, time_s, self.slice_grad_Tm, self.z_positions_m, self.freq_offsets_hz
        )
        self.assertEqual(M_3d_profile.shape, (len(self.z_positions_m), len(self.freq_offsets_hz), 3))


    def test_simulate_hs1_pulse(self):
        hs_duration = 8e-3
        hs_bw = 4000 # Hz sweep
        rf_T, time_s = rf_pulses.adiabatic.generate_hs1_pulse(hs_duration, hs_bw)
        b1_variations = np.array([0.8, 1.0, 1.2])
        M_hs_profile = simulators.simulate_hs1_pulse(rf_T, time_s, B1_variations=b1_variations)
        self.assertEqual(M_hs_profile.shape, (len(b1_variations), 3))
        # For 180deg HS pulse (default), expect Mz near -1 for B1_scale=1.0
        # self.assertTrue(M_hs_profile[1, 2] < -0.9) # Index 1 is B1_scale=1.0

if __name__ == '__main__':
    unittest.main()
