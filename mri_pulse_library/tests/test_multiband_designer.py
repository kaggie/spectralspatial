import unittest
import numpy as np
from mri_pulse_library.rf_pulses.multiband.designer import MultibandPulseDesigner
from mri_pulse_library.rf_pulses.simple.hard_pulse import generate_hard_pulse # For creating a base pulse
from mri_pulse_library.rf_pulses.simple.sinc_pulse import generate_sinc_pulse # For spectral test

# Helper to get spectrum (simplified)
def get_spectrum(signal, dt):
    if len(signal) == 0:
        return np.array([]), np.array([])
    n = len(signal)
    # Frequencies are from -Fs/2 to Fs/2 if using fftshift
    # Fs = 1/dt
    freq = np.fft.fftfreq(n, d=dt)
    spectrum = np.fft.fft(signal)
    return np.fft.fftshift(freq), np.fft.fftshift(np.abs(spectrum))

class TestMultibandPulseDesigner(unittest.TestCase):
    def setUp(self):
        self.designer = MultibandPulseDesigner(verbose=False) # Turn off prints for tests
        self.dt = 1e-5 # 10 us
        self.base_duration = 0.001 # 1ms -> 100 samples

        self.base_rf_hard, _, _ = generate_hard_pulse(
            flip_angle_deg=10,
            duration_s=self.base_duration,
            dt_s=self.dt
        )
        self.base_rf_hard = self.base_rf_hard.astype(np.complex128)
        self.base_grad = np.linspace(0, 1, len(self.base_rf_hard)) * 10 # mT/m

    def test_basic_invocation(self):
        num_bands = 2
        offsets = [-1000, 1000] # Hz

        mb_rf, t_vec, mb_grad = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard,
            base_pulse_dt_s=self.dt,
            num_bands=num_bands,
            band_offsets_hz=offsets,
            base_pulse_gradient=self.base_grad
        )
        self.assertIsInstance(mb_rf, np.ndarray)
        self.assertTrue(np.iscomplexobj(mb_rf))
        self.assertEqual(len(mb_rf), len(self.base_rf_hard))
        self.assertEqual(len(t_vec), len(self.base_rf_hard))
        self.assertIsNotNone(mb_grad)
        self.assertTrue(np.array_equal(mb_grad, self.base_grad))
        if len(t_vec) > 1: # Avoid error if t_vec is very short
            self.assertAlmostEqual(t_vec[1] - t_vec[0], self.dt, delta=1e-9)

    def test_spectral_content_two_bands(self):
        base_rf_sinc, _, _ = generate_sinc_pulse(
            flip_angle_deg=10, duration_s=self.base_duration,
            time_bw_product=4, dt_s=self.dt
        )
        base_rf_sinc = base_rf_sinc.astype(np.complex128)

        num_bands = 2
        offsets = [-5000, 5000]

        mb_rf, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=base_rf_sinc,
            base_pulse_dt_s=self.dt,
            num_bands=num_bands,
            band_offsets_hz=offsets
        )

        freqs, spec = get_spectrum(mb_rf, self.dt)
        if len(freqs) == 0: # Handle empty spectrum case
            self.fail("Spectrum calculation failed for multiband RF.")

        idx_offset1 = np.argmin(np.abs(freqs - offsets[0]))
        idx_offset2 = np.argmin(np.abs(freqs - offsets[1]))
        idx_center = np.argmin(np.abs(freqs - 0))

        window_hz = 1000
        samples_in_window = int(window_hz / (freqs[1]-freqs[0] if len(freqs)>1 and (freqs[1]-freqs[0])!=0 else window_hz)) if len(freqs)>1 else 1

        peak1_val = np.max(spec[max(0, idx_offset1-samples_in_window) : min(len(spec), idx_offset1+samples_in_window+1)])
        peak2_val = np.max(spec[max(0, idx_offset2-samples_in_window) : min(len(spec), idx_offset2+samples_in_window+1)])
        center_val_arr = spec[max(0, idx_center-samples_in_window) : min(len(spec), idx_center+samples_in_window+1)]
        center_val = np.max(center_val_arr) if len(center_val_arr) > 0 else 0.0

        self.assertAlmostEqual(peak1_val, peak2_val, delta=peak1_val * 0.2, msg="Peaks for symmetric bands should be similar")
        if abs(offsets[0]) > window_hz * 2:
             self.assertGreater(peak1_val, center_val * 1.5 if center_val > 1e-9 else 1e-9,
                             msg="Peak at offset should be significantly larger than at center for well-separated bands")

    def test_band_phases(self):
        num_bands = 2
        offsets = [-1000, 1000]
        phases1 = [0, 0]
        phases2 = [0, 90]

        mb_rf1, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets, band_phases_deg=phases1
        )
        mb_rf2, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets, band_phases_deg=phases2
        )
        self.assertFalse(np.allclose(mb_rf1, mb_rf2), "Changing band phases should alter the RF waveform")

    def test_peak_b1_scaling(self):
        num_bands = 3
        offsets = [-2000, 0, 2000]

        mb_rf_unscaled, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets
        )
        peak_unscaled = np.max(np.abs(mb_rf_unscaled)) if len(mb_rf_unscaled) > 0 else 0

        limit_high = peak_unscaled * 1.1 if peak_unscaled > 0 else 0.001 # ensure limit_high is non-zero if peak_unscaled is zero
        mb_rf_high_limit, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets, max_b1_tesla_combined=limit_high
        )
        if peak_unscaled > 0: # Only assert if there was something to scale
            self.assertAlmostEqual(np.max(np.abs(mb_rf_high_limit)), peak_unscaled, delta=1e-9)

        limit_low = peak_unscaled * 0.5
        mb_rf_low_limit, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets, max_b1_tesla_combined=limit_low
        )
        if peak_unscaled > 0: # Only assert if there was something to scale
            self.assertAlmostEqual(np.max(np.abs(mb_rf_low_limit)), limit_low, delta=1e-9)
            self.assertTrue(np.allclose(mb_rf_low_limit, mb_rf_unscaled * 0.5, atol=1e-9))

        mb_rf_zero_limit, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=num_bands, band_offsets_hz=offsets, max_b1_tesla_combined=0.0
        )
        self.assertAlmostEqual(np.max(np.abs(mb_rf_zero_limit)) if len(mb_rf_zero_limit) > 0 else 0.0, 0.0, delta=1e-9)

    def test_gradient_passthrough(self):
        mb_rf, t, mb_grad = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=1, band_offsets_hz=[0], base_pulse_gradient=self.base_grad
        )
        self.assertIsNotNone(mb_grad)
        self.assertTrue(np.array_equal(mb_grad, self.base_grad))

        mb_rf, t, mb_grad_none = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=1, band_offsets_hz=[0], base_pulse_gradient=None
        )
        self.assertIsNone(mb_grad_none)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.designer.design_simultaneous_multiband_pulse(self.base_rf_hard, self.dt, 2, [0])
        with self.assertRaises(ValueError):
            self.designer.design_simultaneous_multiband_pulse(self.base_rf_hard, self.dt, 2, [0,0], band_phases_deg=[0])
        with self.assertRaises(ValueError):
            self.designer.design_simultaneous_multiband_pulse(self.base_rf_hard, -0.01, 1, [0])
        with self.assertRaises(ValueError):
            self.designer.design_simultaneous_multiband_pulse(self.base_rf_hard, self.dt, 0, [])
        with self.assertRaises(ValueError):
            self.designer.design_simultaneous_multiband_pulse(np.zeros((2,2), dtype=np.complex128), self.dt, 1, [0])
        with self.assertRaises(ValueError):
            short_grad = self.base_grad[:-10]
            self.designer.design_simultaneous_multiband_pulse(self.base_rf_hard, self.dt, 1, [0], base_pulse_gradient=short_grad)

    def test_edge_cases(self):
        offset = 500
        phase = 30
        mb_rf_single, _, _ = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=self.base_rf_hard, base_pulse_dt_s=self.dt,
            num_bands=1, band_offsets_hz=[offset], band_phases_deg=[phase]
        )
        time_vec = np.arange(len(self.base_rf_hard)) * self.dt
        expected_rf_single = self.base_rf_hard * np.exp(1j * 2 * np.pi * offset * time_vec) * np.exp(1j * np.deg2rad(phase))
        self.assertTrue(np.allclose(mb_rf_single, expected_rf_single))

        empty_rf = np.array([], dtype=np.complex128)
        mb_rf_empty, t_empty, grad_empty = self.designer.design_simultaneous_multiband_pulse(
            base_pulse_rf=empty_rf, base_pulse_dt_s=self.dt,
            num_bands=2, band_offsets_hz=[-100, 100]
        )
        self.assertEqual(len(mb_rf_empty), 0)
        self.assertEqual(len(t_empty), 0)
        self.assertIsNone(grad_empty)

if __name__ == '__main__':
    unittest.main()
