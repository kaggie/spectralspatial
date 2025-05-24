import torch
import numpy as np
import matplotlib.pyplot as plt

class MultiBandMultiSliceDesigner:
    """
    Support for multi-band and multi-slice spectral-spatial pulse design.
    """
    def __init__(self, base_designer):
        """
        Args:
            base_designer: Instance of SpectralSpatialPulseDesigner or similar.
        """
        self.base_designer = base_designer

    def design_multiband_pulse(self, band_centers_cm, *args, **kwargs):
        """
        Design a multi-band pulse by summing shifted single-band pulses.

        Args:
            band_centers_cm (list or array): Centers of each band (cm).
            Other args/kwargs: Passed to design_pulse (e.g., for single-slice).

        Returns:
            dict: Combined pulse with summed rf/grad.
        """
        rf_sum = None
        grad_sum = None

        for center in band_centers_cm:
            # Shift the frequency offset for each band's center
            pulse = self.base_designer.design_pulse(
                f_offset_hz_initial=center * kwargs.get('gamma_hz_g', self.base_designer.gamma_hz_g) * kwargs.get('max_grad_g_cm', self.base_designer.max_grad_g_cm),
                *args, **kwargs
            )
            if rf_sum is None:
                rf_sum = pulse['rf_G']
                grad_sum = pulse['grad_G_cm']
            else:
                rf_sum += pulse['rf_G']
                grad_sum += pulse['grad_G_cm']
        return {
            'rf_G': rf_sum,
            'grad_G_cm': grad_sum
        }

    def design_multislice_pulse(self, slice_centers_cm, *args, **kwargs):
        """
        Design a multi-slice pulse by phase-modulating the RF for each slice.

        Args:
            slice_centers_cm (list or array): Center positions (cm) for each slice.
            Other args/kwargs: Passed to design_pulse.

        Returns:
            dict: Combined pulse.
        """
        N = None
        rf_sum = None
        grad_sum = None
        for center in slice_centers_cm:
            # Add phase modulation for slice selection
            pulse = self.base_designer.design_pulse(
                *args, **kwargs
            )
            if N is None:
                N = len(pulse['rf_G'])
                rf_sum = torch.zeros(N, dtype=torch.complex64, device=pulse['rf_G'].device)
                grad_sum = torch.zeros(N, dtype=grad_sum.dtype or torch.float32, device=pulse['grad_G_cm'].device)
            # Apply spatial phase shift
            phase = torch.exp(1j * 2 * np.pi * self.base_designer.gamma_hz_g * center * torch.arange(N) * self.base_designer.dt)
            rf_sum += pulse['rf_G'] * phase
            grad_sum += pulse['grad_G_cm']
        return {
            'rf_G': rf_sum,
            'grad_G_cm': grad_sum
        }

class VERSEController:
    """
    Variable Rate Selective Excitation (VERSE) tools for advanced re-timing and SAR-aware optimization.
    """
    def __init__(self, dt, max_b1, max_grad, gamma=4257.0):
        self.dt = dt
        self.max_b1 = max_b1
        self.max_grad = max_grad
        self.gamma = gamma

    def apply_verse(self, rf, grad, verse_factor=0.8):
        """
        Basic VERSE re-timing: compress or stretch RF based on gradient amplitude.

        Args:
            rf (torch.Tensor): RF waveform (complex Gauss)
            grad (torch.Tensor): Gradient (G/cm)
            verse_factor (float): Fractional scaling (0 < verse_factor <= 1)

        Returns:
            (rf_verse, grad_verse): VERSEd RF and gradient
        """
        grad_abs = torch.abs(grad)
        max_grad = grad_abs.max()
        scaling = verse_factor * max_grad / (grad_abs + 1e-12)
        # Clamp scaling to avoid infinite dwell
        scaling = torch.clamp(scaling, min=verse_factor, max=10.0)
        rf_verse = rf * scaling
        grad_verse = grad * scaling
        return rf_verse, grad_verse

    def visualize_verse(self, rf, grad, rf_verse, grad_verse):
        t = torch.arange(len(rf)) * self.dt * 1e3
        plt.figure(figsize=(12,6))
        plt.subplot(2,1,1)
        plt.plot(t, torch.abs(rf).cpu(), label='Original |RF|')
        plt.plot(t, torch.abs(rf_verse).cpu(), label='VERSE |RF|', linestyle='--')
        plt.ylabel('B1 (Gauss)')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(t, grad.cpu(), label='Original Grad')
        plt.plot(t, grad_verse.cpu(), label='VERSE Grad', linestyle='--')
        plt.ylabel('G/cm')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.tight_layout()
        plt.show()

class RFSafetyAnalyzer:
    """
    RF Power and SAR calculation tools for safety compliance.
    """
    def __init__(self, rf, dt, coil_resistance=50, body_weight_kg=70, gamma=4257.0):
        """
        Args:
            rf (torch.Tensor): RF waveform (Gauss), complex.
            dt (float): Dwell time (s).
            coil_resistance (float): RF coil resistance (ohms).
            body_weight_kg (float): Patient weight (kg).
        """
        self.rf = rf
        self.dt = dt
        self.coil_resistance = coil_resistance
        self.body_weight_kg = body_weight_kg
        self.gamma = gamma

    def calc_power(self):
        """
        Calculate peak and average RF instantaneous power (arbitrary units or Watts if scaling known).
        """
        b1_rms = torch.sqrt(torch.mean(torch.abs(self.rf) ** 2))
        b1_peak = torch.max(torch.abs(self.rf))
        # Assume power = (B1^2) / R, B1 in Tesla, R in ohm
        # 1 Gauss = 1e-4 Tesla
        power_rms = (b1_rms * 1e-4) ** 2 / self.coil_resistance
        power_peak = (b1_peak * 1e-4) ** 2 / self.coil_resistance
        return {'b1_rms_gauss': b1_rms.item(), 'b1_peak_gauss': b1_peak.item(),
                'power_rms_watt': power_rms.item(), 'power_peak_watt': power_peak.item()}

    def calc_sar(self):
        """
        Estimate Specific Absorption Rate (SAR) [W/kg].
        """
        power = torch.mean(torch.abs(self.rf) ** 2) * 1e-8  # [Gauss^2] * scaling
        # SAR = Power/body_weight (W/kg)
        sar = power / self.body_weight_kg
        return {'sar_w_kg': sar.item()}

    def summary(self):
        power = self.calc_power()
        sar = self.calc_sar()
        print("RF Power & SAR Analysis:")
        print(f"  B1 RMS: {power['b1_rms_gauss']:.4f} G  |  B1 Peak: {power['b1_peak_gauss']:.4f} G")
        print(f"  Power RMS: {power['power_rms_watt']:.6f} W  |  Power Peak: {power['power_peak_watt']:.6f} W")
        print(f"  Estimated SAR: {sar['sar_w_kg']:.6f} W/kg")
