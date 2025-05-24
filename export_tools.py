import torch
import numpy as np

class PulseExporter:
    """
    Export designed RF and gradient waveforms to major MRI vendor formats.
    Supported: Siemens .pta, GE .mod, Philips .rf, Pulseq .txt
    """
    def __init__(self, rf, grad, dt, gamma=4257.0):
        """
        Args:
            rf (torch.Tensor or np.ndarray): RF waveform (Gauss), 1D complex.
            grad (torch.Tensor or np.ndarray): Gradient waveform (G/cm), 1D real.
            dt (float): Dwell time (s).
            gamma (float): Gyromagnetic ratio (Hz/G).
        """
        self.rf = rf.detach().cpu().numpy() if isinstance(rf, torch.Tensor) else np.array(rf)
        self.grad = grad.detach().cpu().numpy() if isinstance(grad, torch.Tensor) else np.array(grad)
        self.dt = dt
        self.gamma = gamma

    def export_siemens_pta(self, filename):
        """
        Export to Siemens .pta format (text; units: µT, G/cm, us).
        """
        # Siemens expects RF amplitude in µT, dwell in us, gradient in G/cm
        rf_amps = np.abs(self.rf) * 1e4  # Gauss to µT
        rf_phase = np.angle(self.rf)
        grad = self.grad
        dwell_us = self.dt * 1e6

        with open(filename, 'w') as f:
            f.write("# Siemens .pta spectral-spatial pulse\n")
            f.write("# Columns: RF (uT), Phase (rad), Gradient (G/cm), Dwell(us)\n")
            for a, p, g in zip(rf_amps, rf_phase, grad):
                f.write(f"{a:.6f}\t{p:.6f}\t{g:.6f}\t{dwell_us:.1f}\n")

    def export_ge_mod(self, filename):
        """
        Export to GE .mod format (binary).
        RF in 16-bit ints, scaled to max 32767; gradients in G/cm.
        """
        # GE expects real/imag RF, int16; grad is float32
        rf_real = np.real(self.rf)
        rf_imag = np.imag(self.rf)
        rf_max = np.max(np.abs(np.concatenate([rf_real, rf_imag])))
        rf_scale = 32767.0 / (rf_max + 1e-12)
        rf_real_int = (rf_real * rf_scale).astype(np.int16)
        rf_imag_int = (rf_imag * rf_scale).astype(np.int16)
        grad = self.grad.astype(np.float32)
        n_samp = len(self.rf)

        with open(filename, 'wb') as f:
            # Header: [n_samp (int32), dwell (float32), ...]
            f.write(np.array([n_samp], dtype=np.int32).tobytes())
            f.write(np.array([self.dt], dtype=np.float32).tobytes())
            # RF: real, imag pairs
            for re, im in zip(rf_real_int, rf_imag_int):
                f.write(np.array([re, im], dtype=np.int16).tobytes())
            # Gradients
            grad.tofile(f)

    def export_philips_rf(self, filename):
        """
        Export to Philips .rf format (text; units: µT per time step).
        """
        rf_amps = np.abs(self.rf) * 1e4  # Gauss to µT
        rf_phase = np.angle(self.rf)
        with open(filename, 'w') as f:
            f.write("# Philips .rf file\n")
            f.write(f"# Dwell time: {self.dt * 1e6:.1f} us\n")
            for a, p in zip(rf_amps, rf_phase):
                f.write(f"{a:.6f}\t{p:.6f}\n")

    def export_pulseq_txt(self, filename):
        """
        Export to Pulseq (open format) .txt file: columns for RF (real, imag), grad, dwell.
        """
        rf_real = np.real(self.rf)
        rf_imag = np.imag(self.rf)
        grad = self.grad
        dwell_us = self.dt * 1e6
        with open(filename, 'w') as f:
            f.write("# Pulseq compatible export\n")
            f.write("# Columns: RF_real (Gauss), RF_imag (Gauss), Gradient (G/cm), Dwell (us)\n")
            for re, im, g in zip(rf_real, rf_imag, grad):
                f.write(f"{re:.6e}\t{im:.6e}\t{g:.6e}\t{dwell_us:.1f}\n")
