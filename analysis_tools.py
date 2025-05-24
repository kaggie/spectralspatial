import torch
import numpy as np
import matplotlib.pyplot as plt

class PulseAnalysis:
    """
    Analysis and visualization tools for spectral-spatial pulses.
    """

    def __init__(self, rf, grad, dt, gamma=4257.0, device='cpu'):
        """
        Args:
            rf (torch.Tensor): Complex RF waveform (Gauss), 1D tensor.
            grad (torch.Tensor): Gradient waveform (G/cm), 1D tensor.
            dt (float): Time step (s).
            gamma (float): Hz/Gyromagnetic ratio (Hz/G).
            device (str): PyTorch device.
        """
        self.rf = rf.to(device)
        self.grad = grad.to(device)
        self.dt = dt
        self.gamma = gamma
        self.device = device

    def plot_rf(self):
        """Plot RF magnitude and phase."""
        t = torch.arange(len(self.rf), device=self.device) * self.dt * 1e3  # ms
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        ax[0].plot(t.cpu(), torch.abs(self.rf).cpu(), label='|RF|')
        ax[0].set_ylabel('Magnitude (Gauss)')
        ax[0].set_title('RF Waveform')
        ax[1].plot(t.cpu(), torch.angle(self.rf).cpu(), label='Phase')
        ax[1].set_ylabel('Phase (rad)')
        ax[1].set_xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

    def plot_gradient(self):
        """Plot gradient waveform."""
        t = torch.arange(len(self.grad), device=self.device) * self.dt * 1e3  # ms
        plt.figure(figsize=(10,3))
        plt.plot(t.cpu(), self.grad.cpu())
        plt.title('Gradient Waveform')
        plt.xlabel('Time (ms)')
        plt.ylabel('Gradient (G/cm)')
        plt.tight_layout()
        plt.show()

    def plot_kspace(self):
        """Plot k-space trajectory (1D)."""
        k = torch.cumsum(self.grad, dim=0) * self.dt * self.gamma / 10.0  # [cycles/cm]
        t = torch.arange(len(k), device=self.device) * self.dt * 1e3  # ms
        plt.figure(figsize=(10,3))
        plt.plot(t.cpu(), k.cpu())
        plt.title('K-space Trajectory (1D)')
        plt.xlabel('Time (ms)')
        plt.ylabel('k (cycles/cm)')
        plt.tight_layout()
        plt.show()
        return k

    def frequency_response(self, N=1024, apply_window=True):
        """
        Compute frequency response of the RF pulse.

        Args:
            N (int): Number of frequency points.
            apply_window (bool): Apply Hamming window (recommended for non-integer cycles).
        Returns:
            freq (torch.Tensor): Frequency axis (Hz).
            resp (torch.Tensor): Frequency response (complex).
        """
        rf = self.rf
        if apply_window:
            window = torch.hamming_window(rf.shape[0], periodic=False, device=self.device)
            rf = rf * window
        resp = torch.fft.fftshift(torch.fft.fft(rf, n=N))
        freq = torch.fft.fftshift(torch.fft.fftfreq(N, d=self.dt))
        return freq.cpu(), resp.cpu()

    def plot_frequency_response(self, N=1024, apply_window=True, db=True):
        """Plot magnitude and phase of frequency response."""
        freq, resp = self.frequency_response(N=N, apply_window=apply_window)
        plt.figure(figsize=(10,4))
        plt.subplot(2,1,1)
        if db:
            plt.plot(freq, 20 * np.log10(np.abs(resp) + 1e-12))
            plt.ylabel('Magnitude (dB)')
        else:
            plt.plot(freq, np.abs(resp))
            plt.ylabel('Magnitude')
        plt.title('RF Frequency Response')
        plt.subplot(2,1,2)
        plt.plot(freq, np.angle(resp))
        plt.ylabel('Phase (rad)')
        plt.xlabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    def spatial_response(self, positions, grad_axis='slice', Npulse=None):
        """
        Simulate spatial response along a 1D axis (e.g., slice axis).
        Args:
            positions (1D array): Spatial positions (cm).
            grad_axis (str): (unused, for extension).
            Npulse (int, optional): Use first Npulse points of rf/grad if set.
        Returns:
            Mxy (np.ndarray): Transverse magnetization at each position.
        """
        # Small-tip approximation (linear): Mxy(x) = gamma*dt*sum(rf(t) * exp(-i*2pi*gamma*int_0^t g(tau)dtau * x))
        rf = self.rf[:Npulse] if Npulse else self.rf
        grad = self.grad[:Npulse] if Npulse else self.grad
        tsteps = rf.shape[0]
        grad_cum = torch.cumsum(grad, dim=0) * self.dt  # G/cm * s
        Mxy = torch.zeros(len(positions), dtype=torch.complex64, device=self.device)
        for ti in range(tsteps):
            phi = 2 * np.pi * self.gamma * grad_cum[ti] * torch.tensor(positions, device=self.device)
            Mxy += rf[ti] * torch.exp(-1j * phi) * self.dt
        return Mxy.cpu().numpy()

    def plot_spatial_profile(self, positions, db=True):
        """Plot the spatial profile (magnitude and phase)."""
        Mxy = self.spatial_response(positions)
        plt.figure(figsize=(10,4))
        plt.subplot(2,1,1)
        if db:
            plt.plot(positions, 20*np.log10(np.abs(Mxy) + 1e-12))
            plt.ylabel('Magnitude (dB)')
        else:
            plt.plot(positions, np.abs(Mxy))
            plt.ylabel('Magnitude')
        plt.title('Spatial Profile (Small-Tip Approximation)')
        plt.subplot(2,1,2)
        plt.plot(positions, np.angle(Mxy))
        plt.ylabel('Phase (rad)')
        plt.xlabel('Position (cm)')
        plt.tight_layout()
        plt.show()

    def analyze_passband_stopband(self, positions, passband, stopband, method='magnitude'):
        """
        Analyze passband, stopband, and transition band properties.

        Args:
            positions (1D array): Spatial positions (cm).
            passband (tuple): (min, max) cm for passband.
            stopband (tuple): (min, max) cm for stopband.
            method (str): 'magnitude'/'db'
        Returns:
            dict with passband_ripple, stopband_ripple, transition_width
        """
        Mxy = self.spatial_response(positions)
        absMxy = np.abs(Mxy)
        dbMxy = 20 * np.log10(absMxy + 1e-12)

        # Passband/stopband mask
        pb_mask = (positions >= passband[0]) & (positions <= passband[1])
        sb_mask = (positions >= stopband[0]) & (positions <= stopband[1])

        if method == 'magnitude':
            pb_ripple = absMxy[pb_mask].max() - absMxy[pb_mask].min()
            sb_ripple = absMxy[sb_mask].max() - absMxy[sb_mask].min()
        else:
            pb_ripple = dbMxy[pb_mask].max() - dbMxy[pb_mask].min()
            sb_ripple = dbMxy[sb_mask].max() - dbMxy[sb_mask].min()

        # Transition width: distance between passband edge where Mxy drops below 0.5
        pb_edge = passband[1]
        sb_edge = stopband[0]
        cross_idx = np.where(absMxy < 0.5 * absMxy[pb_mask].max())[0]
        transition_idx = cross_idx[(positions[cross_idx] > pb_edge) & (positions[cross_idx] < sb_edge)]
        transition_width = positions[transition_idx[0]] - pb_edge if len(transition_idx) > 0 else None

        return {
            'passband_ripple': pb_ripple,
            'stopband_ripple': sb_ripple,
            'transition_width': transition_width
        }
