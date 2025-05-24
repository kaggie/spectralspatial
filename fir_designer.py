import torch
import numpy as np
from scipy import signal
import math

class FIRFilterDesigner:
    """
    A class for designing Finite Impulse Response (FIR) filters.
    """

    @staticmethod
    def windowed_sinc(
        length: int, 
        cutoff_normalized: float, 
        window_type: str = 'hamming', 
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Designs an FIR filter using the windowed sinc method.

        Args:
            length: The desired length of the filter (number of taps). 
                    Should be odd for a Type I linear phase filter (symmetric).
            cutoff_normalized: The cutoff frequency, normalized to be between 
                               0 and 0.5 (where 0.5 is the Nyquist frequency).
            window_type: The type of window to use. Can be any window function 
                         supported by `scipy.signal.get_window`. Common examples:
                         'hamming', 'hann', 'blackman', 'boxcar'.
            device: PyTorch device for the output tensor ('cpu' or 'cuda').

        Returns:
            A 1D PyTorch tensor containing the filter coefficients.
        """
        if not (0 < cutoff_normalized <= 0.5):
            raise ValueError("cutoff_normalized must be between 0 (exclusive) and 0.5 (inclusive).")
        if length <= 0:
            raise ValueError("length must be a positive integer.")

        # Create the sinc function
        # For a Type I filter (symmetric, odd length), the center is (length-1)/2
        # n ranges from -(length-1)/2 to (length-1)/2
        if length % 2 == 0:
            # print("Warning: For Type I linear phase (symmetric), filter length should ideally be odd.")
            # For even length, it's Type II (symmetric) or Type IV (anti-symmetric)
            # Sinc function is typically defined around 0.
             n = np.arange(length) - (length - 1) / 2.0
        else:
             n = np.arange(length) - (length - 1) / 2.0
        
        # Handle n=0 case for sinc function (limit is 1)
        h_sinc = np.zeros_like(n, dtype=float)
        non_zero_n_mask = (n != 0)
        zero_n_mask = (n == 0)

        # sinc(x) = sin(pi*x) / (pi*x)
        # Here, x = 2 * cutoff_normalized * n
        # So, h_sinc = sin(2 * pi * cutoff_normalized * n) / (pi * n) -- no, this is not standard for FIR design directly.
        # Standard definition for FIR from sinc:
        # h[n] = 2*fc * sinc(2*fc*n) where sinc(x) = sin(pi*x)/(pi*x)
        # fc is cutoff_normalized.
        # So, h[n] = 2*cutoff_normalized * sin(2*pi*cutoff_normalized*n) / (2*pi*cutoff_normalized*n)
        # h[n] = sin(2*pi*cutoff_normalized*n) / (pi*n)
        
        h_sinc[non_zero_n_mask] = np.sin(2 * np.pi * cutoff_normalized * n[non_zero_n_mask]) / (np.pi * n[non_zero_n_mask])
        h_sinc[zero_n_mask] = 2 * cutoff_normalized # Value of 2*fc*sinc(0) = 2*fc
        
        # Apply the window
        try:
            win = signal.get_window(window_type, length)
        except ValueError as e:
            raise ValueError(f"Unknown window_type: {window_type}. Error: {e}")
            
        h_windowed = h_sinc * win
        
        # Normalize the filter coefficients (optional, but common for lowpass)
        # Sum of taps to 1 for DC gain of 1.
        h_normalized = h_windowed / np.sum(h_windowed)
        
        return torch.tensor(h_normalized, dtype=torch.float32, device=device)

    @staticmethod
    def gaussian(
        length: int, 
        std_normalized: float, 
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Designs an FIR filter using a Gaussian window.
        The filter coefficients are samples of a Gaussian function.

        Args:
            length: The desired length of the filter (number of taps).
            std_normalized: The standard deviation of the Gaussian, normalized 
                            to the filter length (e.g., 0.1 means std is 0.1 * length).
                            A common definition for scipy.signal.windows.gaussian is
                            std relative to (M-1)/2. Here, let's use std in sample units.
                            If std_normalized refers to (M-1)/2, then std = std_normalized * (length-1)/2.
                            Let's assume std is provided in sample units directly for clarity.
                            Or, if std_normalized is a fraction of length, std = std_normalized * length.
                            The prompt says "relative to the length", let's use `std = std_normalized * (length-1)/2`.
            device: PyTorch device for the output tensor ('cpu' or 'cuda').

        Returns:
            A 1D PyTorch tensor containing the filter coefficients.
        """
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        if std_normalized <= 0:
            raise ValueError("std_normalized must be positive.")

        # Scipy's gaussian window takes std in number of samples.
        # M is length. The window is defined for n from -(M-1)/2 to (M-1)/2.
        # std = sigma.
        # Let's use the scipy definition: std is sigma.
        # If std_normalized is relative to (length-1)/2:
        std_samples = std_normalized * (length - 1) / 2.0
        if std_samples == 0 and length > 1 : # Avoid zero std if length > 1, causes issues
            # print("Warning: std_samples is zero for Gaussian window. Using a small epsilon.")
            std_samples = 1e-3 * (length-1)/2.0 # A very small std

        win_coeffs = signal.windows.gaussian(M=length, std=std_samples)
        
        # Normalize the filter coefficients (sum to 1 for DC gain)
        h_normalized = win_coeffs / np.sum(win_coeffs)
        
        return torch.tensor(h_normalized, dtype=torch.float32, device=device)

    def design_least_squares(
        self, 
        num_taps: int, 
        bands_normalized: list[float], 
        desired_amplitudes: list[float], 
        weights: list[float] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Designs an FIR filter using the least-squares method (scipy.signal.firls).

        Args:
            num_taps: The number of filter coefficients (filter length).
            bands_normalized: A list of frequency band edges, normalized to Nyquist 
                              (0 to 1.0, where 1.0 is Nyquist frequency). 
                              Must be non-decreasing and start with 0, end with 1.
                              E.g., [0, 0.2, 0.3, 0.5, 0.6, 1.0] for pass, transition, stop.
            desired_amplitudes: A list of desired amplitudes at the band edges. 
                                Must have the same number of elements as bands_normalized.
                                Typically, pairs of points define constant regions.
                                E.g., [1, 1, 0, 0, 0.5, 0.5] for pass, stop, pass.
            weights: (Optional) A list of weights for each band. If None, defaults to 1.
            device: PyTorch device for the output tensor ('cpu' or 'cuda').

        Returns:
            A 1D PyTorch tensor containing the filter coefficients.
        """
        if not bands_normalized or bands_normalized[0] != 0.0 or bands_normalized[-1] != 1.0:
            raise ValueError("bands_normalized must start with 0.0 and end with 1.0.")
        if len(bands_normalized) != len(desired_amplitudes):
            raise ValueError("bands_normalized and desired_amplitudes must have the same length.")
        if weights is not None and len(weights) * 2 != len(bands_normalized):
             # SciPy firls expects weights to be half the size of bands if specified
            raise ValueError("Length of weights must be half the length of bands_normalized.")


        coeffs_np = signal.firls(
            numtaps=num_taps,
            bands=bands_normalized,
            desired=desired_amplitudes,
            weight=weights,
            # fs=2.0 # Assuming Nyquist is 1.0, so sampling frequency is 2.0
        )
        return torch.tensor(coeffs_np, dtype=torch.float32, device=device)

    def design_parks_mcclellan_real(
        self, 
        num_taps: int, 
        bands_normalized: list[float], 
        desired_amplitudes: list[float], # Should be desired gains for each band, not edges.
        weights: list[float] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Designs a real, linear-phase FIR filter using the Parks-McClellan algorithm 
        (scipy.signal.remez).

        Args:
            num_taps: The number of filter coefficients (filter length).
            bands_normalized: A list of frequency band edges, normalized to Nyquist
                              (0 to 1.0, where 1.0 is Nyquist frequency).
                              Must be non-decreasing and start with 0, end with 1.
                              E.g., [0, 0.2, 0.3, 1.0] for passband 0-0.2, stopband 0.3-1.0.
            desired_amplitudes: A list of desired gain values for each band. 
                                Length must be half the length of bands_normalized.
                                E.g., [1, 0] for a lowpass filter.
            weights: (Optional) A list of weights for each band, same length as desired_amplitudes.
                     If None, defaults to 1.
            device: PyTorch device for the output tensor ('cpu' or 'cuda').

        Returns:
            A 1D PyTorch tensor containing the filter coefficients.
        """
        if not bands_normalized or bands_normalized[0] != 0.0 or bands_normalized[-1] != 1.0:
            raise ValueError("bands_normalized must start with 0.0 and end with 1.0.")
        if len(bands_normalized) % 2 != 0:
            raise ValueError("bands_normalized must have an even number of elements (pairs of band edges).")
        if len(desired_amplitudes) != len(bands_normalized) / 2:
            raise ValueError("desired_amplitudes must have half the length of bands_normalized.")
        if weights is not None and len(weights) != len(desired_amplitudes):
            raise ValueError("weights must have the same length as desired_amplitudes.")

        coeffs_np = signal.remez(
            numtaps=num_taps,
            bands=bands_normalized,
            desired=desired_amplitudes,
            weight=weights,
            # fs=2.0 # Assuming Nyquist is 1.0, sampling frequency is 2.0
            type='bandpass' # Default type for remez if desired has multiple entries.
                            # Could also be 'differentiator' or 'hilbert'.
                            # 'bandpass' implies filter type based on bands and desired.
        )
        return torch.tensor(coeffs_np, dtype=torch.float32, device=device)