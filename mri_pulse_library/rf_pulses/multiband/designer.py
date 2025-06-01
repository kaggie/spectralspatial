import numpy as np
# from mri_pulse_library.core import constants # Not strictly needed for this implementation yet

class MultibandPulseDesigner:
    """
    Designs simultaneous multiband RF pulses by combining a baseband RF pulse
    with frequency offsets and phase modulation for each band.
    """
    def __init__(self, verbose: bool = True):
        """
        Initializes the MultibandPulseDesigner.

        Args:
            verbose (bool, optional): If True, prints design information and warnings.
                                      Defaults to True.
        """
        self.verbose = verbose

    def design_simultaneous_multiband_pulse(
            self,
            base_pulse_rf: np.ndarray,
            base_pulse_dt_s: float,
            num_bands: int,
            band_offsets_hz: list,
            base_pulse_gradient: np.ndarray = None,
            band_phases_deg: list = None,
            max_b1_tesla_combined: float = None
        ):
        """
        Generates a simultaneous multiband RF pulse using phase/frequency modulation
        of a baseband pulse.

        Args:
            base_pulse_rf (np.ndarray): Complex array of the baseband RF pulse waveform (Tesla).
                                        Assumed to be appropriately scaled for its target flip angle.
            base_pulse_dt_s (float): Time step (sampling interval) of the baseband pulse in seconds.
            num_bands (int): The number of simultaneous bands to generate.
            band_offsets_hz (list): A list of frequency offsets in Hz for each band,
                                    relative to the carrier of the base_pulse_rf.
                                    Length must equal num_bands.
            base_pulse_gradient (np.ndarray, optional): Gradient waveform associated with the
                                                       base_pulse_rf (e.g., mT/m). If provided,
                                                       it's typically returned as the gradient for
                                                       the multiband pulse. Defaults to None.
            band_phases_deg (list, optional): A list of phases in degrees to apply to each band.
                                              If None, all phases are set to 0. Length must
                                              equal num_bands if provided. Defaults to None.
            max_b1_tesla_combined (float, optional): If provided, the peak B1 amplitude of the
                                                     combined multiband pulse will be scaled down
                                                     to not exceed this limit. If None, no scaling
                                                     for peak B1 is performed by this function directly.
                                                     Defaults to None.

        Returns:
            tuple: (multiband_rf_pulse_tesla, time_vector_s, multiband_gradient)
                multiband_rf_pulse_tesla (np.ndarray): The combined complex multiband RF waveform (Tesla).
                time_vector_s (np.ndarray): Time vector for the multiband pulse (seconds).
                multiband_gradient (np.ndarray or None): The gradient waveform (e.g., mT/m).
        """

        # Input validation
        if not isinstance(base_pulse_rf, np.ndarray) or not np.iscomplexobj(base_pulse_rf):
            raise ValueError("base_pulse_rf must be a complex numpy array.")
        if base_pulse_rf.ndim != 1:
            raise ValueError("base_pulse_rf must be a 1D array.")
        if base_pulse_dt_s <= 0:
            raise ValueError("base_pulse_dt_s must be positive.")
        if not isinstance(num_bands, int) or num_bands <= 0:
            raise ValueError("num_bands must be a positive integer.")
        if not isinstance(band_offsets_hz, list) or len(band_offsets_hz) != num_bands:
            raise ValueError("band_offsets_hz must be a list of length num_bands.")
        if band_phases_deg is not None and (not isinstance(band_phases_deg, list) or len(band_phases_deg) != num_bands):
            raise ValueError("band_phases_deg must be a list of length num_bands if provided.")
        if base_pulse_gradient is not None:
            if not isinstance(base_pulse_gradient, np.ndarray):
                raise ValueError("base_pulse_gradient must be a numpy array if provided.")
            # Allowing gradient to be multi-dimensional, but length check is against base_pulse_rf's length (time dimension)
            if base_pulse_gradient.shape[-1] != len(base_pulse_rf): # Assuming time is the last dimension for gradient
                raise ValueError("Time dimension of base pulse RF and gradient must have the same length.")
        if max_b1_tesla_combined is not None and max_b1_tesla_combined < 0:
            raise ValueError("max_b1_tesla_combined must be non-negative if provided.")

        if self.verbose:
            self._log(f"Starting multiband pulse design for {num_bands} bands.")
            self._log(f"Base pulse: {len(base_pulse_rf)} samples, dt={base_pulse_dt_s*1e6:.2f} us, peak B1={np.max(np.abs(base_pulse_rf))*1e6:.2f} uT.")

        num_samples = len(base_pulse_rf)
        if num_samples == 0:
            self._log("Base pulse is empty, returning empty multiband pulse.")
            return np.array([], dtype=np.complex128), np.array([]), np.array([]) if base_pulse_gradient is None else np.array([])


        time_vector_s = np.arange(num_samples) * base_pulse_dt_s
        multiband_rf_pulse_tesla = np.zeros_like(base_pulse_rf, dtype=np.complex128)

        actual_band_phases_deg = band_phases_deg if band_phases_deg is not None else [0.0] * num_bands

        for i in range(num_bands):
            offset_hz = band_offsets_hz[i]
            phase_deg = actual_band_phases_deg[i]

            freq_phasor = np.exp(1j * 2 * np.pi * offset_hz * time_vector_s)
            band_phasor = np.exp(1j * np.deg2rad(phase_deg))

            current_band_rf = base_pulse_rf * freq_phasor * band_phasor
            multiband_rf_pulse_tesla += current_band_rf

            if self.verbose and num_bands < 10:
                self._log(f"  Band {i+1}/{num_bands}: offset={offset_hz:.1f} Hz, phase={phase_deg:.1f} deg. Max |RF_band|={np.max(np.abs(current_band_rf))*1e6:.2f} uT")

        current_peak_b1_combined = np.max(np.abs(multiband_rf_pulse_tesla)) if num_samples > 0 else 0.0
        if self.verbose:
            self._log(f"Initial combined peak B1: {current_peak_b1_combined*1e6:.2f} uT.")

        if max_b1_tesla_combined is not None and current_peak_b1_combined > max_b1_tesla_combined:
            if max_b1_tesla_combined == 0 :
                scaling_factor = 0.0
            elif current_peak_b1_combined > 1e-12:
                scaling_factor = max_b1_tesla_combined / current_peak_b1_combined
            else:
                scaling_factor = 1.0

            multiband_rf_pulse_tesla *= scaling_factor
            if self.verbose:
                final_peak_b1 = np.max(np.abs(multiband_rf_pulse_tesla)) if num_samples > 0 else 0.0
                self._log(f"Combined pulse peak B1 exceeded target ({max_b1_tesla_combined*1e6:.2f} uT). Scaled by {scaling_factor:.3f}. Final peak B1: {final_peak_b1*1e6:.2f} uT.")
                if scaling_factor != 1.0: # Only print warning if actual scaling happened
                    self._log("Warning: This scaling affects flip angles of all bands proportionally.")

        multiband_gradient = np.copy(base_pulse_gradient) if base_pulse_gradient is not None else None

        if self.verbose:
            self._log("Multiband pulse design complete.")

        return multiband_rf_pulse_tesla, time_vector_s, multiband_gradient

    def _log(self, message):
        if self.verbose:
            print(f"[MultibandPulseDesigner] {message}")
