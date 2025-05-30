# File: mri_pulse_library/rf_pulses/spectral_spatial/multiband_spsp.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_3d_multiband_spsp(duration,
                               spatial_bandwidths_hz,
                               slice_select_gradient_tm,
                               slice_positions_m,
                               spectral_bandwidths_hz, # Not directly used for common_spectral_env shape
                               spectral_center_freqs_hz,
                               flip_angle_deg,
                               n_subpulses=16,
                               gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON,
                               dt=1e-6):
    """
    Generates a 3D multi-band Spectral-Spatial Pulse (SPSP).
    This version uses a common sinc spatial subpulse shape, phase modulated for slice selectivity,
    and a common Gaussian spectral envelope shape, phase modulated for spectral band selectivity.

    Args:
        duration (float): Total pulse duration in seconds.
        spatial_bandwidths_hz (list/np.ndarray): List of spatial bandwidths (Hz) for each slice's sinc subpulse.
        slice_select_gradient_tm (float): Slice-selection gradient strength (T/m). Assumed constant during RF.
        slice_positions_m (list/np.ndarray): List of slice center positions (m) relative to isocenter.
        spectral_bandwidths_hz (list/np.ndarray): List of spectral bandwidths (Hz).
                                                  Note: Not directly used for shaping the common Gaussian spectral
                                                  envelope in this implementation. Kept for API or future use.
        spectral_center_freqs_hz (list/np.ndarray): List of center frequencies (Hz) for each spectral band, relative to carrier.
        flip_angle_deg (float): Desired total flip angle (degrees). The interpretation is complex for multiband;
                                this scales the final summed RF pulse using sum of absolute values.
        n_subpulses (int, optional): Number of subpulses in the train. Defaults to 16.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T. Defaults to GAMMA_HZ_PER_T_PROTON.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse, time_vector)
            rf_pulse (np.ndarray): Complex RF waveform in Tesla.
            time_vector (np.ndarray): Time points in seconds, starting from near 0.
    """

    if duration <= 0 or n_subpulses <= 0:
        return np.array([]), np.array([])

    # Validate inputs
    if not isinstance(spatial_bandwidths_hz, (list, np.ndarray)) or \
       not isinstance(slice_positions_m, (list, np.ndarray)) or \
       not isinstance(spectral_bandwidths_hz, (list, np.ndarray)) or \
       not isinstance(spectral_center_freqs_hz, (list, np.ndarray)):
        raise ValueError("spatial_bandwidths_hz, slice_positions_m, spectral_bandwidths_hz, and spectral_center_freqs_hz must be lists or numpy arrays.")

    num_slices = len(spatial_bandwidths_hz)
    if not (num_slices == len(slice_positions_m)):
        raise ValueError("spatial_bandwidths_hz and slice_positions_m must have the same length.")

    num_spectral_bands = len(spectral_bandwidths_hz)
    if not (num_spectral_bands == len(spectral_center_freqs_hz)):
        raise ValueError("spectral_bandwidths_hz and spectral_center_freqs_hz must have the same length.")

    if num_slices == 0 or num_spectral_bands == 0: # Must have at least one slice and one band
        return np.array([]), np.array([])


    subpulse_duration = duration / n_subpulses
    if subpulse_duration <= dt and subpulse_duration > 0 :
        num_samples_subpulse = 1
    elif subpulse_duration <= 0:
         return np.array([]), np.array([])
    else:
        num_samples_subpulse = int(round(subpulse_duration / dt))
        if num_samples_subpulse == 0 and subpulse_duration > 0:
             num_samples_subpulse = 1

    time_subpulse = (np.arange(num_samples_subpulse) - (num_samples_subpulse - 1) / 2) * dt

    total_samples = num_samples_subpulse * n_subpulses
    rf_pulse = np.zeros(total_samples, dtype=complex)

    # Time points for the center of each subpulse envelope modulation
    spectral_time_points = np.linspace(subpulse_duration/2, duration - subpulse_duration/2, n_subpulses)

    # Common Gaussian spectral envelope amplitude (before phase modulation for different bands)
    sigma_spectral_env = duration / 4.0
    if sigma_spectral_env == 0:
        common_spectral_env_amplitude = np.ones(n_subpulses) if duration > 0 else np.zeros(n_subpulses)
    else:
        common_spectral_env_amplitude = np.exp(-((spectral_time_points - duration/2)**2) / (2 * sigma_spectral_env**2))

    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    # Loop over slices
    for slice_idx in range(num_slices):
        slice_bw_hz = spatial_bandwidths_hz[slice_idx]
        slice_pos_m = slice_positions_m[slice_idx]

        if slice_bw_hz <= 0:
            raise ValueError(f"Spatial bandwidth for slice {slice_idx} must be positive.")

        # Spatial subpulse: sinc shape, phase modulated for slice position
        # Phase ramp for slice selection: exp(1j * gamma_rad_s_t * Gs * z * t)
        # This makes the k-space center of the sinc pulse shift, exciting a different slice.
        phase_ramp_for_slice = np.exp(1j * gyromagnetic_ratio_rad_s_t * slice_select_gradient_tm * slice_pos_m * time_subpulse)
        base_spatial_subpulse = np.sinc(slice_bw_hz * time_subpulse)
        spatial_subpulse_for_slice = base_spatial_subpulse * phase_ramp_for_slice

        # Loop over spectral bands
        for spec_idx in range(num_spectral_bands):
            # spec_bw_hz = spectral_bandwidths_hz[spec_idx] # Not used for shaping common_spectral_env_amplitude
            spec_center_hz = spectral_center_freqs_hz[spec_idx]

            # Spectral phase modulation for this band, applied to the common amplitude envelope
            spectral_phase_mod = np.exp(1j * 2 * np.pi * spec_center_hz * spectral_time_points)
            current_spectral_env_complex = common_spectral_env_amplitude * spectral_phase_mod

            # Combine spatial and spectral components by summing contributions
            for i in range(n_subpulses): # Iterate over subpulse temporal segments
                start_idx = i * num_samples_subpulse
                end_idx = (i + 1) * num_samples_subpulse
                # Add contribution of this slice/band to the total RF pulse
                # Each slice/band contributes equally before normalization (divide by num_slices*num_spectral_bands if needed for flip angle definition)
                rf_pulse[start_idx:end_idx] += spatial_subpulse_for_slice * current_spectral_env_complex[i]

    # Normalize for flip angle
    # Using sum of absolute values for normalization as per plan, acknowledging its complexity for multiband.
    # This means flip_angle_deg is a target for the peak B1 derived from sum(abs(pulse)).
    current_area_mag = np.sum(np.abs(rf_pulse)) * dt

    flip_angle_rad = np.deg2rad(flip_angle_deg)

    if abs(current_area_mag) < 1e-20: # rf_pulse is essentially all zeros
        if flip_angle_rad == 0:
            B1_amplitude_scalar = 0.0
            # rf_pulse remains all zeros, which is correct.
        else:
            # Cannot achieve non-zero flip if pulse integral is zero.
            # This check might be too sensitive if sum(abs()) is used, as it should always be non-negative.
            # More likely, if rf_pulse is all zeros, current_area_mag is zero.
            raise ValueError(f"Cannot achieve non-zero flip angle. Pulse energy (sum abs) is near zero. Check inputs.")
    else:
        # Scale the RF pulse so that gyromagnetic_ratio_rad_s_t * sum(abs(scaled_rf_pulse)) * dt = flip_angle_rad
        # So, B1_amplitude_scalar * sum(abs(unscaled_rf_pulse)) * dt * gyromagnetic_ratio_rad_s_t = flip_angle_rad
        B1_amplitude_scalar = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * current_area_mag)

    rf_pulse = B1_amplitude_scalar * rf_pulse # Resulting B1 in Tesla

    # Create corresponding time vector
    full_time_vector = np.zeros(total_samples)
    for i in range(n_subpulses):
        start_idx = i * num_samples_subpulse
        end_idx = (i + 1) * num_samples_subpulse
        current_subpulse_global_time_start = i * subpulse_duration
        full_time_vector[start_idx:end_idx] = time_subpulse + current_subpulse_global_time_start + subpulse_duration/2

    return rf_pulse, full_time_vector
