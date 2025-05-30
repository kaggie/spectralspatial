# File: mri_pulse_library/rf_pulses/spectral_spatial/spsp_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_spsp_pulse(duration,
                        spatial_bandwidth,
                        spectral_bandwidth, # Note: This parameter is in the signature but not used in the Gaussian spectral_env's sigma per original plan.
                        flip_angle_deg,
                        n_subpulses=16,
                        gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON,
                        dt=1e-6):
    """
    Generates a basic Spectral-Spatial Pulse (SPSP) using a sinc spatial subpulse
    and a Gaussian spectral envelope.

    Args:
        duration (float): Total pulse duration in seconds.
        spatial_bandwidth (float): Spatial selection bandwidth (Hz) for the sinc subpulse.
        spectral_bandwidth (float): Spectral selection bandwidth (Hz).
                                    Note: Current implementation uses a fixed sigma for the
                                    Gaussian spectral envelope (duration/4) and does not directly
                                    use this parameter for shaping the envelope. It's kept for API
                                    consistency or future refinement.
        flip_angle_deg (float): Desired total flip angle in degrees.
        n_subpulses (int, optional): Number of subpulses in the train. Defaults to 16.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
                                                   Defaults to GAMMA_HZ_PER_T_PROTON.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse, time_vector)
            rf_pulse (np.ndarray): RF waveform in Tesla (real-valued in this basic version).
            time_vector (np.ndarray): Time points in seconds, starting from near 0.
    """

    if duration <= 0 or n_subpulses <= 0:
        return np.array([]), np.array([])
    if spatial_bandwidth <= 0:
        raise ValueError("Spatial bandwidth must be positive.")
    if spectral_bandwidth <=0: # Though not used for shaping, check for validity
        raise ValueError("Spectral bandwidth must be positive.")


    subpulse_duration = duration / n_subpulses
    if subpulse_duration <= dt and subpulse_duration > 0: # if subpulse duration is less than or equal to dt
        num_samples_subpulse = 1
    elif subpulse_duration <= 0:
        return np.array([]), np.array([]) # Should be caught by earlier checks
    else:
        num_samples_subpulse = int(round(subpulse_duration / dt))
        if num_samples_subpulse == 0 and subpulse_duration > 0: # Safety for rounding very small subpulse_duration
            num_samples_subpulse = 1


    # Centered time vector for subpulse for better sinc properties
    time_subpulse = (np.arange(num_samples_subpulse) - (num_samples_subpulse - 1) / 2) * dt

    # Design spatial subpulse (e.g., sinc)
    # np.sinc(x) = sin(pi*x)/(pi*x). Argument should be unitless.
    # spatial_bandwidth * time_subpulse -> Hz * s = unitless
    spatial_subpulse = np.sinc(spatial_bandwidth * time_subpulse)

    # Design spectral envelope (e.g., Gaussian for frequency selectivity)
    # Time points for the center of each subpulse envelope modulation
    spectral_time_points = np.linspace(subpulse_duration/2, duration - subpulse_duration/2, n_subpulses)

    # Gaussian envelope for spectral part, centered at duration/2
    sigma_spectral_env = duration / 4.0 # As per original plan
    if sigma_spectral_env == 0: # if duration is tiny
        # if duration is positive but sigma is zero, make it a flat envelope
        spectral_env = np.ones(n_subpulses) if duration > 0 else np.zeros(n_subpulses)
    else:
        spectral_env = np.exp(-((spectral_time_points - duration/2)**2) / (2 * sigma_spectral_env**2))

    # Modulate subpulses with spectral envelope
    total_samples = num_samples_subpulse * n_subpulses
    rf_pulse = np.zeros(total_samples, dtype=float) # SPSP is real-valued in this version

    full_time_vector = np.zeros(total_samples)

    for i in range(n_subpulses):
        start_idx = i * num_samples_subpulse
        end_idx = (i + 1) * num_samples_subpulse

        # Sub-pulse time starts from its center, shift it to global time for the full_time_vector
        current_subpulse_global_time_start = i * subpulse_duration
        full_time_vector[start_idx:end_idx] = time_subpulse + current_subpulse_global_time_start + subpulse_duration/2

        rf_pulse[start_idx:end_idx] = spatial_subpulse * spectral_env[i]


    # Normalize for flip angle
    # For real pulses, sum(rf_pulse) is the integral for flip angle calculation.
    current_area = np.sum(rf_pulse) * dt

    flip_angle_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if abs(current_area * gyromagnetic_ratio_rad_s_t) < 1e-20:
        if flip_angle_rad == 0:
            B1_amplitude_scalar = 0.0
        else:
            raise ValueError(f"Cannot achieve non-zero flip angle. Pulse shape integral is near zero (Area: {current_area}).")
    else:
        B1_amplitude_scalar = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * current_area)

    rf_pulse = B1_amplitude_scalar * rf_pulse # Resulting B1 in Tesla

    return rf_pulse, full_time_vector
