# File: mri_pulse_library/rf_pulses/simple/hard_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_hard_pulse(duration, flip_angle_deg, gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON):
    """
    Generates a hard RF pulse.

    Args:
        duration (float): Pulse duration in seconds.
        flip_angle_deg (float): Desired flip angle in degrees.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
            Defaults to GAMMA_HZ_PER_T_PROTON.

    Returns:
        tuple: (rf_pulse, time_vector)
            rf_pulse (np.ndarray): RF waveform in Tesla.
            time_vector (np.ndarray): Time points in seconds.
    """
    if duration < 0:
        raise ValueError("Duration cannot be negative.")
    if duration == 0:
        return np.array([]), np.array([])

    # Convert flip angle to radians for calculation
    flip_angle_rad = np.deg2rad(flip_angle_deg)

    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    # B1_amplitude * gyromagnetic_ratio_rad_s_t * duration = flip_angle_rad
    # B1_amplitude = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * duration)

    if gyromagnetic_ratio_rad_s_t == 0:
        if flip_angle_rad == 0:
             B1_amplitude = 0.0
        else:
             raise ValueError("Cannot achieve non-zero flip angle with zero gyromagnetic ratio.")
    elif duration == 0: # Should have been caught by the first check, but for safety
        if flip_angle_rad == 0:
            B1_amplitude = 0.0
        else:
            # This case implies infinite B1, which is not feasible.
            # However, duration == 0 returns empty arrays.
            # If we reach here by some logic error, handle it.
            raise ValueError("Cannot achieve non-zero flip angle with zero duration.")
    else:
         B1_amplitude = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * duration)

    # Define time step, e.g., 1 µs.
    # For hard pulses, the exact number of samples isn't as critical as for shaped pulses,
    # but a consistent dt is good.
    # However, for very short pulses, np.arange might produce an empty array if duration < dt.
    # The problem description uses dt = 1e-6 and time = np.arange(0, duration, dt)
    # This means the last point is duration - dt.
    # A single sample pulse is often represented by a single point at t=0 or t=duration/2.
    # If duration is very small, e.g. 1e-7s, time = np.arange(0, 1e-7, 1e-6) is empty.
    # Let's ensure at least one sample if duration > 0.

    dt = 1e-6  # Standard time step (1 µs) - can be made a parameter if needed.

    if duration < dt and duration > 0: # Duration is very short but non-zero
        # Produce a single sample pulse. Effective duration is dt.
        # B1_amplitude would need to be scaled if we consider effective duration dt.
        # Or, consider it a single point representing the amplitude over 'duration'.
        # Let's stick to requested B1 for the given 'duration'.
        time = np.array([0.0]) # A single point at the beginning of the pulse interval
    elif duration == 0: # Already handled, returns empty
        time = np.array([])
    else:
        # Samples from 0 up to, but not including, duration
        time = np.arange(0, duration, dt)
        # If time becomes empty due to duration barely equal to dt or rounding, handle it.
        if time.size == 0 and duration > 0:
             time = np.array([0.0]) # Ensure at least one sample point

    n_samples = len(time)
    if n_samples == 0 : # Should only occur if duration was 0 initially.
        return np.array([]), np.array([])

    rf_pulse = np.ones(n_samples) * B1_amplitude  # Uniform B1 field (Tesla)

    return rf_pulse, time
