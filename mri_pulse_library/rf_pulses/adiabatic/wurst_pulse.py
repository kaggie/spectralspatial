# mri_pulse_library/rf_pulses/adiabatic/wurst_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_wurst_pulse(duration: float,
                         bandwidth: float, # Hz, total bandwidth of the pulse
                         flip_angle_deg: float = 180.0,
                         num_sweeps: int = 1, # Number of WURST sweeps (e.g., for WURST-2, WURST-4)
                         power_n: float = 4.0, # Exponent for amplitude modulation (A(t) = (1 - |t'|^n)^k )
                         phase_k: float = 1.0, # Exponent for amplitude modulation
                         gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                         dt: float = 1e-6):
    """
    Generates a WURST (Wideband, Uniform Rate, Smooth Truncation) adiabatic RF pulse.
    (Placeholder for detailed WURST implementation)

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz.
        flip_angle_deg (float, optional): Target flip angle in degrees. Defaults to 180.0.
        num_sweeps (int, optional): Number of frequency sweeps. Typically 1 for standard WURST.
        power_n (float, optional): Exponent 'n' in amplitude modulation (1 - |t'|^n)^k.
        phase_k (float, optional): Exponent 'k' in amplitude modulation (1 - |t'|^n)^k.
                                   Also related to the phase modulation.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds.
    """
    num_samples = max(1, int(round(duration / dt)))
    time_vector_s = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    # Normalized time from -1 to 1
    t_prime = 2 * time_vector_s / duration

    # Placeholder for WURST amplitude and phase modulation
    # A(t') = A_max * (1 - |t'|^n)^k
    # Frequency modulation: omega(t') = +/- (bandwidth/2) * t' (linear sweep for basic WURST)

    am_modulation = (1 - np.abs(t_prime)**power_n)**phase_k
    am_modulation[np.isnan(am_modulation)] = 0 # Handle potential NaNs if (1 - |t'|^n) < 0

    # Linear frequency sweep
    instantaneous_freq_hz = (bandwidth / 2.0) * t_prime * num_sweeps
    phase_modulation_rad = np.cumsum(2 * np.pi * instantaneous_freq_hz * dt)
    # Center phase for symmetry if desired, though often not critical for pulse construction
    phase_modulation_rad -= phase_modulation_rad[num_samples // 2]


    rf_pulse_shaped = am_modulation * np.exp(1j * phase_modulation_rad)

    # Scale to achieve flip_angle_deg (simplified placeholder scaling)
    flip_angle_target_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    integral_abs_shape_dt = np.sum(np.abs(rf_pulse_shaped)) * dt

    if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
        peak_B1_Tesla = 0.0 if flip_angle_target_rad == 0 else 1.0 # Avoid division by zero
    else:
        # For WURST, B1max is often set based on adiabatic condition: (gamma_rad_s_t * B1max)^2 / (d_omega/dt) >> 1
        # d_omega/dt = sweep_rate_rad_s2 = 2 * pi * bandwidth / duration
        # This placeholder scaling is a simplification.
        peak_B1_Tesla = flip_angle_target_rad / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)

    rf_pulse_tesla = peak_B1_Tesla * rf_pulse_shaped

    if not np.iscomplexobj(rf_pulse_tesla):
        rf_pulse_tesla = rf_pulse_tesla.astype(np.complex128)

    return rf_pulse_tesla, time_vector_s
