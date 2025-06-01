# mri_pulse_library/rf_pulses/adiabatic/bir4_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_bir4_pulse(duration: float,
                        bandwidth: float,
                        flip_angle_deg: float = 180.0,
                        beta_bir4: float = 10.0,
                        kappa_bir4: float = np.pi/2,
                        delta_omega_0_scaling: float = 1.0,
                        gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                        dt: float = 1e-6):
    """
    Generates a BIR-4 (B1-Insensitive Rotation) adiabatic RF pulse.
    (Placeholder for detailed BIR-4 implementation)

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz.
        flip_angle_deg (float, optional): Target flip angle in degrees. Defaults to 180.0.
        beta_bir4 (float, optional): BIR-4 parameter, controls sharpness of transitions.
        kappa_bir4 (float, optional): BIR-4 phase parameter.
        delta_omega_0_scaling (float, optional): Factor to scale the off-resonance for segments.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds.
    """
    num_samples = max(1, int(round(duration / dt)))
    time_vector_s = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    # Placeholder for BIR-4 amplitude and phase modulation
    am_modulation = np.ones(num_samples) # Simplified placeholder
    phase_modulation_rad = np.zeros(num_samples) # Simplified placeholder

    rf_pulse_shaped = am_modulation * np.exp(1j * phase_modulation_rad)

    flip_angle_target_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    integral_abs_shape_dt = np.sum(np.abs(rf_pulse_shaped)) * dt

    if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
        peak_B1_Tesla = 0.0 if flip_angle_target_rad == 0 else 1.0 # Avoid division by zero
    else:
        peak_B1_Tesla = flip_angle_target_rad / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)

    rf_pulse_tesla = peak_B1_Tesla * rf_pulse_shaped

    if not np.iscomplexobj(rf_pulse_tesla):
        rf_pulse_tesla = rf_pulse_tesla.astype(np.complex128)

    return rf_pulse_tesla, time_vector_s
