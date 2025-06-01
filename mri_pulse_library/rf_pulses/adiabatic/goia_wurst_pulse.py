# mri_pulse_library/rf_pulses/adiabatic/goia_wurst_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_goia_wurst_pulse(duration: float,
                              bandwidth: float, # Hz, total bandwidth of the pulse
                              flip_angle_deg: float = 180.0,
                              gradient_amplitude_mT_m: float = 10.0, # Peak gradient amplitude for GOIA encoding
                              slice_thickness_m: float = 0.005, # Slice thickness for GOIA
                              power_n: float = 4.0, # Exponent for WURST amplitude modulation
                              phase_k: float = 1.0, # Exponent for WURST amplitude modulation
                              gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                              dt: float = 1e-6):
    """
    Generates a GOIA-WURST (Gradient Offset Independent Adiabaticity) RF pulse.
    This is a WURST pulse with a specific phase modulation to achieve slice selectivity
    robust to gradient variations. (Placeholder for detailed GOIA implementation)

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz for the WURST component.
        flip_angle_deg (float, optional): Target flip angle in degrees. Defaults to 180.0.
        gradient_amplitude_mT_m (float, optional): Peak amplitude of the selection gradient in mT/m.
        slice_thickness_m (float, optional): Target slice thickness in meters.
        power_n (float, optional): Exponent 'n' in WURST amplitude modulation (1 - |t'|^n)^k.
        phase_k (float, optional): Exponent 'k' in WURST amplitude modulation (1 - |t'|^n)^k.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s, gradient_waveform_mT_m)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds.
            gradient_waveform_mT_m (np.ndarray): Gradient waveform in mT/m (placeholder).
    """
    num_samples = max(1, int(round(duration / dt)))
    time_vector_s = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    # Normalized time from -1 to 1
    t_prime = 2 * time_vector_s / duration

    # WURST Amplitude Modulation
    am_modulation = (1 - np.abs(t_prime)**power_n)**phase_k
    am_modulation[np.isnan(am_modulation)] = 0

    # GOIA Phase Modulation (simplified placeholder)
    # True GOIA phase involves integrating G(t) * z, and relating G(t) to A(t)
    # For placeholder, use WURST linear frequency sweep + a conceptual quadratic phase.
    # The actual GOIA phase is more complex: phi_GOIA(t) = integral( (A(t)^2 * K_GOIA) / Gz(t) ) dt
    # where K_GOIA is a constant. Or, simpler forms exist where phase is proportional to integral of AM.

    instantaneous_freq_hz_wurst = (bandwidth / 2.0) * t_prime
    phase_modulation_rad_wurst = np.cumsum(2 * np.pi * instantaneous_freq_hz_wurst * dt)

    # Placeholder for GOIA-specific phase component (e.g., could be quadratic or related to AM integral)
    # This is highly simplified.
    goia_phase_component = np.zeros_like(phase_modulation_rad_wurst) # No GOIA effect in this placeholder

    phase_modulation_rad = phase_modulation_rad_wurst + goia_phase_component
    phase_modulation_rad -= phase_modulation_rad[num_samples // 2] # Center phase

    rf_pulse_shaped = am_modulation * np.exp(1j * phase_modulation_rad)

    # Scale to achieve flip_angle_deg (simplified placeholder scaling)
    flip_angle_target_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    integral_abs_shape_dt = np.sum(np.abs(rf_pulse_shaped)) * dt

    if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
        peak_B1_Tesla = 0.0 if flip_angle_target_rad == 0 else 1.0
    else:
        peak_B1_Tesla = flip_angle_target_rad / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)

    rf_pulse_tesla = peak_B1_Tesla * rf_pulse_shaped

    if not np.iscomplexobj(rf_pulse_tesla):
        rf_pulse_tesla = rf_pulse_tesla.astype(np.complex128)

    # Placeholder for gradient waveform (e.g., simple trapezoid or constant)
    # For GOIA, G(t) is often related to AM envelope, e.g., G(t) ~ A(t) or A(t)^2
    gradient_waveform_mT_m = np.ones(num_samples) * gradient_amplitude_mT_m # Simplistic constant gradient

    return rf_pulse_tesla, time_vector_s, gradient_waveform_mT_m
