# File: mri_pulse_library/rf_pulses/adiabatic/hs_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_hs1_pulse(duration,
                       bandwidth, # Bandwidth of the frequency sweep
                       flip_angle_deg=180.0, # Adiabatic pulses often target 180 for inversion
                       mu=4.9, # Adiabaticity factor for HS1
                       beta_rad_s=800.0, # Frequency modulation parameter (rad/s) for HS1
                       gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON,
                       dt=1e-6):
    """
    Generates a Hyperbolic Secant (HS1) adiabatic RF pulse.

    The HS1 pulse is defined by:
    - Amplitude modulation: A(t) = sech(beta * t)
    - Frequency modulation: d(phi)/dt (t) = mu * beta * tanh(beta * t)

    Args:
        duration (float): Pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz (full width).
                           Note: The actual sweep range is determined by mu and beta_rad_s.
                           This parameter serves as a target or reference. The function calculates
                           the actual bandwidth achieved with given mu and beta_rad_s and can be
                           compared by the user if needed.
        flip_angle_deg (float, optional): Target flip angle in degrees. Defaults to 180.0.
                                          Used for scaling the pulse amplitude.
        mu (float, optional): Adiabaticity factor (dimensionless). Defaults to 4.9 for HS1.
        beta_rad_s (float, optional): Parameter related to the shape of the sech/tanh
                                      functions (rad/s). Defaults to 800.0 rad/s.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
                                                   Defaults to GAMMA_HZ_PER_T_PROTON.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse, time_vector)
            rf_pulse (np.ndarray): Complex RF waveform in Tesla.
            time_vector (np.ndarray): Time points in seconds, centered around 0.
    """

    if duration <= 0:
        return np.array([]), np.array([])
    if mu <= 0 or beta_rad_s <= 0:
        raise ValueError("mu and beta_rad_s must be positive.")

    num_samples = max(1, int(round(duration / dt)))
    time = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    # Amplitude modulation: A(t) = sech(beta * t) = 1/cosh(beta * t)
    # beta_rad_s * time is in radians.
    am_envelope = 1.0 / np.cosh(beta_rad_s * time)

    # Frequency modulation (instantaneous frequency offset from carrier):
    # dw(t) = mu * beta * tanh(beta * t)  (in rad/s)
    instantaneous_freq_offset_rad_s = mu * beta_rad_s * np.tanh(beta_rad_s * time)

    # The maximum instantaneous frequency offset achieved is mu * beta_rad_s (at t -> +/- infinity for tanh).
    # The actual bandwidth (full width) swept by this definition is:
    # actual_swept_bw_hz = (mu * beta_rad_s / np.pi)
    # This can be compared to the input 'bandwidth' parameter by the user.
    # The function proceeds with the given mu and beta_rad_s.

    # Phase modulation: Integrate the instantaneous frequency
    # phi(t) = integral from -duration/2 to t of (dw(tau) dtau)
    # Or, more simply, phi(t) = integral from 0 to t of (dw(tau) dtau) if time starts at 0.
    # Since time is centered, integrate from time[0] up to current time.
    # np.cumsum(instantaneous_freq_offset_rad_s * dt) gives the phase relative to the phase at time[0].
    phase_rad = np.cumsum(instantaneous_freq_offset_rad_s * dt)
    # Optional: Center the phase accumulation if desired, e.g., subtract phase_rad[num_samples//2]
    # For RF pulse construction, absolute phase matters less than relative phase changes.

    # Create complex RF pulse shape (before scaling)
    rf_pulse_shaped = am_envelope * np.exp(1j * phase_rad)

    # Normalize the pulse amplitude
    # The flip_angle_deg for an adiabatic pulse is achieved by satisfying the adiabatic condition,
    # which depends on B1_max, sweep rate, etc., not just the integral of B1(t) like simple pulses.
    # However, a common approach to set B1_max is to scale based on a nominal flip angle using
    # the integral of the absolute value of the pulse shape.
    flip_angle_target_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    # area_term_s is sum(|shape(t)|)*dt. Peak of am_envelope is 1.
    # This term is used to scale the pulse to achieve peak_B1_Tesla.
    area_term_s = np.sum(np.abs(rf_pulse_shaped)) * dt

    if abs(area_term_s * gyromagnetic_ratio_rad_s_t) < 1e-20:
        # This would happen if rf_pulse_shaped is all zeros (e.g. dt is huge or num_samples is 0)
        # or if gyromagnetic ratio is zero.
        if flip_angle_target_rad == 0:
            peak_B1_Tesla = 0.0
        else:
            raise ValueError(f"Cannot achieve non-zero flip angle. Pulse shape integral (sum abs * dt) is near zero (Area term: {area_term_s}).")
    else:
        # Scale such that: peak_B1_Tesla * area_term_s * gamma_rad_s_t = flip_angle_target_rad
        # This means peak_B1_Tesla is the scaling factor for rf_pulse_shaped (whose peak abs value is 1).
        peak_B1_Tesla = flip_angle_target_rad / (gyromagnetic_ratio_rad_s_t * area_term_s)

    rf_pulse_final_Tesla = peak_B1_Tesla * rf_pulse_shaped # B1 in Tesla

    return rf_pulse_final_Tesla, time
