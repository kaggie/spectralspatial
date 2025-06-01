import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_wurst_pulse(duration: float,
                         bandwidth: float, # Hz, total bandwidth of the pulse
                         flip_angle_deg: float = 180.0, # Target flip angle (for B1 estimation if not adiabatic)
                         power_n: float = 20.0, # Exponent 'n' in AM: (1 - |t'|^n)^k
                         phase_k: float = 1.0,  # Exponent 'k' in AM: (1 - |t'|^n)^k
                         adiabaticity_factor_Q: float = 5.0, # Factor Q for B1max estimation (B1max ~ Q*sqrt(sweep_rate))
                         peak_b1_tesla: float = None, # Optional: directly set peak B1 (Tesla). If None, estimated.
                         gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                         dt: float = 1e-6):
    """
    Generates a WURST (Wideband, Uniform Rate, Smooth Truncation) adiabatic RF pulse.

    Amplitude Modulation (AM): A(t') = (1 - |t'|^power_n)^phase_k
        where t' is time normalized from -1 to 1.
    Frequency Modulation (FM): Linear sweep across 'bandwidth'.
        f(t') = (bandwidth / 2) * t' (Hz)

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz (full width).
        flip_angle_deg (float, optional): Target flip angle. Used for B1 estimation if
                                          peak_b1_tesla is not provided and adiabatic estimation
                                          is not chosen/fails. Defaults to 180.0.
        power_n (float, optional): Exponent 'n' for amplitude modulation. Controls flatness
                                   and transition sharpness. Defaults to 20.0.
        phase_k (float, optional): Exponent 'k' for amplitude modulation. Controls smoothness
                                   of truncation. Defaults to 1.0.
        adiabaticity_factor_Q (float, optional): Factor used for estimating peak B1 amplitude
                                     to satisfy adiabatic condition (B1_max_rad_s ~ Q * sqrt(sweep_rate_rad_s2)).
                                     Defaults to 5.0.
        peak_b1_tesla (float, optional): If provided, this value is used as the peak B1 amplitude
                                         in Tesla. If None (default), peak B1 is estimated using
                                         the adiabaticity_factor_Q and sweep rate.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds, centered around 0.
    """

    if duration <= 0:
        return np.array([], dtype=np.complex128), np.array([])
    if power_n < 1 or phase_k < 0: # phase_k can be 0 for pure window
        raise ValueError("power_n must be >= 1 and phase_k must be >= 0.")
    # No specific check for bandwidth == 0 here, as it's handled in B1 estimation logic.

    num_samples = max(1, int(round(duration / dt)))
    time_vector_s = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    # Normalized time from -1 to 1
    if duration > dt/2 : # Avoid division by zero or large t_prime if duration is very small
        t_prime = (2.0 * time_vector_s) / duration
        t_prime = np.clip(t_prime, -1.0, 1.0)
    elif num_samples == 1: # Single point pulse (duration is likely <= dt)
        t_prime = np.array([0.0])
    else: # duration is very small, less than dt/2, but somehow num_samples > 1 (should not happen with max(1, round(duration/dt)))
          # This implies round(duration/dt) yielded 0, then max(1,0) = 1. So num_samples should be 1.
          # If duration is extremely small (e.g., dt/100) leading to num_samples=1, t_prime=[0] is correct.
        t_prime = np.zeros(num_samples)


    # Amplitude Modulation: (1 - |t'|^n)^k
    base_am = 1.0 - np.abs(t_prime)**power_n
    base_am = np.maximum(base_am, 0) # Ensure non-negative base for power k (replaces base_am[base_am < 0] = 0)

    am_envelope = base_am**phase_k
    am_envelope[np.isnan(am_envelope)] = 0

    # Frequency Modulation (linear sweep)
    instantaneous_freq_hz = (bandwidth / 2.0) * t_prime

    phase_rad = np.cumsum(2 * np.pi * instantaneous_freq_hz * dt)
    if num_samples > 0 and len(phase_rad) > 0: # Check if phase_rad is not empty
        phase_rad -= phase_rad[num_samples // 2]


    rf_pulse_shaped = am_envelope * np.exp(1j * phase_rad)

    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if peak_b1_tesla is None:
        if bandwidth != 0 and duration > 0: # Check bandwidth is not zero for sweep rate calc
            sweep_rate_rad_s2 = abs((2 * np.pi * bandwidth) / duration) # Use abs for safety, though bandwidth usually positive

            if sweep_rate_rad_s2 > 1e-9: # Ensure sweep_rate is meaningfully positive
                b1_max_target_rad_s = adiabaticity_factor_Q * np.sqrt(sweep_rate_rad_s2)
                actual_peak_b1_tesla = b1_max_target_rad_s / gyromagnetic_ratio_rad_s_t
            else:
                integral_abs_shape_dt = np.sum(np.abs(rf_pulse_shaped)) * dt
                if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
                    actual_peak_b1_tesla = 0.0 if np.deg2rad(flip_angle_deg) == 0 else 1e-7
                else:
                    actual_peak_b1_tesla = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
                # print(f"Warning: WURST pulse peak_b1_tesla estimated via integral scaling due to near-zero sweep rate. Adiabaticity may not be met. B1_max: {actual_peak_b1_tesla*1e6:.2f} uT")
        else:
            integral_abs_shape_dt = np.sum(np.abs(rf_pulse_shaped)) * dt
            if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
                actual_peak_b1_tesla = 0.0 if np.deg2rad(flip_angle_deg) == 0 else 1e-7
            else:
                actual_peak_b1_tesla = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
            # print(f"Warning: WURST pulse peak_b1_tesla estimated via integral scaling (bandwidth or duration was zero/small). Adiabaticity may not be met. B1_max: {actual_peak_b1_tesla*1e6:.2f} uT")
    else:
        actual_peak_b1_tesla = peak_b1_tesla

    rf_pulse_final_tesla = actual_peak_b1_tesla * rf_pulse_shaped

    if rf_pulse_final_tesla.size == 0:
        rf_pulse_final_tesla = np.array([], dtype=np.complex128)
    elif not np.iscomplexobj(rf_pulse_final_tesla): # Ensure complex output even if shape makes it real (e.g. all zeros)
        rf_pulse_final_tesla = rf_pulse_final_tesla.astype(np.complex128)

    return rf_pulse_final_tesla, time_vector_s
