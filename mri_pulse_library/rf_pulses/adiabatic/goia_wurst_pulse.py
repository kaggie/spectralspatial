import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON, GAMMA_RAD_PER_S_PER_T_PROTON

def generate_goia_wurst_pulse(duration: float,
                              bandwidth: float, # Hz, for the underlying WURST frequency sweep
                              slice_thickness_m: float, # Target slice thickness in meters
                              peak_gradient_mT_m: float, # Peak gradient amplitude in mT/m
                              flip_angle_deg: float = 180.0,
                              power_n: float = 20.0, # WURST AM exponent 'n'
                              phase_k: float = 1.0,  # WURST AM exponent 'k'
                              goia_factor_C: float = None, # GOIA constant C (rad/Tm). If None, will be derived.
                                                           # phi_goia(t) = C * integral(G(t) * A(t)/A_max dt) / integral(A(t)/A_max dt) ? No this is not it.
                                                           # phi_goia_deriv(t) = C * G(t) * A(t) / A_max ?
                                                           # More common: phi_goia_deriv(t) ~ A(t) or A(t)^2
                                                           # Let's use a formulation where omega_GOIA(t) = K * A(t)
                              peak_b1_tesla: float = None,
                              adiabaticity_factor_Q: float = 5.0, # For WURST B1 estimation
                              gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                              dt: float = 1e-6):
    """
    Generates a GOIA-WURST (Gradient Offset Independent Adiabaticity) RF pulse.
    The implementation uses a WURST amplitude modulation and a linear frequency sweep,
    with an additional GOIA-specific phase modulation. The gradient waveform G(t)
    is designed to follow the shape of the RF amplitude modulation A(t).

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Bandwidth of the underlying WURST frequency sweep in Hz.
        slice_thickness_m (float): Target slice thickness in meters.
        peak_gradient_mT_m (float): Peak amplitude of the selection gradient in mT/m.
        flip_angle_deg (float, optional): Target flip angle. Defaults to 180.0. Used for B1 estimation if peak_b1_tesla is None.
        power_n (float, optional): WURST AM exponent 'n'. Defaults to 20.0.
        phase_k (float, optional): WURST AM exponent 'k'. Defaults to 1.0.
        goia_factor_C (float, optional): A constant determining the strength of GOIA phase modulation.
                                         If None, it's derived such that the maximum instantaneous GOIA frequency
                                         offset is gamma_rad_s_t * peak_gradient_T_m * (slice_thickness_m / 2.0).
                                         This derived value is then used as the coefficient for AM-scaled GOIA frequency.
        peak_b1_tesla (float, optional): Peak B1 amplitude. If None, estimated like WURST.
        adiabaticity_factor_Q (float, optional): Q factor for WURST B1 estimation.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s, gradient_waveform_mT_m)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds.
            gradient_waveform_mT_m (np.ndarray): Gradient waveform in mT/m.
    """

    if duration <= 0:
        return np.array([], dtype=np.complex128), np.array([]), np.array([])
    if slice_thickness_m <= 0 or peak_gradient_mT_m < 0:
        raise ValueError("slice_thickness_m must be > 0, peak_gradient_mT_m must be >= 0.")

    num_samples = max(1, int(round(duration / dt)))
    time_vector_s = (np.arange(num_samples) - (num_samples - 1) / 2) * dt

    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi
    peak_gradient_T_m = peak_gradient_mT_m * 1e-3

    # 1. Generate WURST AM envelope (normalized to 1.0 peak)
    if duration > dt/2: # Avoid division by zero if duration is very small
        t_prime = np.clip((2.0 * time_vector_s) / duration, -1.0, 1.0)
    elif num_samples == 1:
        t_prime = np.array([0.0])
    else: # Should ideally not be reached if num_samples = max(1, round(duration/dt))
        t_prime = np.zeros(num_samples)

    base_am = 1.0 - np.abs(t_prime)**power_n
    base_am = np.maximum(base_am, 0) # Ensure non-negative base
    am_envelope_normalized = base_am**phase_k
    am_envelope_normalized[np.isnan(am_envelope_normalized)] = 0

    # Normalize AM envelope to have a peak of 1, if it's not all zeros
    max_am = np.max(am_envelope_normalized)
    if max_am > 1e-9:
        am_envelope_normalized /= max_am
    else: # If AM envelope is all zeros (e.g. power_n very high and k very low on discrete grid)
        am_envelope_normalized = np.zeros(num_samples) # Ensure it's all zero

    # 2. Design Gradient Waveform G(t) ~ AM(t)
    gradient_waveform_T_m = peak_gradient_T_m * am_envelope_normalized
    gradient_waveform_mT_m = gradient_waveform_T_m * 1e3

    # 3. WURST Frequency Modulation (linear sweep component)
    instantaneous_freq_wurst_hz = (bandwidth / 2.0) * t_prime
    phase_wurst_rad = np.cumsum(2 * np.pi * instantaneous_freq_wurst_hz * dt)
    if num_samples > 0 and len(phase_wurst_rad) > 0:
        phase_wurst_rad -= phase_wurst_rad[num_samples // 2]

    # 4. GOIA-specific Frequency Modulation component
    if goia_factor_C is None:
        # Max instantaneous GOIA freq offset at the center of the slice (z = slice_thickness_m / 2)
        # This is the coefficient K in omega_GOIA(t) = K * AM_normalized(t)
        # such that K * max(AM_normalized) = gamma * G_max * (slice_thickness_m / 2)
        # Since max(AM_normalized) is 1, K = gamma * G_max * (slice_thickness_m / 2)
        C_eff_rad_s = gyromagnetic_ratio_rad_s_t * peak_gradient_T_m * (slice_thickness_m / 2.0)
    else:
        C_eff_rad_s = goia_factor_C

    instantaneous_freq_goia_rad_s = C_eff_rad_s * am_envelope_normalized

    phase_goia_rad = np.cumsum(instantaneous_freq_goia_rad_s * dt)
    if num_samples > 0 and len(phase_goia_rad) > 0:
         phase_goia_rad -= phase_goia_rad[0] # Start GOIA phase accumulation from 0

    total_phase_rad = phase_wurst_rad + phase_goia_rad

    # 5. Determine Peak B1 amplitude
    if peak_b1_tesla is None:
        if bandwidth != 0 and duration > 0:
            sweep_rate_rad_s2 = abs((2 * np.pi * bandwidth) / duration)
            if sweep_rate_rad_s2 > 1e-9:
                b1_max_target_rad_s = adiabaticity_factor_Q * np.sqrt(sweep_rate_rad_s2)
                actual_peak_b1_tesla = b1_max_target_rad_s / gyromagnetic_ratio_rad_s_t
            else:
                integral_abs_shape_dt = np.sum(am_envelope_normalized) * dt
                if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
                    actual_peak_b1_tesla = 0.0 if np.deg2rad(flip_angle_deg) == 0 else 1e-7
                else:
                    actual_peak_b1_tesla = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
                # print(f"Warning: GOIA-WURST peak_b1_tesla estimated via integral scaling (WURST BW near zero). Adiabaticity may not be met. B1_max: {actual_peak_b1_tesla*1e6:.2f} uT")
        else:
            integral_abs_shape_dt = np.sum(am_envelope_normalized) * dt
            if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
                actual_peak_b1_tesla = 0.0 if np.deg2rad(flip_angle_deg) == 0 else 1e-7
            else:
                actual_peak_b1_tesla = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
            # print(f"Warning: GOIA-WURST peak_b1_tesla estimated via integral scaling. Adiabaticity may not be met. B1_max: {actual_peak_b1_tesla*1e6:.2f} uT")
    else:
        actual_peak_b1_tesla = peak_b1_tesla

    # 6. Final RF pulse
    rf_pulse_final_tesla = (actual_peak_b1_tesla * am_envelope_normalized) * np.exp(1j * total_phase_rad)

    if rf_pulse_final_tesla.size == 0:
        rf_pulse_final_tesla = np.array([], dtype=np.complex128)
    elif not np.iscomplexobj(rf_pulse_final_tesla):
        rf_pulse_final_tesla = rf_pulse_final_tesla.astype(np.complex128)

    return rf_pulse_final_tesla, time_vector_s, gradient_waveform_mT_m
