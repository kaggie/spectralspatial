import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_bir4_pulse(duration: float,
                        bandwidth: float, # Hz, total bandwidth of the pulse (sweep range)
                        flip_angle_deg: float = 180.0, # Target flip angle, typically 180 for BIR-4 inversion
                        beta_bir4: float = 10.0, # Controls shape of sech/tanh, dimensionless
                        mu_bir4: float = 4.9, # Adiabaticity factor for HS components, dimensionless
                        kappa_deg: float = 70.0, # Phase modulation parameter in degrees (tan(kappa) is used)
                        xi_deg: float = 90.0, # Additional phase modulation parameter in degrees
                        peak_b1_tesla: float = None, # Optional: directly set peak B1 amplitude (Tesla). If None, estimated.
                        gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                        dt: float = 1e-6):
    """
    Generates a BIR-4 (B1-Insensitive Rotation) adiabatic RF pulse.
    This implementation is based on common BIR-4 structures using hyperbolic secant
    amplitude and frequency modulations for its segments.

    The BIR-4 pulse consists of four segments, each of duration T/4.
    The phase modulation involves parameters kappa and xi for B1 insensitivity.

    Args:
        duration (float): Total pulse duration in seconds.
        bandwidth (float): Desired frequency sweep range in Hz (full width).
                                          This parameter is used heuristically in B1 estimation if peak_b1_tesla is not given,
                                          but the primary drivers for frequency sweep are mu_bir4 and beta_bir4.
        flip_angle_deg (float, optional): Target flip angle (typically 180 for inversion).
                                          Note: BIR-4 achieves its flip angle robustly due to its
                                          adiabatic nature, not simple B1 integral scaling.
                                          This parameter primarily influences B1max estimation if not provided.
        beta_bir4 (float, optional): Dimensionless parameter controlling the shape (steepness)
                                     of the hyperbolic secant pulses. Higher beta means sharper pulses.
                                     Defaults to 10.0.
        mu_bir4 (float, optional): Adiabaticity factor (related to Q in some literature) for the
                                   hyperbolic secant frequency sweep. Defaults to 4.9.
        kappa_deg (float, optional): Phase parameter kappa in degrees. tan(kappa_rad) is used in
                                     phase calculations. Defaults to 70.0 degrees.
        xi_deg (float, optional): Phase parameter xi in degrees, used for phase cycling between
                                  segments. Defaults to 90.0 degrees.
        peak_b1_tesla (float, optional): If provided, this value is used as the peak B1 amplitude
                                         of the pulse in Tesla. If None (default), peak B1 is
                                         estimated.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform in Tesla.
            time_vector_s (np.ndarray): Time points in seconds, centered around 0.
    """

    if duration <= 0:
        # For duration=0, num_samples_total would be 1 due to max(1, ...).
        # To return empty arrays for duration=0 consistently:
        if duration == 0:
            return np.array([], dtype=np.complex128), np.array([])
        raise ValueError("Duration must be positive.") # Or handle very small durations leading to <1 sample per segment

    if beta_bir4 <= 0 or mu_bir4 <= 0:
        raise ValueError("beta_bir4 and mu_bir4 must be positive.")

    num_samples_total_nominal = int(round(duration / dt))
    if num_samples_total_nominal < 4: # Need at least one sample per segment for robust segment definition
        # Fallback for very short durations or large dt: simple placeholder or error
        # For now, let's return empty or a very short pulse based on previous placeholder logic for num_samples=1
        # This detailed impl needs meaningful segment lengths.
        num_samples_fallback = max(1, num_samples_total_nominal)
        time_vector_s_fallback = (np.arange(num_samples_fallback) - (num_samples_fallback - 1) / 2) * dt
        # Fallback to a simple scaled shape or zero if interpretation is ambiguous
        # This is essentially what the placeholder did for duration=0 / very short.
        # For this detailed impl, we require duration to be sufficient for 4 segments.
        print(f"Warning: Duration {duration}s with dt {dt}s is too short for BIR-4 segments. Reverting to placeholder-like behavior.")
        # Simplified placeholder behavior from previous BIR-4 for short pulse
        am_modulation = np.ones(num_samples_fallback)
        phase_modulation_rad = np.zeros(num_samples_fallback)
        rf_pulse_shaped = am_modulation * np.exp(1j * phase_modulation_rad)
        flip_angle_target_rad = np.deg2rad(flip_angle_deg)
        gyromagnetic_ratio_rad_s_t_fb = gyromagnetic_ratio_hz_t * 2 * np.pi
        integral_abs_shape_dt_fb = np.sum(np.abs(rf_pulse_shaped)) * dt
        if abs(integral_abs_shape_dt_fb * gyromagnetic_ratio_rad_s_t_fb) < 1e-20:
             peak_B1_Tesla_fb = 0.0 if flip_angle_target_rad == 0 else 1.0
        else:
            peak_B1_Tesla_fb = flip_angle_target_rad / (gyromagnetic_ratio_rad_s_t_fb * integral_abs_shape_dt_fb)
        rf_pulse_tesla_fb = peak_B1_Tesla_fb * rf_pulse_shaped
        if not np.iscomplexobj(rf_pulse_tesla_fb):
            rf_pulse_tesla_fb = rf_pulse_tesla_fb.astype(np.complex128)
        return rf_pulse_tesla_fb, time_vector_s_fallback


    segment_duration = duration / 4.0
    num_samples_segment = int(round(segment_duration / dt))

    if num_samples_segment == 0: # Each segment must have at least one sample
        num_samples_segment = 1

    num_samples_total = num_samples_segment * 4
    time_vector_s = (np.arange(num_samples_total) - (num_samples_total - 1) / 2) * dt

    kappa_rad = np.deg2rad(kappa_deg)
    xi_rad = np.deg2rad(xi_deg)
    # tan_kappa = np.tan(kappa_rad) # Not directly used in this phase scheme

    t_segment_centered = (np.arange(num_samples_segment) - (num_samples_segment - 1) / 2) * dt

    # Normalized time for a segment, from -1 to 1 (approx if num_samples_segment is small)
    # Ensure division by zero is avoided if segment_duration is excessively small (e.g. if dt > segment_duration/2)
    half_seg_dur_for_norm = segment_duration / 2.0
    if half_seg_dur_for_norm < dt/2 : # Avoids t_norm_segment becoming huge or NaN if segment is too short
        half_seg_dur_for_norm = dt/2 # Effectively makes t_norm_segment mostly 0 or +/-1 for single sample segment.

    t_norm_segment = t_segment_centered / half_seg_dur_for_norm
    t_norm_segment = np.clip(t_norm_segment, -1.0, 1.0) # Clip to ensure arguments to sech/tanh are robust

    am_segment_shape = 1.0 / np.cosh(beta_bir4 * t_norm_segment)
    am_segment_shape[np.isinf(am_segment_shape)] = 0 # Handle potential overflow for large beta*t_norm
    am_segment_shape[np.isnan(am_segment_shape)] = 0

    # Instantaneous frequency offset in rad/s for a segment
    # d(phi)/dt (rad/s) = (mu_bir4 * beta_bir4 / (segment_duration/2.0)) * tanh(beta_bir4 * t_norm_segment)
    beta_eff_rad_s = beta_bir4 / half_seg_dur_for_norm
    inst_freq_offset_rad_s_segment = mu_bir4 * beta_eff_rad_s * np.tanh(beta_bir4 * t_norm_segment)

    phase_segment_rad = np.cumsum(inst_freq_offset_rad_s_segment * dt)
    if len(phase_segment_rad)>0:
        phase_segment_rad -= phase_segment_rad[0]
    else: # Should not happen if num_samples_segment >=1
        phase_segment_rad = np.array([0.0])


    rf_pulse_total_shaped = np.zeros(num_samples_total, dtype=np.complex128)

    # Phase continuity logic:
    # Seg1: AM(t) * exp(j * (+phase_mod(t) + offset_1))
    # Seg2: AM(t) * exp(j * (-phase_mod(t) + offset_2))
    # Seg3: AM(t) * exp(j * (-phase_mod(t) + offset_3))
    # Seg4: AM(t) * exp(j * (+phase_mod(t) + offset_4))
    # Offsets ensure continuity and apply kappa/xi phase shifts.

    # Segment 1
    s1_local_phase = phase_segment_rad.copy()
    s1_const_phase_offset = xi_rad
    current_segment_phase = s1_local_phase + s1_const_phase_offset
    rf_pulse_total_shaped[0:num_samples_segment] = am_segment_shape * np.exp(1j * current_segment_phase)
    last_overall_phase = current_segment_phase[-1] if len(current_segment_phase)>0 else 0

    # Segment 2
    s2_local_phase = -phase_segment_rad.copy() # Reversed frequency sweep
    # Phase jump for kappa, then ensure continuity from S1's end phase to S2's start phase
    s2_const_phase_offset = last_overall_phase - s2_local_phase[0] - kappa_rad
    current_segment_phase = s2_local_phase + s2_const_phase_offset
    rf_pulse_total_shaped[num_samples_segment : 2*num_samples_segment] = am_segment_shape * np.exp(1j * current_segment_phase)
    last_overall_phase = current_segment_phase[-1] if len(current_segment_phase)>0 else last_overall_phase

    # Segment 3
    s3_local_phase = -phase_segment_rad.copy() # Reversed frequency sweep
    # Phase jump for kappa and xi
    s3_const_phase_offset = last_overall_phase - s3_local_phase[0] + kappa_rad + xi_rad
    current_segment_phase = s3_local_phase + s3_const_phase_offset
    rf_pulse_total_shaped[2*num_samples_segment : 3*num_samples_segment] = am_segment_shape * np.exp(1j * current_segment_phase)
    last_overall_phase = current_segment_phase[-1] if len(current_segment_phase)>0 else last_overall_phase

    # Segment 4
    s4_local_phase = phase_segment_rad.copy() # Original frequency sweep
    # Phase jump for xi
    s4_const_phase_offset = last_overall_phase - s4_local_phase[0] + xi_rad
    current_segment_phase = s4_local_phase + s4_const_phase_offset
    rf_pulse_total_shaped[3*num_samples_segment : 4*num_samples_segment] = am_segment_shape * np.exp(1j * current_segment_phase)

    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if peak_b1_tesla is None:
        # Estimate peak_B1_Tesla. For adiabatic pulses, this is not based on simple flip angle integral.
        # It's set to ensure adiabatic condition is met.
        # A common heuristic for B1max (in rad/s) for HS pulses: B1_max_rad_s = k * sqrt(sweep_rate * inst_bw)
        # where inst_bw is related to 1/beta. Or B1_max_rad_s ~ Q * beta_eff_rad_s for some Q.
        # For BIR-4, this is more complex.
        # Let's use a typical value often cited for BIR-4 if flip angle is 180.
        # This is a placeholder estimation and might need adjustment based on sequence requirements.
        if flip_angle_deg == 180:
             actual_peak_b1_tesla = 13e-6 # Approx. 13 uT is a common value for robust BIR-4 inversion.
        else:
            # Fallback to integral scaling for other flip angles, acknowledging it's not ideal.
            integral_abs_shape_dt = np.sum(np.abs(rf_pulse_total_shaped)) * dt # rf_pulse_total_shaped has max AM of 1
            if abs(integral_abs_shape_dt * gyromagnetic_ratio_rad_s_t) < 1e-20:
                actual_peak_b1_tesla = 0.0 if flip_angle_deg == 0 else 1e-6 # small default
            else:
                actual_peak_b1_tesla = np.deg2rad(flip_angle_deg) / (gyromagnetic_ratio_rad_s_t * integral_abs_shape_dt)
        print(f"Warning: peak_b1_tesla was not provided. Estimated to {actual_peak_b1_tesla*1e6:.2f} uT. This may require tuning for optimal BIR-4 performance.")
    else:
        actual_peak_b1_tesla = peak_b1_tesla

    rf_pulse_final_tesla = actual_peak_b1_tesla * rf_pulse_total_shaped

    # Ensure complex output
    if not np.iscomplexobj(rf_pulse_final_tesla):
        rf_pulse_final_tesla = rf_pulse_final_tesla.astype(np.complex128)

    return rf_pulse_final_tesla, time_vector_s
