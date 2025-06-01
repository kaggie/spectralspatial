import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

# Import necessary simple pulse generators
from mri_pulse_library.rf_pulses.simple.hard_pulse import generate_hard_pulse
from mri_pulse_library.rf_pulses.simple.sinc_pulse import generate_sinc_pulse
from mri_pulse_library.rf_pulses.simple.gaussian_pulse import generate_gaussian_pulse

def generate_composite_pulse_sequence(sub_pulses: list,
                                      gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                                      dt: float = 1e-6):
    """
    Generates a composite RF pulse from a sequence of sub-pulses,
    allowing for different types of sub-pulses and inter-pulse delays.

    Each sub-pulse is defined by a dictionary with keys:
    - 'pulse_type' (str): 'hard', 'sinc', or 'gaussian'.
    - 'flip_angle_deg' (float): Desired flip angle in degrees.
    - 'phase_deg' (float): Phase of the sub-pulse in degrees.
    - 'duration_s' (float): Duration of the sub-pulse in seconds.
    - 'delay_s' (float, optional): Delay after this sub-pulse in seconds. Defaults to 0.
    - Additional parameters based on 'pulse_type':
        - For 'sinc': 'time_bw_product' (float, e.g., 4.0)
                      'center_freq_offset_hz' (float, optional, defaults to 0)
        - For 'gaussian': 'time_bw_product' (float, e.g., 4.0 for ~99%lobes) - Note: Gaussian definition might vary.
                           'center_freq_offset_hz' (float, optional, defaults to 0)
                           'truncation_factor' (float, optional, for Gaussian, e.g. duration = X * sigma) -
                                              Let's assume duration is primary, and shape fits within.
                                              Or, use a common definition where time_bw_product defines shape.
                                              For now, let's assume for Gaussian, like sinc, time_bw_product is a good way to define shape.

    Args:
        sub_pulses (list): A list of dictionaries, each defining a sub-pulse.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform of the composite pulse in Tesla.
            time_vector_s (np.ndarray): Time points for the composite pulse in seconds.
                                        Time vector starts from 0.
    """
    full_rf_waveform_list = []
    # total_duration_s = 0.0 # Not strictly needed for time vector starting at 0

    for pulse_info in sub_pulses:
        pulse_type = pulse_info.get('pulse_type', 'hard').lower()
        flip_angle_deg = pulse_info.get('flip_angle_deg', 0.0)
        phase_deg = pulse_info.get('phase_deg', 0.0)
        duration_s = pulse_info.get('duration_s', 0.0)
        delay_s = pulse_info.get('delay_s', 0.0)

        sub_rf = np.array([], dtype=np.complex128)

        if duration_s > 0:
            if pulse_type == 'hard':
                rf_shape, _, _ = generate_hard_pulse(
                    flip_angle_deg=flip_angle_deg,
                    duration_s=duration_s,
                    dt_s=dt,
                    gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t
                )
                # Apply phase to hard pulse as generate_hard_pulse might not have phase arg or apply it differently
                sub_rf = rf_shape * np.exp(1j * np.deg2rad(phase_deg))

            elif pulse_type == 'sinc':
                time_bw_product = pulse_info.get('time_bw_product', 4.0)
                center_freq_offset_hz = pulse_info.get('center_freq_offset_hz', 0.0)
                sub_rf, _, _ = generate_sinc_pulse(
                    flip_angle_deg=flip_angle_deg,
                    duration_s=duration_s,
                    time_bw_product=time_bw_product,
                    phase_deg=phase_deg,
                    center_freq_offset_hz=center_freq_offset_hz,
                    dt_s=dt,
                    gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t
                )

            elif pulse_type == 'gaussian':
                time_bw_product = pulse_info.get('time_bw_product', 4.0)
                center_freq_offset_hz = pulse_info.get('center_freq_offset_hz', 0.0)
                sub_rf, _, _ = generate_gaussian_pulse(
                    flip_angle_deg=flip_angle_deg,
                    duration_s=duration_s,
                    time_bw_product=time_bw_product,
                    phase_deg=phase_deg,
                    center_freq_offset_hz=center_freq_offset_hz,
                    dt_s=dt,
                    gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t
                )
            else:
                print(f"Warning: Unknown pulse_type '{pulse_type}'. Skipping sub-pulse.")
                # Consider raising ValueError for unknown pulse_type for stricter error handling.
                # raise ValueError(f"Unknown pulse_type: {pulse_type}")

            if sub_rf.size > 0 :
                 full_rf_waveform_list.append(sub_rf)
                 # total_duration_s += duration_s # Accumulate if centering or specific end time needed

        if delay_s > 0:
            num_delay_samples = int(round(delay_s / dt))
            if num_delay_samples > 0:
                full_rf_waveform_list.append(np.zeros(num_delay_samples, dtype=np.complex128))
                # total_duration_s += delay_s

    if not full_rf_waveform_list:
        return np.array([], dtype=np.complex128), np.array([])

    final_rf_waveform = np.concatenate(full_rf_waveform_list)
    num_total_samples = len(final_rf_waveform)
    time_vector_s = np.arange(num_total_samples) * dt

    return final_rf_waveform, time_vector_s
