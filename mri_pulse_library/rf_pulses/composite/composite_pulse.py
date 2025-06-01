# mri_pulse_library/rf_pulses/composite/composite_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
# Assuming simple hard pulses can be generated internally or by calling a utility
# from mri_pulse_library.rf_pulses.simple.hard_pulse import generate_hard_pulse # If we want to use it

def generate_hard_pulse_basic(duration, flip_angle_deg, phase_deg, gyromagnetic_ratio_hz_t, dt):
    """Generates a simple hard pulse component for composite pulses."""
    num_samples = max(1, int(round(duration / dt)))
    time_vector = np.arange(num_samples) * dt

    flip_angle_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if duration > 0 and gyromagnetic_ratio_rad_s_t > 0:
        amplitude_tesla = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * duration)
    else:
        amplitude_tesla = 0

    phase_rad = np.deg2rad(phase_deg)

    rf_waveform = amplitude_tesla * np.exp(1j * phase_rad) * np.ones(num_samples)
    return rf_waveform, time_vector

def generate_composite_pulse_sequence(sub_pulses: list,
                                      gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                                      dt: float = 1e-6):
    """
    Generates a composite RF pulse from a sequence of sub-pulses.

    Each sub-pulse is defined by a dictionary with keys like:
    'flip_angle_deg', 'phase_deg', 'duration'.
    For now, all sub-pulses are generated as hard pulses.

    Args:
        sub_pulses (list): A list of dictionaries. Each dictionary defines a sub-pulse.
                           Required keys: 'flip_angle_deg' (float), 'phase_deg' (float),
                                          'duration' (float).
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform of the composite pulse in Tesla.
            time_vector_s (np.ndarray): Time points in seconds for the composite pulse.
    """
    full_rf_waveform = np.array([], dtype=np.complex128)
    current_time_offset = 0.0

    for pulse_info in sub_pulses:
        flip_angle_deg = pulse_info.get('flip_angle_deg', 0)
        phase_deg = pulse_info.get('phase_deg', 0)
        duration = pulse_info.get('duration', 0)

        if duration <= 0:
            # Potentially skip zero-duration pulses or handle as needed
            continue

        # For this version, we use a basic internal hard pulse generator.
        # In a more advanced version, this could call other pulse generation functions
        # based on a 'pulse_type' key in pulse_info.
        sub_rf, _ = generate_hard_pulse_basic(
            duration=duration,
            flip_angle_deg=flip_angle_deg,
            phase_deg=phase_deg,
            gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t,
            dt=dt
        )

        full_rf_waveform = np.concatenate((full_rf_waveform, sub_rf))
        current_time_offset += duration

    total_samples = len(full_rf_waveform)
    time_vector_s = np.arange(total_samples) * dt

    return full_rf_waveform, time_vector_s

# Example usage (can be removed or kept for testing):
# if __name__ == '__main__':
#     # Example: A simple 90x - 180y refocusing pulse (nominal)
#     # Durations are arbitrary for this example
#     example_sequence = [
#         {'flip_angle_deg': 90, 'phase_deg': 0, 'duration': 1e-3},   # 90x
#         {'flip_angle_deg': 180, 'phase_deg': 90, 'duration': 2e-3}  # 180y
#     ]
#     rf_composite, t_composite = generate_composite_pulse_sequence(example_sequence)
#     print(f"Generated composite pulse with {len(rf_composite)} samples.")
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # plt.subplot(2,1,1)
#     # plt.plot(t_composite * 1000, np.abs(rf_composite))
#     # plt.ylabel('Amplitude (T)')
#     # plt.subplot(2,1,2)
#     # plt.plot(t_composite * 1000, np.angle(rf_composite, deg=True))
#     # plt.ylabel('Phase (deg)')
#     # plt.xlabel('Time (ms)')
#     # plt.show()
