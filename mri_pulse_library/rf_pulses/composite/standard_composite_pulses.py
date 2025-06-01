import numpy as np
from .composite_pulse import generate_composite_pulse_sequence
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_refocusing_90x_180y_90x(
        duration_90_s: float,
        duration_180_s: float,
        gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
        dt: float = 1e-6,
        subpulse_type: str = 'hard' # Allow specifying sub-pulse type
    ):
    """
    Generates a 90x - 180y - 90x composite refocusing pulse.
    This sequence is known for its robustness to off-resonance effects and B1 inhomogeneities
    for spin refocusing compared to a single 180-degree pulse.

    Args:
        duration_90_s (float): Duration of each 90-degree sub-pulse in seconds.
        duration_180_s (float): Duration of the 180-degree sub-pulse in seconds.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s.
        subpulse_type (str, optional): Type of sub-pulses to use ('hard', 'sinc', 'gaussian').
                                     Defaults to 'hard'. Note: 'sinc'/'gaussian' would require
                                     additional parameters like 'time_bw_product'. This basic
                                     version primarily supports 'hard'. For sinc/gaussian,
                                     the caller might need to construct the sub_pulses list manually
                                     or this function could be extended to accept **kwargs for them.

    Returns:
        tuple: (rf_pulse_tesla, time_vector_s)
            rf_pulse_tesla (np.ndarray): Complex RF waveform of the composite pulse in Tesla.
            time_vector_s (np.ndarray): Time points for the composite pulse in seconds.
    """
    if subpulse_type not in ['hard']: # Simplified for now, extend if needed to pass **kwargs
        # For 'sinc' or 'gaussian', this function would need to accept their specific params (e.g. time_bw_product)
        # and pass them in the dictionaries below.
        raise NotImplementedError(f"Sub-pulse type '{subpulse_type}' not yet directly supported with specific parameters in this wrapper. Use 'hard' or build the sub_pulses list manually for generate_composite_pulse_sequence.")

    sub_pulses_definition = [
        {
            'pulse_type': subpulse_type,
            'flip_angle_deg': 90.0,
            'phase_deg': 0.0,  # x-axis
            'duration_s': duration_90_s
            # If subpulse_type were 'sinc', add e.g.: 'time_bw_product': 4.0
        },
        {
            'pulse_type': subpulse_type,
            'flip_angle_deg': 180.0,
            'phase_deg': 90.0, # y-axis
            'duration_s': duration_180_s
        },
        {
            'pulse_type': subpulse_type,
            'flip_angle_deg': 90.0,
            'phase_deg': 0.0,  # x-axis
            'duration_s': duration_90_s
        }
    ]

    return generate_composite_pulse_sequence(
        sub_pulses=sub_pulses_definition,
        gyromagnetic_ratio_hz_t=gyromagnetic_ratio_hz_t,
        dt=dt
    )

# Example of another common one: 90x - 180x - 90x (less common for refocusing, more for B1)
# def generate_excitation_90x_180x_90x(...):
#    ...similar structure...
