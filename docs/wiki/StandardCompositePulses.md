# Composite Pulses

Composite pulses in MRI consist of a sequence of simpler sub-pulses applied contiguously or with short delays. They are designed to achieve a desired magnetization manipulation (e.g., excitation, inversion, refocusing) with improved performance compared to a single pulse, particularly in the presence of experimental imperfections like B0 (off-resonance) or B1 (RF field inhomogeneity) variations.

## Generic Composite Pulse Generation

The function `generate_composite_pulse_sequence` in `mri_pulse_library.rf_pulses.composite.composite_pulse` provides a flexible way to construct arbitrary composite pulses.

### Function: `generate_composite_pulse_sequence`

#### Key Parameters:

*   `sub_pulses` (list): A list of dictionaries, where each dictionary defines a sub-pulse. Key fields for each sub-pulse dictionary:
    *   `'pulse_type'` (str): Type of the sub-pulse. Supported: `'hard'`, `'sinc'`, `'gaussian'`.
    *   `'flip_angle_deg'` (float): Flip angle of the sub-pulse in degrees.
    *   `'phase_deg'` (float): Phase of the sub-pulse in degrees.
    *   `'duration_s'` (float): Duration of the sub-pulse in seconds.
    *   `'delay_s'` (float, optional): Delay *after* this sub-pulse in seconds. Default: 0.
    *   Additional parameters based on `pulse_type`:
        *   For `'sinc'`: `'time_bw_product'` (float), `'center_freq_offset_hz'` (optional).
        *   For `'gaussian'`: `'time_bw_product'` (float), `'center_freq_offset_hz'` (optional).
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio. Default: Proton.
*   `dt` (float, optional): Time step in seconds. Default: 1e-6 s.

#### Returns:

*   `rf_pulse_tesla` (np.ndarray): Complex RF waveform of the full composite pulse.
*   `time_vector_s` (np.ndarray): Corresponding time vector.

#### Example: Custom Sequence using `generate_composite_pulse_sequence`

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.rf_pulses.composite.composite_pulse import generate_composite_pulse_sequence

# Define a custom composite pulse: 90(x) - delay - 90(y) using hard pulses
custom_sequence = [
    {'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 0, 'duration_s': 0.0005, 'delay_s': 0.0002},
    {'pulse_type': 'hard', 'flip_angle_deg': 90, 'phase_deg': 90, 'duration_s': 0.0005}
]

rf_custom, time_custom = generate_composite_pulse_sequence(sub_pulses=custom_sequence, dt=1e-6)

print(f"Generated custom composite pulse with {len(rf_custom)} samples.")

# Plotting (optional)
fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(time_custom * 1000, np.abs(rf_custom) * 1e6)
axs[0].set_ylabel('Amplitude (µT)')
axs[0].set_title('Custom Composite Pulse (90x - delay - 90y)')
axs[1].plot(time_custom * 1000, np.angle(rf_custom, deg=True))
axs[1].set_ylabel('Phase (degrees)')
axs[1].set_xlabel('Time (ms)')
plt.show()
```

## Standard Composite Pulses

The module `mri_pulse_library.rf_pulses.composite.standard_composite_pulses` provides functions for generating well-known composite pulse sequences.

### 1. 90x - 180y - 90x Refocusing Pulse

This is a widely used composite pulse for spin refocusing, known for its improved off-resonance performance and B1 insensitivity compared to a single 180-degree pulse.

#### Function: `generate_refocusing_90x_180y_90x`

##### Key Parameters:

*   `duration_90_s` (float): Duration of each 90-degree sub-pulse in seconds.
*   `duration_180_s` (float): Duration of the 180-degree sub-pulse in seconds.
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio. Default: Proton.
*   `dt` (float, optional): Time step. Default: 1e-6 s.
*   `subpulse_type` (str, optional): Type of sub-pulses. Currently, `'hard'` is directly supported by this convenience function. For other types, use `generate_composite_pulse_sequence` manually.

##### Returns:

*   `rf_pulse_tesla` (np.ndarray): Complex RF waveform.
*   `time_vector_s` (np.ndarray): Time vector.

##### Example: `generate_refocusing_90x_180y_90x`

```python
from mri_pulse_library.rf_pulses.composite.standard_composite_pulses import generate_refocusing_90x_180y_90x

# Parameters for the 90x-180y-90x pulse
dur_90 = 0.0005  # 0.5 ms for 90-deg pulses
dur_180 = 0.001 # 1.0 ms for 180-deg pulse

rf_refocus, time_refocus = generate_refocusing_90x_180y_90x(
    duration_90_s=dur_90,
    duration_180_s=dur_180,
    dt=1e-6
)

print(f"Generated 90x-180y-90x refocusing pulse with {len(rf_refocus)} samples.")

# Plotting (optional)
fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(time_refocus * 1000, np.abs(rf_refocus) * 1e6)
axs[0].set_ylabel('Amplitude (µT)')
axs[0].set_title('90x - 180y - 90x Composite Refocusing Pulse')
axs[1].plot(time_refocus * 1000, np.angle(rf_refocus, deg=True))
axs[1].set_ylabel('Phase (degrees)')
axs[1].set_xlabel('Time (ms)')
plt.show()

```

## References

*   Levitt, M. H. (1986). *Composite pulses.* Progress in Nuclear Magnetic Resonance Spectroscopy, 18(2), 61-122. (Classic review on composite pulses).
*   Freeman, R. (1998). *Spin choreography: basic steps in high resolution NMR.* Oxford University Press. (Chapter on composite pulses).
