# WURST Adiabatic Pulse

**WURST (Wideband, Uniform Rate, Smooth Truncation)** pulses are a type of adiabatic pulse known for their excellent performance in achieving uniform excitation or inversion over a wide range of frequencies and B1 field strengths.

## Theory Overview

The WURST pulse design is characterized by:
-   **Amplitude Modulation (AM):** The RF amplitude envelope is shaped by the function `A(t') = A_max * (1 - |t'|^n)^k`, where `t'` is time normalized from -1 to 1 across the pulse duration.
    -   The exponent `n` (parameter `power_n`) controls the flatness of the pulse's main body and the steepness of its transitions. Higher `n` values lead to a flatter top and sharper edges.
    -   The exponent `k` (parameter `phase_k`) controls the smoothness of the truncation at the beginning and end of the pulse. Higher `k` values provide smoother roll-offs.
-   **Frequency Modulation (FM):** WURST pulses typically employ a linear frequency sweep across a specified bandwidth. The instantaneous frequency is given by `f(t') = (bandwidth / 2) * t'`.

The combination of this specific AM and FM allows the WURST pulse to satisfy the adiabatic condition over a broad range, leading to its robust performance. The peak B1 amplitude (`A_max`) must be sufficient for the given sweep rate (`bandwidth / duration`) to maintain adiabaticity.

## Function: `generate_wurst_pulse`

The `generate_wurst_pulse` function in `mri_pulse_library.rf_pulses.adiabatic.wurst_pulse` implements the WURST pulse.

### Key Parameters:

*   `duration` (float): Total pulse duration in seconds.
*   `bandwidth` (float): Desired frequency sweep range in Hz (full width).
*   `flip_angle_deg` (float, optional): Target flip angle. Primarily used for B1 estimation if `peak_b1_tesla` is not provided and adiabatic estimation is not feasible. Default: 180.0.
*   `power_n` (float, optional): Exponent 'n' for AM. Default: 20.0.
*   `phase_k` (float, optional): Exponent 'k' for AM. Default: 1.0.
*   `adiabaticity_factor_Q` (float, optional): Factor for estimating peak B1 to satisfy the adiabatic condition (`B1_max ~ Q * sqrt(sweep_rate)`). Default: 5.0.
*   `peak_b1_tesla` (float, optional): If provided, sets the peak B1 amplitude in Tesla. If `None`, estimated using `adiabaticity_factor_Q`.
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio in Hz/T. Default: Proton.
*   `dt` (float, optional): Time step in seconds. Default: 1e-6 s.

### Returns:

*   `rf_pulse_tesla` (np.ndarray): Complex RF waveform in Tesla.
*   `time_vector_s` (np.ndarray): Time points in seconds.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.rf_pulses.adiabatic.wurst_pulse import generate_wurst_pulse

# Define WURST pulse parameters
total_duration = 0.008  # 8 ms
sweep_bandwidth = 6000  # 6 kHz
power_n_param = 20
phase_k_param = 1
# Let the function estimate peak_b1_tesla using adiabaticity_factor_Q
# Or provide one: peak_b1_val = 15e-6 # Tesla

rf_wurst, time_wurst = generate_wurst_pulse(
    duration=total_duration,
    bandwidth=sweep_bandwidth,
    power_n=power_n_param,
    phase_k=phase_k_param,
    adiabaticity_factor_Q=5.0, # Default is 5.0
    # peak_b1_tesla=peak_b1_val, # Uncomment to set B1max
    dt=1e-6  # 1 us time step
)

print(f"Generated WURST pulse with {len(rf_wurst)} samples.")
print(f"Estimated Peak B1: {np.max(np.abs(rf_wurst))*1e6:.2f} uT")

# Plotting (optional)
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
# Amplitude
axs[0].plot(time_wurst * 1000, np.abs(rf_wurst) * 1e6) # Amplitude in uT
axs[0].set_ylabel('Amplitude (µT)')
axs[0].set_title(f'WURST Pulse (n={power_n_param}, k={phase_k_param})')

# Phase
axs[1].plot(time_wurst * 1000, np.unwrap(np.angle(rf_wurst, deg=False))) # Phase in radians
axs[1].set_ylabel('Phase (rad)')

# Instantaneous Frequency (approximate)
freq_wurst_hz = np.diff(np.unwrap(np.angle(rf_wurst))) / (2 * np.pi * 1e-6)
axs[2].plot(time_wurst[:-1] * 1000, freq_wurst_hz / 1000) # Freq in kHz
axs[2].set_ylabel('Frequency (kHz)')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylim([-sweep_bandwidth/2000.0 * 1.1, sweep_bandwidth/2000.0 * 1.1])

plt.tight_layout()
plt.show()
```

## References

*   Kupce, E., & Freeman, R. (1995). *Adiabatic pulses for wideband inversion and refocusing.* Journal of Magnetic Resonance, Series A, 115(2), 273-276. (Original WURST paper)
*   Garwood, M., & DelaBarre, L. (2001). *The return of the frequency sweep: designing adiabatic pulses for contemporary NMR.* Journal of Magnetic Resonance, 153(2), 155-177. (Comprehensive review on adiabatic pulses).
