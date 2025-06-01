# BIR-4 Adiabatic Pulse

The **BIR-4 (B1-Insensitive Rotation) pulse** is an adiabatic pulse renowned for its ability to achieve accurate flip angles (typically inversion or excitation) over a wide range of B1 RF field strengths. This makes it highly robust to B1 inhomogeneities.

## Theory Overview

BIR-4 pulses achieve their B1 insensitivity through a sophisticated design typically involving four distinct segments. Each segment is often based on adiabatic principles, such as hyperbolic secant (HS) pulses for amplitude and frequency modulation. The critical aspect of BIR-4 is the precise phase relationship (phase cycling) between these four segments.

Commonly, the pulse structure is designed to:
1.  Move magnetization from the longitudinal axis to the transverse plane.
2.  Rotate magnetization within the transverse plane.
3.  Return magnetization from the transverse plane to the longitudinal axis (often inverted).

The specific phase shifts between segments, often controlled by parameters like `kappa` and `xi` (or `zeta`), are key to its performance. The pulse is designed such that the desired final state (e.g., perfect inversion) is reached regardless of the exact B1 amplitude, provided it is above a certain minimum threshold to maintain adiabaticity.

## Function: `generate_bir4_pulse`

The `generate_bir4_pulse` function in `mri_pulse_library.rf_pulses.adiabatic.bir4_pulse` provides a detailed implementation of a BIR-4 pulse.

### Key Parameters:

*   `duration` (float): Total pulse duration in seconds.
*   `bandwidth` (float): Desired frequency sweep range in Hz (full width of the adiabatic sweep).
*   `flip_angle_deg` (float, optional): Target flip angle, typically 180.0 for inversion. While BIR-4 is adiabatic, this can influence B1max estimation if not provided.
*   `beta_bir4` (float, optional): Dimensionless parameter controlling the shape (steepness) of the hyperbolic secant components. Default: 10.0.
*   `mu_bir4` (float, optional): Adiabaticity factor for the hyperbolic secant frequency sweep. Default: 4.9.
*   `kappa_deg` (float, optional): Phase parameter kappa in degrees, used in inter-segment phase calculations (e.g., via `tan(kappa_rad)`). Default: 70.0.
*   `xi_deg` (float, optional): Additional phase parameter xi in degrees, for phase cycling between segments. Default: 90.0.
*   `peak_b1_tesla` (float, optional): If provided, sets the peak B1 amplitude in Tesla. If `None`, it's estimated (e.g., a heuristic for 180-deg inversion, or integral scaling for other flip angles, which is less optimal).
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio in Hz/T. Default: Proton.
*   `dt` (float, optional): Time step in seconds. Default: 1e-6 s.

### Returns:

*   `rf_pulse_tesla` (np.ndarray): Complex RF waveform in Tesla.
*   `time_vector_s` (np.ndarray): Time points in seconds.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.rf_pulses.adiabatic.bir4_pulse import generate_bir4_pulse
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

# Define BIR-4 pulse parameters
total_duration = 0.010  # 10 ms
sweep_bandwidth = 4000  # 4 kHz
target_flip_angle = 180.0 # degrees
kappa_parameter_deg = 70.0
xi_parameter_deg = 90.0
# Let the function estimate peak_b1_tesla for 180 deg inversion
# Or provide one: peak_b1_val = 13e-6 # Tesla

rf_bir4, time_bir4 = generate_bir4_pulse(
    duration=total_duration,
    bandwidth=sweep_bandwidth,
    flip_angle_deg=target_flip_angle,
    kappa_deg=kappa_parameter_deg,
    xi_deg=xi_parameter_deg,
    # peak_b1_tesla=peak_b1_val, # Uncomment to set B1max
    dt=1e-6  # 1 us time step
)

print(f"Generated BIR-4 pulse with {len(rf_bir4)} samples.")
print(f"Estimated Peak B1: {np.max(np.abs(rf_bir4))*1e6:.2f} uT")


# Plotting (optional)
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(time_bir4 * 1000, np.abs(rf_bir4) * 1e6) # Amplitude in uT
axs[0].set_ylabel('Amplitude (µT)')
axs[0].set_title('BIR-4 Pulse')

axs[1].plot(time_bir4 * 1000, np.angle(rf_bir4, deg=True)) # Phase in degrees
axs[1].set_ylabel('Phase (degrees)')
axs[1].set_xlabel('Time (ms)')
plt.show()
```

## References

For more detailed information on BIR-4 pulses, consider these classic papers:
*   Staewen, R. S., Johnson, A. J., Ross, B. D., Parrish, T., Merkle, H., & Garwood, M. (1990). *A B1-insensitive, single-shot localization and water suppression sequence for in vivo 1H NMR spectroscopy.* Journal of Magnetic Resonance, 89(3), 598-605. (Introduced BIR-1, precursor concepts)
*   Garwood, M., & DelaBarre, L. (2001). *The return of the frequency sweep: designing adiabatic pulses for contemporary NMR.* Journal of Magnetic Resonance, 153(2), 155-177. (Excellent review on adiabatic pulses including BIR-4).
*   De Graaf, R. A. (2007). *In Vivo NMR Spectroscopy: Principles and Techniques.* John Wiley & Sons. (Textbook with good explanations).

(Note: Specific BIR-4 design details can vary slightly between implementations based on different literature sources or optimization goals.)
