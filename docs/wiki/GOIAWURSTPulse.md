# GOIA-WURST Adiabatic Pulse

**GOIA-WURST (Gradient Offset Independent Adiabaticity - WURST)** pulses are a sophisticated type of adiabatic pulse that combines the robustness of WURST pulses with slice selectivity that is insensitive to variations in slice selection gradient strength and B0 offsets.

## Theory Overview

GOIA-WURST pulses build upon the WURST pulse design (smooth amplitude modulation and a base frequency sweep) by incorporating a specific, synchronized gradient waveform and an additional RF phase (or frequency) modulation.

Key characteristics include:
-   **WURST Amplitude Modulation (AM):** The RF amplitude envelope `A(t)` follows the standard WURST shape: `A(t') = A_max * (1 - |t'|^n)^k`.
-   **Synchronized Gradient Waveform `G(t)`:** The gradient waveform is typically designed to mirror the RF amplitude envelope, i.e., `G(t) = G_max * A(t) / A_max`. This synchronized variation is crucial for the GOIA properties.
-   **Combined Frequency Modulation (FM):** The total RF frequency modulation consists of two parts:
    1.  The underlying linear frequency sweep of the WURST pulse, covering a specified `bandwidth`.
    2.  A GOIA-specific frequency modulation component, `omega_GOIA(t)`. This component is often proportional to the RF amplitude envelope (`AM_normalized(t)`) and scaled by factors related to the `peak_gradient_mT_m` and `slice_thickness_m`. A common formulation is `omega_GOIA(t) = C_eff * AM_normalized(t)`, where `C_eff` (effective GOIA factor) is derived such that the frequency coverage due to this term matches the Larmor frequency range across the desired slice thickness under the peak gradient.
    `C_eff_rad_s = gamma_rad_s_t * peak_gradient_T_m * (slice_thickness_m / 2.0)`

This careful interplay between RF amplitude, RF phase/frequency, and the time-varying gradient ensures that spins within the target slice experience an effective adiabatic passage, largely independent of small variations in the gradient strength or static field offsets.

## Function: `generate_goia_wurst_pulse`

The `generate_goia_wurst_pulse` function in `mri_pulse_library.rf_pulses.adiabatic.goia_wurst_pulse` implements the GOIA-WURST pulse.

### Key Parameters:

*   `duration` (float): Total pulse duration in seconds.
*   `bandwidth` (float): Bandwidth of the underlying WURST frequency sweep in Hz.
*   `slice_thickness_m` (float): Target slice thickness in meters.
*   `peak_gradient_mT_m` (float): Peak amplitude of the selection gradient in mT/m.
*   `flip_angle_deg` (float, optional): Target flip angle. Default: 180.0.
*   `power_n` (float, optional): WURST AM exponent 'n'. Default: 20.0.
*   `phase_k` (float, optional): WURST AM exponent 'k'. Default: 1.0.
*   `goia_factor_C` (float, optional): Overrides the derived `C_eff` for the GOIA FM component if provided (units: rad/s).
*   `peak_b1_tesla` (float, optional): Peak B1 amplitude. If `None`, estimated like a WURST pulse based on `adiabaticity_factor_Q` and the WURST `bandwidth`.
*   `adiabaticity_factor_Q` (float, optional): Q factor for WURST B1 estimation. Default: 5.0.
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio in Hz/T. Default: Proton.
*   `dt` (float, optional): Time step in seconds. Default: 1e-6 s.

### Returns:

*   `rf_pulse_tesla` (np.ndarray): Complex RF waveform in Tesla.
*   `time_vector_s` (np.ndarray): Time points in seconds.
*   `gradient_waveform_mT_m` (np.ndarray): Gradient waveform in mT/m.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.rf_pulses.adiabatic.goia_wurst_pulse import generate_goia_wurst_pulse

# Define GOIA-WURST pulse parameters
total_duration = 0.010  # 10 ms
wurst_sweep_bandwidth = 4000  # 4 kHz (for the WURST component)
slice_thick_m = 0.005  # 5 mm
peak_grad_mT_m = 10.0   # 10 mT/m
# Let B1 peak be estimated

rf_goia, time_goia, grad_goia = generate_goia_wurst_pulse(
    duration=total_duration,
    bandwidth=wurst_sweep_bandwidth,
    slice_thickness_m=slice_thick_m,
    peak_gradient_mT_m=peak_grad_mT_m,
    dt=1e-6  # 1 us time step
)

print(f"Generated GOIA-WURST pulse with {len(rf_goia)} samples.")
print(f"Estimated Peak B1: {np.max(np.abs(rf_goia))*1e6:.2f} uT")
print(f"Peak Gradient: {np.max(np.abs(grad_goia)):.2f} mT/m")

# Plotting (optional)
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))

# RF Amplitude
axs[0].plot(time_goia * 1000, np.abs(rf_goia) * 1e6) # Amplitude in uT
axs[0].set_ylabel('RF Amp (µT)')
axs[0].set_title('GOIA-WURST Pulse')

# RF Phase
axs[1].plot(time_goia * 1000, np.unwrap(np.angle(rf_goia, deg=False))) # Phase in radians
axs[1].set_ylabel('RF Phase (rad)')

# Gradient Waveform
axs[2].plot(time_goia * 1000, grad_goia) # Gradient in mT/m
axs[2].set_ylabel('Gradient (mT/m)')

# Instantaneous Frequency (approximate)
freq_goia_hz = np.diff(np.unwrap(np.angle(rf_goia))) / (2 * np.pi * 1e-6)
axs[3].plot(time_goia[:-1] * 1000, freq_goia_hz / 1000) # Freq in kHz
axs[3].set_ylabel('RF Freq (kHz)')
axs[3].set_xlabel('Time (ms)')

plt.tight_layout()
plt.show()
```

## References

*   Jiru, F., & Klose, U. (2006). *Gradient offset independent adiabaticity for slice-selective adiabatic RF pulses.* Journal of Magnetic Resonance, 180(1), 50-58.
*   Andronesi, O. C., Jiru, F., Klose, U., & Natt, O. (2008). *Slice-selective adiabatic radiofrequency pulses for in vivo applications.* Journal of Magnetic Resonance, 193(2), 207-216.
*   Garwood, M., & DelaBarre, L. (2001). *The return of the frequency sweep: designing adiabatic pulses for contemporary NMR.* Journal of Magnetic Resonance, 153(2), 155-177.
