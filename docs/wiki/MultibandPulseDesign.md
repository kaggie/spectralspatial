# Simultaneous Multiband Pulse Design

The `MultibandPulseDesigner` class provides tools for creating simultaneous multiband RF pulses. This is typically achieved by taking a single-band (baseband) RF pulse and modulating it to excite multiple frequency bands (and thus multiple spatial slices when a gradient is active) at the same time.

This approach is commonly used in techniques like PINS (Power Independent Number of Slices) to accelerate imaging.

## `MultibandPulseDesigner` Class

The primary tool is the `MultibandPulseDesigner` located in `mri_pulse_library.rf_pulses.multiband.designer`.

### Initialization

```python
from mri_pulse_library.rf_pulses.multiband.designer import MultibandPulseDesigner

mb_designer = MultibandPulseDesigner(verbose=True)
```
-   `verbose` (bool, optional): If `True`, prints design information and warnings.

### Method: `design_simultaneous_multiband_pulse`

This method generates the multiband RF pulse.

#### Key Parameters:

*   `base_pulse_rf` (np.ndarray): Complex array of the baseband RF pulse waveform (units: Tesla). This pulse is typically designed first (e.g., using a sinc pulse generator or the `SpectralSpatialPulseDesigner`) and is assumed to be scaled for its desired single-band flip angle.
*   `base_pulse_dt_s` (float): Time step (sampling interval) of the `base_pulse_rf` in seconds.
*   `num_bands` (int): The number of simultaneous bands to generate.
*   `band_offsets_hz` (list of float): A list of frequency offsets in Hz for each band, relative to the carrier frequency of the `base_pulse_rf`. The length of this list must equal `num_bands`.
*   `base_pulse_gradient` (np.ndarray, optional): The gradient waveform (e.g., mT/m) associated with the `base_pulse_rf`. For phase-encoded multiband techniques, this gradient is typically reused directly.
*   `band_phases_deg` (list of float, optional): A list of phases in degrees to apply to each individual band. If `None`, all bands default to 0 degrees phase. Useful for phase optimization strategies (though advanced optimization is not part of the basic superposition).
*   `max_b1_tesla_combined` (float, optional): If provided, the peak B1 amplitude of the final combined multiband pulse will be scaled down to not exceed this value. If `None`, no such scaling is applied, and the peak B1 can be up to `num_bands` times the peak of the `base_pulse_rf` (in the worst-case constructive interference). A warning is issued if scaling occurs, as it proportionally affects the flip angle of all bands.

#### Returns:

*   `multiband_rf_pulse_tesla` (np.ndarray): The combined complex multiband RF waveform (Tesla).
*   `time_vector_s` (np.ndarray): Time vector for the multiband pulse (seconds).
*   `multiband_gradient` (np.ndarray or None): The gradient waveform, typically the same as `base_pulse_gradient`.

## Example Usage

First, design a baseband pulse (e.g., a slice-selective sinc pulse). Then, use `MultibandPulseDesigner` to create a multiband version.

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.rf_pulses.simple.sinc_pulse import generate_sinc_pulse
from mri_pulse_library.rf_pulses.multiband.designer import MultibandPulseDesigner
# Assuming you might have a gradient defined for the SPSP pulse or a simple one for slice selection context
# For this example, we'll focus on the RF. Gradient would be passed if available.

# 1. Define baseband pulse parameters (e.g., a sinc pulse for slice selection)
dt_s = 4e-6  # 4 us time step
sinc_duration_s = 0.003  # 3 ms
sinc_flip_angle_deg = 15 # Low flip for base, as it will be summed
sinc_time_bw_product = 4

base_rf, base_time, _ = generate_sinc_pulse(
    flip_angle_deg=sinc_flip_angle_deg,
    duration_s=sinc_duration_s,
    time_bw_product=sinc_time_bw_product,
    dt_s=dt_s
)
base_rf = base_rf.astype(np.complex128) # Ensure complex type

# (Optional) Define a base gradient if you have one (e.g., from an SPSP design)
# For a simple sinc, a constant slice-select gradient would be active during the pulse.
# base_gradient_example = np.ones_like(base_rf_real) * 10 # Example: 10 mT/m
# For this RF-focused example, we'll pass gradient as None.

# 2. Define multiband parameters
num_slices = 3
# E.g., for a 5mm slice with Gz=10mT/m, slice BW = gamma_Hz/T * Gz * thk
# Example: 42.577e6 Hz/T * 0.01 T/m * 0.005m = 2128 Hz slice BW
# If base sinc has TBW=4, its BW is 4/3ms = 1333 Hz.
# Offsets should be larger than half the base pulse BW to separate slices.
# Let's aim for offsets that clearly separate bands in frequency.
# If base pulse BW is ~1.3 kHz, offsets of +/- 2.5 kHz should work.
frequency_offsets_hz = [-2500, 0, 2500]  # For a 3-band pulse
band_custom_phases_deg = [0, 30, 60]    # Example custom phases per band

# (Optional) Define a maximum allowed combined B1 peak
# Peak of our 15-deg, 3ms, TBW4 sinc is ~0.32uT. 3 bands could reach ~1uT.
max_peak_b1_uT_combined = 0.8  # uT
max_peak_b1_T_combined = max_peak_b1_uT_combined * 1e-6

# 3. Initialize the designer and create the multiband pulse
mb_designer = MultibandPulseDesigner(verbose=True)
mb_rf, mb_time, mb_grad = mb_designer.design_simultaneous_multiband_pulse(
    base_pulse_rf=base_rf,
    base_pulse_dt_s=dt_s,
    num_bands=num_slices,
    band_offsets_hz=frequency_offsets_hz,
    band_phases_deg=band_custom_phases_deg,
    # base_pulse_gradient=base_gradient_example, # Pass if you have one
    max_b1_tesla_combined=max_peak_b1_T_combined
)

print(f"Generated {num_slices}-band pulse with {len(mb_rf)} samples.")
print(f"Final combined peak B1: {np.max(np.abs(mb_rf))*1e6:.2f} uT")

# 4. Plotting (optional)
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,7))
axs[0].plot(mb_time * 1000, np.abs(mb_rf) * 1e6)
axs[0].set_ylabel('Amplitude (µT)')
axs[0].set_title(f'{num_slices}-Band RF Pulse')

axs[1].plot(mb_time * 1000, np.angle(mb_rf, deg=True))
axs[1].set_ylabel('Phase (degrees)')

# 간단한 스펙트럼 분석 (Simple spectrum analysis)
from scipy.fft import fft, fftfreq, fftshift
spectrum = fftshift(fft(mb_rf))
freqs = fftshift(fftfreq(len(mb_rf), d=dt_s))
axs[2].plot(freqs / 1000, np.abs(spectrum))
axs[2].set_xlabel('Frequency (kHz)')
axs[2].set_ylabel('Spectrum Magnitude')
axs[2].grid(True)

plt.tight_layout()
plt.show()
```

## Further Considerations

-   **Peak B1 and SAR:** Superimposing multiple RF pulses significantly increases the peak B1 amplitude and SAR. The `max_b1_tesla_combined` parameter provides a basic mechanism to cap the peak B1 by scaling the entire waveform. More advanced methods involve optimizing the relative phases of the bands (e.g., PINS) or using techniques like VERSE-MB to manage RF power and gradient demands, which are potential future enhancements.
-   **Base Pulse Choice:** The quality of the multiband pulse heavily depends on the characteristics of the `base_pulse_rf`. A well-designed slice-selective base pulse is crucial.
-   **Gradient Waveforms:** For standard phase-encoded simultaneous multislice, the gradient waveform of the baseband pulse is typically reused. If the baseband pulse is from an SPSP design, its specific gradient (often complex for SPSP) should be passed.
