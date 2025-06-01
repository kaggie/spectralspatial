# SLR (Shinnar-LeRoux) Pulses

The Shinnar-LeRoux (SLR) algorithm is a powerful and widely used method for designing RF pulses with precisely controlled frequency selectivity (slice profiles). It's particularly known for creating linear-phase pulses with excellent passband ripple, stopband attenuation, and transition width characteristics.

This library provides the `SLRTransform` class in `mri_pulse_library.slr_transform` to facilitate the design of Small-Tip Angle (STA) SLR pulses and to perform Large-Tip Angle (LTA) correction for excitation pulses.

## SLR Design Workflow (Small-Tip Angle - STA)

The STA SLR design process generally involves these stages:

1.  **Define Target Magnetization Profile**: Specify the desired magnetization response (e.g., Mxy for excitation, Mz for refocusing/inversion) across different frequency bands, including desired amplitudes and acceptable ripples.
2.  **Convert to B-Polynomial Specifications**: Use `SLRTransform.magnetization_to_b_poly_specs()` to translate the magnetization targets into specifications (target amplitudes and ripples) for the SLR B-polynomial (Beta(z)). This step also considers the nominal flip angle and pulse type.
3.  **Design B-Polynomial FIR Filter**: Design a Finite Impulse Response (FIR) filter whose frequency response matches the B-polynomial specifications. The coefficients of this filter are the `b_poly_coeffs`. This library's `FIRFilterDesigner` (e.g., `design_parks_mcclellan_real`) can be used for this.
    *   **Important**: After FIR design, the resulting B-polynomial's frequency response `B(omega)` must be normalized such that `max|B(omega)| <= 1.0`. This is crucial for the stability of the inverse SLR transform.
4.  **Inverse SLR Transform (B-Poly to RF)**: Use `SLRTransform.b_poly_to_rf()` to convert the (normalized) `b_poly_coeffs` into the complex RF pulse waveform. This step involves spectral factorization to find the A-polynomial (Alpha(z)) and then combines A(z) and B(z) based on the pulse type.

## `SLRTransform` Class Methods

### `magnetization_to_b_poly_specs`
Converts desired magnetization specs to B-polynomial specs.
*   **Key Args**: `desired_mag_ripples`, `desired_mag_amplitudes`, `nominal_flip_angle_rad`, `pulse_type` ('ex', 'se', 'inv', 'sat').
*   **Returns**: `b_poly_ripples`, `b_poly_amplitudes` (targets for B-poly FIR design, normalized so max B-poly amplitude is 1), `adjusted_flip_angle_rad`.

### `b_poly_to_rf`
Performs the inverse SLR transform (B-polynomial to RF pulse).
*   **Key Args**: `b_poly_coeffs` (normalized), `pulse_type` ('ex', 'se', 'inv').
    *   **'ex' (Excitation)**: Uses RF(z) = B(z) / A*(1/z*)
    *   **'se' (Refocusing)**: Uses RF(z) = B(z) / A(z) (typically for linear-phase B(z))
    *   **'inv' (Inversion)**: Uses RF(z) = B(z) / A*(1/z*) (like 'ex', B(z) is designed from Mz=-1 target)
*   **Returns**: Complex RF pulse waveform (torch.Tensor, unitless shape).

### `design_rf_pulse_from_mag_specs` (High-Level Wrapper)
Orchestrates the full STA SLR design process from magnetization specs to RF pulse.
*   **Key Args**: Combines inputs for `magnetization_to_b_poly_specs` and FIR design parameters for B(z) (`num_taps_b_poly`, `fir_bands_normalized`, `fir_desired_b_gains`, `fir_weights`).
*   **Internal Steps**: Calls `magnetization_to_b_poly_specs`, designs FIR B-poly, normalizes B-poly, calls `b_poly_to_rf`.
*   **Returns**: `rf_pulse` (torch.Tensor), `adjusted_flip_angle_rad` (float).

### Example: STA Slice-Selective Excitation (90-degree)

```python
import torch
import math
import matplotlib.pyplot as plt
from mri_pulse_library.slr_transform import SLRTransform
from fir_designer import FIRFilterDesigner # Assuming fir_designer.py is accessible

# --- Parameters ---
slr_designer = SLRTransform(verbose=True)
device = 'cpu'

# Magnetization specs for Mxy
mag_ripples_ex = [0.01, 0.01]  # Passband, Stopband Mxy ripple
mag_amplitudes_ex = [1.0, 0.0] # Mxy=1 in passband, Mxy=0 in stopband
nominal_flip_ex = math.pi / 2.0 # 90 degrees
pulse_type_ex = 'ex'

# B-polynomial FIR filter design specs
num_taps_b = 65
# For 'ex' and Mxy=[1,0], b_poly_specs_amplitudes will be [~1.414, 0] before normalization,
# then [1.0, 0.0] after normalization by SLRTransform.
# So, the FIR B-poly should target [1.0, 0.0] for its gains.
fir_bands_norm = [0.0, 0.1, 0.15, 0.5] # Normalized to Nyquist (0.0 to 0.5 for real filters)
fir_gains_b = [1.0, 0.0] # Target gains for B(omega)
fir_w = [1.0, 10.0]      # Weight stopband more for B(omega)

# --- Design Pulse using Wrapper ---
rf_ex, adj_flip_ex = slr_designer.design_rf_pulse_from_mag_specs(
    desired_mag_ripples=mag_ripples_ex,
    desired_mag_amplitudes=mag_amplitudes_ex,
    nominal_flip_angle_rad=nominal_flip_ex,
    pulse_type=pulse_type_ex,
    num_taps_b_poly=num_taps_b,
    fir_bands_normalized=fir_bands_norm,
    fir_desired_b_gains=fir_gains_b,
    fir_weights=fir_w,
    device=device
)
print(f"STA Excitation Pulse: Length={len(rf_ex)}, Adj. Flip={adj_flip_ex:.3f} rad")
# This rf_ex is a unitless shape. Scale for B1 amplitude and assign dt for use.
# Example scaling: Scale peak to 0.1 Gauss
# rf_ex_gauss = rf_ex * (0.1 / torch.max(torch.abs(rf_ex))) if torch.max(torch.abs(rf_ex)) > 0 else rf_ex

# Plot (optional)
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(torch.abs(rf_ex).cpu().numpy(), label='Magnitude')
    plt.plot(torch.angle(rf_ex).cpu().numpy(), label='Phase (rad)')
    plt.title(f'STA SLR Excitation Pulse (Adj. Flip ~{np.rad2deg(adj_flip_ex):.1f} deg)')
    plt.legend()
    plt.show()
except ImportError:
    print("Matplotlib not found, skipping example plot for STA pulse.")
```

## Large-Tip Angle (LTA) Correction for SLR Pulses

The STA SLR design is accurate for small flip angles. For large flips (e.g., 90-deg excitation, 180-deg refocusing/inversion), the actual on-resonance flip angle can deviate from the target. The `iterative_lta_rf_scale` method provides a way to correct an initial (STA) RF pulse to achieve a more accurate on-resonance large flip angle.

### `iterative_lta_rf_scale`
Iteratively scales an RF pulse to meet a target on-resonance LTA.
*   **Key Args**:
    *   `initial_rf_gauss` (torch.Tensor): Initial complex RF waveform (Gauss).
    *   `target_flip_angle_rad` (float): Desired on-resonance flip angle.
    *   `gradient_waveform_gcms` (torch.Tensor): Slice-select gradient (G/cm) active during RF.
    *   `dt_s` (float): Time step of RF/gradient.
    *   `gyromagnetic_ratio_hz_g` (float, optional): Hz/Gauss.
    *   `num_iterations` (int, optional): Max iterations.
    *   `b0_offset_hz` (float, optional): B0 offset for simulation (0 for on-resonance).
    *   `target_tolerance_rad` (float, optional): Early stopping tolerance.
    *   `max_b1_amplitude_gauss` (float, optional): Peak B1 clipping limit (Gauss).
*   **Internal Steps**:
    1.  Simulates current RF with `bloch_simulate` at slice center to get achieved flip angle.
    2.  Scales RF by `target_flip / achieved_flip`.
    3.  Optionally clips RF to `max_b1_amplitude_gauss`.
    4.  Repeats.
*   **Returns**: `scaled_rf_gauss` (torch.Tensor), `final_achieved_flip_angle_rad` (float).

### Example: LTA Correction of a 90-degree Excitation Pulse

```python
# (Continuing from previous STA excitation example - rf_ex, adj_flip_ex obtained)
# Assume rf_ex is the unitless shape, and adj_flip_ex is its STA-predicted flip.
import numpy as np # For np.rad2deg

# --- Parameters for LTA correction & simulation ---
dt_pulse = 4e-6  # Example: 4 us time step for the pulse
# Scale initial RF to a nominal peak B1 (e.g., 0.1 G) for LTA correction
# This scaling depends on how RF units relate to physical units.
# Here, assume rf_ex is a shape and we give it physical scale.
if torch.max(torch.abs(rf_ex)) > 1e-9:
    rf_sta_physical_gauss = rf_ex * (0.1 / torch.max(torch.abs(rf_ex)))
else:
    rf_sta_physical_gauss = rf_ex # Avoid division by zero if pulse is zero

# Define a simple slice-select gradient (constant during RF)
# Its strength relative to RF bandwidth determines slice thickness.
# For this example, just needs to be same length as RF.
grad_shape = torch.ones(len(rf_sta_physical_gauss), device=device)
slice_select_gradient_g_cm = grad_shape * 0.1 # Example: 0.1 G/cm

target_lta_flip = math.pi / 2.0 # 90 degrees

# --- Perform LTA Correction ---
try:
    rf_lta_corrected, achieved_lta_flip = slr_designer.iterative_lta_rf_scale(
        initial_rf_gauss=rf_sta_physical_gauss,
        target_flip_angle_rad=target_lta_flip,
        gradient_waveform_gcms=slice_select_gradient_g_cm,
        dt_s=dt_pulse,
        num_iterations=15,
        target_tolerance_rad=0.005, # ~0.3 degrees
        max_b1_amplitude_gauss=0.15, # Optional: clip if peak B1 gets too high
        device=device
    )
    print(f"LTA Corrected Pulse: Length={len(rf_lta_corrected)}, Achieved LTA Flip={np.rad2deg(achieved_lta_flip):.2f} deg")

    # Plot LTA corrected pulse (optional)
    plt.figure()
    plt.plot(torch.abs(rf_lta_corrected).cpu().numpy(), label='LTA Magnitude (G)')
    plt.plot(torch.angle(rf_lta_corrected).cpu().numpy(), label='LTA Phase (rad)')
    plt.title(f'LTA Corrected SLR Excitation (Target ~90deg, Achieved ~{np.rad2deg(achieved_lta_flip):.1f}deg)')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An error occurred during LTA correction example: {e}")
    import traceback
    traceback.print_exc()
```

## Designing Refocusing and Inversion Pulses (STA)

The workflow for designing STA refocusing (`'se'`) or inversion (`'inv'`) pulses is similar to excitation:
1.  Use `magnetization_to_b_poly_specs` with `pulse_type='se'` or `'inv'`, and appropriate `desired_mag_amplitudes` for Mz (e.g., `[-1.0, 1.0]` for Mz_passband=-1, Mz_stopband=1 for a 180-degree inversion/refocusing pulse). The `nominal_flip_angle_rad` would typically be `math.pi`.
2.  Design the `b_poly_coeffs` using an FIR filter designer. For refocusing pulses, `b_poly_coeffs` should be symmetric (linear phase).
3.  Normalize `b_poly_coeffs` so `max|B(omega)| <= 1`.
4.  Call `b_poly_to_rf` with the correct `pulse_type`.
    *   For `'se'`: Uses `RF(z) = B(z) / A(z)`.
    *   For `'inv'`: Uses `RF(z) = B(z) / A*(1/z*)`.

LTA correction for refocusing/inversion pulses using `iterative_lta_rf_scale` would require careful consideration of the target magnetization state (e.g., pure Mz inversion, or specific transverse phase for refocusing) in the Bloch simulation step. The current `iterative_lta_rf_scale` is primarily set up for excitation flip angle.

## References
*   Pauly, J., Le Roux, P., Nishimura, D., & Macovski, A. (1991). *Parameter relations for the Shinnar-Le Roux selective excitation pulse design algorithm.* IEEE Transactions on Medical Imaging, 10(1), 53-65.
*   Shinnar, M., & Le Roux, S. (1988). *The design of Shinnar-Le Roux selective pulses for in vivo NMR.* Magnetic Resonance in Medicine, 7(1), 101-105. (Conceptual, actual algorithm in Pauly's work)
*   De Graaf, R. A. (2007). *In Vivo NMR Spectroscopy: Principles and Techniques.* John Wiley & Sons.
