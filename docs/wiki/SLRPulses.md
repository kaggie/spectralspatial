# SLR (Shinnar-LeRoux) Pulses

The Shinnar-LeRoux (SLR) algorithm is a powerful method for designing RF pulses with well-defined frequency selectivity, commonly used for slice selection, fat/water suppression, or other spectral selection tasks. SLR pulses allow for precise control over passband and stopband ripple characteristics.

This library provides the `SLRTransform` class to facilitate the design of SLR pulses. The design process typically involves three main stages:

1.  **Magnetization to B-Polynomial Specifications**: Defining the desired magnetization response (e.g., Mxy for excitation, Mz for inversion/saturation) in terms of passband/stopband amplitudes and ripples, and converting these into specifications for the B-polynomial (Beta(z) in SLR theory).
2.  **FIR Filter Design for B-Polynomial**: Designing an FIR filter that meets the B-polynomial specifications. The coefficients of this filter are the B-polynomial coefficients.
3.  **B-Polynomial to RF Pulse (Inverse SLR Transform)**: Converting the B-polynomial coefficients into the actual RF pulse waveform using the inverse SLR transform, which involves spectral factorization to find the A-polynomial (Alpha(z)).

## Using `SLRTransform`

The `SLRTransform` class provides methods for steps 1 and 3. Step 2 (FIR filter design) is typically handled by an FIR filter design tool, such as `FIRFilterDesigner` also available in this library or Scipy's signal processing tools.

### 1. `magnetization_to_b_poly_specs`

This method converts your desired magnetization profile into B-polynomial specifications.

*   **Key Parameters**:
    *   `desired_mag_ripples` (List[float]): Ripples in the target magnetization profile (e.g., `[d_pass, d_stop]`).
    *   `desired_mag_amplitudes` (List[float]): Amplitudes in the target magnetization profile (e.g., `[1, 0]` for Mxy excitation).
    *   `nominal_flip_angle_rad` (float): Overall target flip angle.
    *   `pulse_type` (str): 'ex', 'se', 'inv', 'sat'.
*   **Returns**: `b_poly_ripples`, `b_poly_amplitudes` (for B-poly filter design, normalized so max amplitude is 1), `adjusted_flip_angle_rad`.

### 2. FIR Filter Design for B-Polynomial (using `FIRFilterDesigner`)

The `b_poly_amplitudes` (normalized target gains for B(z)) and `b_poly_ripples` (can inform weights) from the previous step are used to design the B-polynomial FIR filter coefficients. For example, using `FIRFilterDesigner.design_parks_mcclellan_real`:

*   **Key Parameters for FIR designer**:
    *   `num_taps`: Number of coefficients for the B-polynomial.
    *   `bands_normalized`: Frequency band edges (normalized to Nyquist=1.0) for B(z).
    *   `desired_amplitudes`: Desired gains for B(z) in each band (e.g., `[1, 0]` for a lowpass B(z) if `max(b_poly_amplitudes)` from step 1 was 1).
    *   `weights`: Weights for each band, can be derived from `b_poly_ripples`.
*   **Returns**: `b_poly_coeffs` (torch.Tensor).

**Important**: Before passing `b_poly_coeffs` to the next step, ensure `max|B(e^jω)| <= 1`. This usually involves calculating the frequency response of the designed `b_poly_coeffs`, finding `max_abs_B_omega`, and if it's `> 1`, normalizing `b_poly_coeffs /= max_abs_B_omega`.

### 3. `b_poly_to_rf`

This method performs the inverse SLR transform.

*   **Key Parameters**:
    *   `b_poly_coeffs` (torch.Tensor): The (normalized) B-polynomial coefficients from FIR design.
    *   `pulse_type` (str, optional): 'ex' is currently supported.
*   **Returns**: Complex RF pulse waveform (torch.Tensor, unitless shape, same length as `b_poly_coeffs`).

### High-Level Wrapper: `design_rf_pulse_from_mag_specs`

To simplify the workflow, `SLRTransform` also provides a wrapper method:

*   **Key Parameters**: Combines inputs for `magnetization_to_b_poly_specs` and direct FIR design parameters for the B-polynomial (e.g., `num_taps_b_poly`, `fir_bands_normalized`, `fir_desired_b_gains`, `fir_weights`).
*   **Internal Steps**:
    1.  Calls `magnetization_to_b_poly_specs`.
    2.  Calls `FIRFilterDesigner.design_parks_mcclellan_real` using the provided FIR parameters.
    3.  Normalizes the resulting `b_poly_coeffs` so `max|B(e^jω)| <= 1`.
    4.  Calls `b_poly_to_rf`.
*   **Returns**: `rf_pulse` (torch.Tensor), `adjusted_flip_angle_rad` (float).

### Example: Designing a Slice-Selective Excitation Pulse

```python
from mri_pulse_library.slr_transform import SLRTransform
# Assuming fir_designer.py is in the path for FIRFilterDesigner
from fir_designer import FIRFilterDesigner
import torch
import math

# --- Instantiate SLR Transformer ---
slr_designer = SLRTransform()

# --- 1. Define Magnetization Specifications ---
mag_ripples = [0.01, 0.01]  # Passband Mxy ripple, Stopband Mxy ripple
mag_amplitudes = [1.0, 0.0] # Mxy=1 in passband, Mxy=0 in stopband
nominal_flip = math.pi / 2.0 # 90 degrees
pulse_type = 'ex'
num_taps_b = 65 # Length of B-polynomial

# --- 2. Define FIR Design Parameters for B(z) ---
# For an excitation pulse, B(z) is typically a lowpass filter.
# Its passband gain should correspond to the (normalized) passband b_poly_amplitude
# derived from mag_specs, and stopband gain to stopband b_poly_amplitude.
# The design_rf_pulse_from_mag_specs wrapper takes these directly.
fir_bands = [0.0, 0.2, 0.25, 1.0] # Normalized to Nyquist for FIR designer (0 to 1.0)
                                 # E.g., passband 0-0.2, stopband 0.25-1.0
fir_b_gains = [1.0, 0.0]         # Design B(z) to be a lowpass filter with gain 1 in passband
fir_b_weights = [1.0, 1.0]       # Equal weights for passband and stopband ripple in B(z)

# --- 3. Design RF Pulse using the wrapper ---
try:
    rf_pulse, adjusted_flip = slr_designer.design_rf_pulse_from_mag_specs(
        desired_mag_ripples=mag_ripples,
        desired_mag_amplitudes=mag_amplitudes,
        nominal_flip_angle_rad=nominal_flip,
        pulse_type=pulse_type,
        num_taps_b_poly=num_taps_b,
        fir_bands_normalized=fir_bands,
        fir_desired_b_gains=fir_b_gains,
        fir_weights=fir_b_weights,
        device='cpu'
    )

    print(f"Designed SLR RF pulse. Length: {len(rf_pulse)}")
    print(f"Adjusted flip angle: {adjusted_flip:.3f} radians")
    print(f"RF pulse (first 5 points): {rf_pulse[:5]}")

    # This rf_pulse is a unitless shape. It needs to be scaled to physical B1 units
    # and assigned a dt for use in simulation or on a scanner.
    # For simulation, one might scale its peak or integral.

except ImportError:
    print("FIRFilterDesigner could not be imported. Please ensure fir_designer.py is accessible.")
except Exception as e:
    print(f"An error occurred during SLR pulse design example: {e}")

```

### Notes on Output

*   The RF pulse generated by `b_poly_to_rf` or `design_rf_pulse_from_mag_specs` is a sequence of complex numbers representing the (unitless) shape of the RF pulse.
*   To use this pulse in a simulation or on a scanner, it needs to be scaled to physical B1 units (e.g., Tesla or Gauss) and associated with a time step (`dt`) per point, which determines its total duration. The `adjusted_flip_angle_rad` helps in this scaling.
