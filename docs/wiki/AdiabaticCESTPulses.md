# Adiabatic Pulses & CEST Simulation

This section covers the generation of adiabatic pulse waveforms and the simulation of their effects in multi-pool systems, particularly relevant for Chemical Exchange Saturation Transfer (CEST) and T1rho imaging.

## Adiabatic Pulse Waveforms

Adiabatic pulses are designed to be robust to B1 RF field inhomogeneities, producing uniform flip angles over a wide range of B1 strengths. This library provides functions to generate common adiabatic pulse types.

### 1. Hyperbolic Secant (HS) Pulses

Generated using `generate_hs_pulse_waveforms`.
-   **Modulation**: `A(t) = A_max * sech(beta * tau)` and `f(t) = (bandwidth_hz / 2) * tanh(beta * tau)`.
-   **Key Parameters**: `pulse_duration_s`, `num_timepoints`, `peak_b1_tesla`, `bandwidth_hz`, `beta`.
-   **Output**: `time_s`, `amplitude_tesla` (real), `freq_offset_hz` (relative to center).

### 2. BIR-4 (B1-Insensitive Rotation) Pulses

Generated using `generate_bir4_waveforms`.
-   **Construction**: Typically from four phase-cycled Hyperbolic Secant (HS) full-passage segments.
-   **Key Parameters**: `segment_duration_s`, `num_timepoints_segment`, `peak_b1_tesla_segment`, `bandwidth_hz_segment`, `beta_segment`, `segment_phases_rad` (list of 4 phases, e.g., `[0, torch.pi, 0, torch.pi]`).
-   **Output**: `time_s`, `amplitude_tesla_complex` (includes segment phases), `freq_offset_hz` (concatenated from HS segments).

### 3. WURST (Wideband, Uniform Rate, Smooth Truncation) Pulses

Generated using `generate_wurst_waveforms`.
-   **Modulation**: `A(t) = A_max * (1 - |tau|^n)` and `f(t) = (bandwidth_hz / 2) * tau`.
-   **Key Parameters**: `pulse_duration_s`, `num_timepoints`, `peak_b1_tesla`, `bandwidth_hz`, `power_n`.
-   **Output**: `time_s`, `amplitude_tesla` (real), `freq_offset_hz`.

### Example: Generating an HS Pulse

```python
from mri_pulse_library.rf_pulses.adiabatic import generate_hs_pulse_waveforms
import torch

time_vec, amp_vec, freq_vec = generate_hs_pulse_waveforms(
    pulse_duration_s=10e-3,
    num_timepoints=1024,
    peak_b1_tesla=15e-6, # 15 uT
    bandwidth_hz=4000,   # 4 kHz
    beta=5.0,
    device='cpu'
)
print(f"HS Pulse: {len(time_vec)} points, Peak Amp: {torch.max(amp_vec).item():.2e} T")
```

## Multi-Pool CEST/T1rho Simulation

The `AdiabaticCESTSimulator` class allows simulating the response of multi-pool systems (e.g., water and CEST agents, or components in T1rho) to these (or other) RF pulses. It uses a Bloch-McConnell solver (`bloch_mcconnell_step`).

### Key Initialization Parameters (`__init__`)

*   `gyromagnetic_ratio_hz_t` (float, optional)
*   `device` (str, optional)

### Key Simulation Method Parameters (`simulate_pulse_response`)

*   `rf_amp_waveform_tesla` (torch.Tensor): RF amplitude modulation (Tesla).
*   `rf_freq_waveform_hz` (torch.Tensor): RF frequency modulation (Hz), relative to scanner's base frequency.
*   `pulse_duration_s` (float): Total pulse duration.
*   `dt_s` (float): Simulation time step (should match RF waveform sampling).
*   `target_b1_scales` (torch.Tensor): 1D Tensor of B1 scaling factors to test.
*   `target_b0_offsets_hz` (torch.Tensor): 1D Tensor of global B0 offsets (Hz) to test.
*   `tissue_params` (dict): Dictionary of multi-pool tissue parameters:
    *   `'num_pools'` (int)
    *   `'M0_fractions'` (Tensor, shape `(num_pools,)`): Equilibrium Mz for each pool.
    *   `'T1s'` (Tensor, shape `(num_pools,)`): T1 times (s).
    *   `'T2s'` (Tensor, shape `(num_pools,)`): T2 times (s).
    *   `'freq_offsets_hz'` (Tensor, shape `(num_pools,)`): Chemical shifts (Hz) relative to reference (e.g., water at 0 Hz).
    *   `'exchange_rates_k_to_from'` (Tensor, shape `(num_pools, num_pools)`): `k[i,j]` is rate from pool `j` to pool `i` (Hz).
*   `initial_M_vector_flat` (torch.Tensor, optional): Initial [Mx,My,Mz,...] for all pools. Defaults to equilibrium.

### Example: Simulating a CEST Z-Spectrum with an HS Pulse

```python
from mri_pulse_library.simulators import AdiabaticCESTSimulator
from mri_pulse_library.rf_pulses.adiabatic import generate_hs_pulse_waveforms
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
import torch

# 1. Generate Adiabatic Pulse (e.g., HS pulse for saturation)
duration = 100e-3 # 100 ms saturation pulse
n_points_rf = 1000
actual_dt_s = duration / n_points_rf
peak_b1_sat_tesla = 2e-6 # 2 uT saturation B1

# For CEST presaturation, frequency modulation is typically zero (on-resonance pulse)
# but we apply the saturation at different offsets using target_b0_offsets_hz
# So, rf_freq_waveform_hz for the pulse itself is zero.
# We can use generate_hs_pulse_waveforms with bandwidth_hz=0 for a non-swept amplitude shape.
# However, the AdiabaticCESTSimulator expects rf_freq_waveform_hz as the pulse's intrinsic FM.
# For a simple saturation pulse applied at various offsets, we can use a constant amplitude
# or a shaped amplitude with zero intrinsic frequency modulation.

# Let's use a constant amplitude pulse for simplicity here for Z-spectrum.
# Or, an HS amplitude shape with zero bandwidth for shaping.
_, sat_amp_shape_tesla, _ = generate_hs_pulse_waveforms(
    pulse_duration_s=duration, num_timepoints=n_points_rf,
    peak_b1_tesla=peak_b1_sat_tesla, bandwidth_hz=0, beta=5.0, device='cpu'
)
sat_freq_shape_hz = torch.zeros_like(sat_amp_shape_tesla)


# 2. Define Tissue Parameters (2-Pool: Water and Amide Proton Transfer)
dev = 'cpu'
m0_water = 0.99
m0_amide = 0.01
amide_ppm = 3.5
# Larmor frequency for 3T example (use actual scanner freq if known)
scanner_freq_mhz = 127.74 # For 3T
amide_offset_hz = amide_ppm * scanner_freq_mhz

pool_params_cest = {
    'num_pools': 2,
    'M0_fractions': torch.tensor([m0_water, m0_amide], device=dev),
    'T1s': torch.tensor([1.3, 0.85], device=dev),      # s (Water, Amide)
    'T2s': torch.tensor([0.05, 0.01], device=dev),    # s (Water, Amide)
    'freq_offsets_hz': torch.tensor([0.0, amide_offset_hz], device=dev), # Water at 0 Hz
    'exchange_rates_k_to_from': torch.zeros((2,2), device=dev) # k_to_from[i,j] = rate from j to i
}
k_amide_water = 30.0 # Hz (exchange rate Amide -> Water)
k_water_amide = k_amide_water * m0_amide / m0_water # Detailed balance
pool_params_cest['exchange_rates_k_to_from'][0,1] = k_amide_water
pool_params_cest['exchange_rates_k_to_from'][1,0] = k_water_amide

# 3. Define Simulation Parameters (Z-spectrum offsets)
# Saturation offsets for Z-spectrum (these are the global B0 offsets)
z_spectrum_offsets_hz = torch.linspace(-500, 500, 101, device=dev) # e.g., -500Hz to +500Hz
# For adiabatic robustness check, B1 scales would be varied. For Z-spectrum, usually one B1_scale.
b1_scales = torch.tensor([1.0], device=dev) # Single B1 scale

# 4. Instantiate Simulator and Run
cest_sim = AdiabaticCESTSimulator(device=dev)
z_spectrum = cest_sim.simulate_pulse_response(
    rf_amp_waveform_tesla=sat_amp_shape_tesla, # Use the shaped amplitude
    rf_freq_waveform_hz=sat_freq_shape_hz,   # Zero intrinsic frequency modulation
    pulse_duration_s=duration,
    dt_s=actual_dt_s,
    target_b1_scales=b1_scales,
    target_b0_offsets_hz=z_spectrum_offsets_hz,
    tissue_params=pool_params_cest
)

# z_spectrum will have shape (1, 101). Squeeze for plotting.
final_mz_water = z_spectrum.squeeze()
print(f"Simulated Z-spectrum (Mz of water): {final_mz_water.shape} points.")
# Plot z_spectrum_offsets_hz vs final_mz_water to see CEST spectrum
```

### Output

The `simulate_pulse_response` method returns a 2D `torch.Tensor` where rows correspond to `target_b1_scales` and columns to `target_b0_offsets_hz`. The values are typically the Mz of the primary pool (e.g., water) after the pulse, which can be used to analyze Z-spectra or B1/B0 robustness.
