# Static pTx (B1+) Shimming

Static B1+ shimming in Parallel Transmit (pTx) MRI aims to improve the homogeneity of the transmit RF field (B1+) over a Region of Interest (ROI) by applying a single, optimized complex weight (amplitude and phase) to each transmit channel. Unlike dynamic pTx, the RF waveform shape itself is common across channels (or not explicitly varied in time beyond the base pulse), and only these static weights are adjusted.

This technique is particularly useful for mitigating B1+ inhomogeneities prevalent at high and ultra-high field strengths.

## Theory Overview

The core idea is to find a set of complex shim weights `w = [w_1, w_2, ..., w_C]` (where C is the number of channels) such that the superposition of the B1+ fields from each channel, `B1_effective(r) = sum_c w_c * B1_c(r)`, is as close as possible to a desired target B1+ field (e.g., a uniform field of a specific amplitude) within the ROI. `B1_c(r)` is the B1+ sensitivity map of channel `c` at spatial location `r`.

This is typically formulated as a least-squares optimization problem:
Minimize: `|| A * w - b ||^2_2 + lambda * ||w||^2_2`
Where:
-   `A` is the system matrix, where each row corresponds to a voxel in the ROI, and each column `c` contains the `B1_c(r)` values for that channel across all ROI voxels.
-   `w` is the vector of complex shim weights to be determined.
-   `b` is the target effective B1+ vector within the ROI (e.g., a vector of ones if aiming for uniform unit amplitude with zero phase).
-   `lambda` is a regularization factor to control the power of the shim weights (indirectly related to SAR) and improve numerical stability.

The solution is found using the normal equations: `w = inv(A_H * A + lambda * I) * (A_H * b)`, where `A_H` is the conjugate transpose of `A`.

## Function: `calculate_static_shims`

The `calculate_static_shims` function in `mri_pulse_library.ptx.shimming` implements this static B1+ shimming.

### Key Parameters:

*   `b1_maps` (np.ndarray): Complex B1+ sensitivity maps for each channel.
    *   Shape: `(num_channels, Nx, Ny, Nz)` or `(num_channels, Nvoxels_total)`.
    *   Units: Consistent (e.g., uT/Volt or arbitrary units).
*   `target_mask` (np.ndarray): Boolean mask defining the ROI.
    *   Shape: `(Nx, Ny, Nz)` or `(Nvoxels_total)`.
*   `target_b1_amplitude` (float, optional): Desired uniform effective B1+ amplitude within the ROI (target phase is implicitly 0). Default: 1.0.
*   `regularization_factor` (float, optional): Lambda for L2 regularization on shim weights. Default: 1e-2.
*   `return_achieved_b1` (bool, optional): If `True`, also returns the achieved effective B1+ field within the ROI. Default: `False`.

### Returns:

*   `shim_weights` (np.ndarray): Complex array of optimal shim weights `(num_channels,)`.
*   `achieved_b1_roi` (np.ndarray, optional): If `return_achieved_b1` is `True`, the complex effective B1+ field within the ROI `(Nvoxels_in_ROI,)`.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.ptx.shimming import calculate_static_shims

# --- 1. Define Inputs ---
num_channels = 4
Nx, Ny, Nz = 32, 32, 1 # Small 2D example for simplicity
num_voxels_total = Nx * Ny * Nz

# Create dummy B1 maps (num_channels, Nx, Ny, Nz) - complex values
# In a real scenario, these would be measured.
b1_maps_vol = (np.random.rand(num_channels, Nx, Ny, Nz) - 0.5 +                1j * (np.random.rand(num_channels, Nx, Ny, Nz) - 0.5)) * 2.0
b1_maps_vol += (0.5 + 0.5j) # Ensure they are not centered at zero

# Create a dummy mask (Nx, Ny, Nz) - e.g., a central square ROI
target_mask_vol = np.zeros((Nx, Ny, Nz), dtype=bool)
center_x, center_y = Nx // 2, Ny // 2
roi_half_width = Nx // 4
target_mask_vol[center_x-roi_half_width : center_x+roi_half_width,
                center_y-roi_half_width : center_y+roi_half_width, :] = True

target_amplitude = 1.0 # Desired B1+ magnitude in ROI
reg_factor = 0.05

# --- 2. Calculate Shim Weights ---
shim_weights, b1_eff_roi = calculate_static_shims(
    b1_maps_vol,
    target_mask_vol,
    target_b1_amplitude=target_amplitude,
    regularization_factor=reg_factor,
    return_achieved_b1=True
)

print(f"Calculated Shim Weights (Channel x [Amp, Phase_deg]):")
for i, w in enumerate(shim_weights):
    print(f"  Ch {i+1}: {np.abs(w):.3f}, {np.angle(w, deg=True):.1f}°")

# --- 3. Analyze Results (Optional) ---
# Reconstruct the full B1_effective field
b1_maps_flat = b1_maps_vol.reshape(num_channels, -1)
b1_effective_flat = b1_maps_flat.T @ shim_weights # (Nvoxels_total,)
b1_effective_vol = b1_effective_flat.reshape(Nx, Ny, Nz)

# Compare B1+ magnitude before and after shimming within ROI
# Before shimming (e.g., sum of channels or CP mode approximation)
b1_sum_abs_roi = np.abs(np.sum(b1_maps_vol[:, target_mask_vol], axis=0))
# After shimming
b1_eff_abs_roi_direct = np.abs(b1_eff_roi) # From function output

print(f"\nB1+ Magnitude in ROI:")
if np.mean(b1_sum_abs_roi) != 0 : # Avoid division by zero if initial sum is zero
    print(f"  Initial (sum channels, mean): {np.mean(b1_sum_abs_roi):.3f}, std: {np.std(b1_sum_abs_roi):.3f}, CV: {np.std(b1_sum_abs_roi)/np.mean(b1_sum_abs_roi):.3f}")
else:
    print(f"  Initial (sum channels, mean): {np.mean(b1_sum_abs_roi):.3f}, std: {np.std(b1_sum_abs_roi):.3f}, CV: N/A")

if np.mean(b1_eff_abs_roi_direct) != 0: # Avoid division by zero
    print(f"  Shimme_d (mean): {np.mean(b1_eff_abs_roi_direct):.3f}, std: {np.std(b1_eff_abs_roi_direct):.3f}, CV: {np.std(b1_eff_abs_roi_direct)/np.mean(b1_eff_abs_roi_direct):.3f}")
else:
    print(f"  Shimme_d (mean): {np.mean(b1_eff_abs_roi_direct):.3f}, std: {np.std(b1_eff_abs_roi_direct):.3f}, CV: N/A")


# Plotting example
if Nz == 1: # Only plot if it's effectively 2D
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(np.abs(np.sum(b1_maps_vol[:,:,:,0], axis=0)), aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title('Initial B1+ Mag (Sum of Channels)')
    axs[0].contour(target_mask_vol[:,:,0], levels=[0.5], colors='red', linewidths=0.8)


    im = axs[1].imshow(np.abs(b1_effective_vol[:,:,0]), aspect='auto', origin='lower', cmap='viridis',
                       vmin=np.min(np.abs(b1_sum_abs_roi)) if b1_sum_abs_roi.size > 0 else 0,
                       vmax=np.max(np.abs(b1_sum_abs_roi)) if b1_sum_abs_roi.size > 0 else 1) # Consistent scale
    axs[1].set_title(f'Shimme_d B1+ Mag (Target={target_amplitude})')
    axs[1].contour(target_mask_vol[:,:,0], levels=[0.5], colors='red', linewidths=0.8)
    fig.colorbar(im, ax=axs[1], label='B1+ Magnitude (a.u.)')
    plt.suptitle('Static B1+ Shimming Example')
    plt.tight_layout()
    plt.show()
```

## References
*   For general concepts on B1 shimming: Zhu, Y. (2004). *Parallel excitation with an array of transmit coils.* Magnetic Resonance in Medicine, 51(4), 775-784.
*   Least-squares optimization is a standard technique described in many numerical methods textbooks.
