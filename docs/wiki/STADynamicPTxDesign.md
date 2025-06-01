# Dynamic pTx Pulse Design with Small Tip Angle (STA) Approximation

Dynamic Parallel Transmit (pTx) techniques allow for spatio-temporal control over the RF transmit field (B1+) by using multiple transmit channels with time-varying waveforms. This enables advanced applications like spatially selective excitation and improved B1+ homogeneity.

The Small Tip Angle (STA) approximation simplifies the Bloch equations, making the relationship between the RF pulse and the resulting transverse magnetization linear. This allows for computationally efficient pulse design using linear algebra, especially for low flip angle pulses.

## `STAPTxDesigner` Class

The `STAPTxDesigner` class, located in `mri_pulse_library.ptx.sta_designer`, provides tools to design multi-channel RF pulses for specified k-space trajectory points (kT-points) using the STA approximation.

### Initialization

```python
from mri_pulse_library.ptx.sta_designer import STAPTxDesigner

sta_designer = STAPTxDesigner(verbose=True)
```
-   `verbose` (bool, optional): If `True` (default), prints design information.

### Method: `design_kta_pulse`

This method designs the RF pulse values (complex amplitudes) for each channel to be applied at each specified kT-point.

#### Key Parameters:

*   `b1_maps` (torch.Tensor): Complex B1+ sensitivity maps (units: Tesla/Volt or arbitrary units that are consistent with desired RF output units).
    *   Shape: `(num_channels, Nx, Ny, Nz)` or `(num_channels, N_voxels_total)`.
*   `target_mxy_pattern` (torch.Tensor): Desired complex transverse magnetization (Mx + iMy) after the pulse.
    *   Shape: `(Nx, Ny, Nz)` or `(N_voxels_total)`.
*   `kt_points` (torch.Tensor): k-space points defining the transmit trajectory (units: rad/m).
    *   Shape: `(Num_kT_points, N_spatial_dims)` (typically `N_spatial_dims` is 2 or 3).
*   `spatial_grid_m` (torch.Tensor): Spatial coordinates (x,y,z) for each voxel (units: meters).
    *   Shape: `(Nx, Ny, Nz, N_spatial_dims)` or `(N_vox_total, N_spatial_dims)`.
*   `dt_per_kt_point` (float): Effective duration or dwell time for each kT-point (units: seconds). This scales the RF pulse amplitudes.
*   `regularization_factor` (float, optional): Lambda for L2 regularization on the RF pulse amplitudes to control power and improve stability. Default: 1e-3.
*   `target_spatial_mask` (torch.Tensor, optional): Boolean mask defining the Region of Interest (ROI) where the `target_mxy_pattern` should be matched. If `None`, all voxels are considered.
    *   Shape: `(Nx, Ny, Nz)` or `(N_voxels_total)`.

#### Returns:

*   `rf_waveforms_per_channel` (torch.Tensor): Complex RF pulse values (units consistent with B1 maps, e.g., Volts) for each channel at each kT-point.
    *   Shape: `(num_channels, Num_kT_points)`.

#### STA Formulation Solved:
The method solves the regularized least-squares problem:
Minimize: `|| S * rf_vec - mxy_target_vec ||^2_2 + lambda * ||rf_vec||^2_2`
Where `S` is the STA system matrix encoding B1 sensitivities and k-space phase modulation, `rf_vec` is the flattened vector of RF values across channels and kT-points, and `mxy_target_vec` is the desired magnetization in the ROI.

## Example Usage

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mri_pulse_library.ptx.sta_designer import STAPTxDesigner
from mri_pulse_library.core.constants import GAMMA_RAD_PER_S_PER_T_PROTON, M0_PROTON

# --- 1. Setup Parameters and Designer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sta_designer = STAPTxDesigner(verbose=True)

num_channels = 2
Nx, Ny, Nz_spatial = 16, 16, 1  # Spatial grid for B1/target
num_kt_points = 20
num_spatial_dims_k = 2 # For kx, ky

# --- 2. Create Dummy Input Tensors ---
# B1 maps (num_channels, Nx, Ny, Nz_spatial) - complex
b1_maps_torch = torch.rand(num_channels, Nx, Ny, Nz_spatial, dtype=torch.complex64, device=device) * 1e-6 + (1e-7+1e-7j)

# Target Mxy pattern (Nx, Ny, Nz_spatial) - complex (e.g., small spot excitation)
target_mxy_torch = torch.zeros(Nx, Ny, Nz_spatial, dtype=torch.complex64, device=device)
center_x, center_y = Nx // 2, Ny // 2
radius = Nx // 4
for r_idx in range(Nx):
    for c_idx in range(Ny):
        if (r_idx - center_x)**2 + (c_idx - center_y)**2 < radius**2:
            target_mxy_torch[r_idx, c_idx, 0] = 0.02 + 0.01j # Small target Mxy

# kT-points (Num_kT_points, num_spatial_dims_k) - rad/m (e.g., simple linear trajectory)
kt_pts_torch = torch.zeros(num_kt_points, num_spatial_dims_k, device=device)
kt_pts_torch[:,0] = torch.linspace(-np.pi/0.05, np.pi/0.05, num_kt_points, device=device) # Sweep kx

# Spatial grid (Nx, Ny, Nz_spatial, num_spatial_dims_k) - meters
grid_x_lin = torch.linspace(-0.1, 0.1, Nx, device=device)
grid_y_lin = torch.linspace(-0.1, 0.1, Ny, device=device)
mg_x, mg_y = torch.meshgrid(grid_x_lin, grid_y_lin, indexing='ij')

# Need to expand for Nz_spatial and stack for num_spatial_dims_k
spatial_grid_torch = torch.stack(
    (mg_x.unsqueeze(2).expand(-1,-1,Nz_spatial),
     mg_y.unsqueeze(2).expand(-1,-1,Nz_spatial)),
    dim=-1
)

dt_val = 4e-6  # Effective time per kT-point
reg_val = 1e-4

# Optional: Target mask (use all voxels for this example)
# mask_torch = torch.ones(Nx, Ny, Nz_spatial, dtype=torch.bool, device=device)

# --- 3. Design the STA pTx Pulse ---
rf_waveforms = sta_designer.design_kta_pulse(
    b1_maps=b1_maps_torch,
    target_mxy_pattern=target_mxy_torch,
    kt_points=kt_pts_torch,
    spatial_grid_m=spatial_grid_torch,
    dt_per_kt_point=dt_val,
    regularization_factor=reg_val,
    target_spatial_mask=None # Using all voxels from target_mxy_pattern
)

print(f"Designed RF waveforms per channel per kT-point. Shape: {rf_waveforms.shape}")
print(f"Example RF value (Ch0, kT0): {rf_waveforms[0,0]}")
print(f"Max RF amplitude: {torch.max(torch.abs(rf_waveforms)):.2e}")

# --- 4. Visualization (Conceptual) ---
# To visualize the actual excitation, these RF waveforms would typically be
# shaped by a basis function (e.g., short rects) to form time-continuous RF,
# and then simulated using a pTx Bloch simulator (like bloch_simulate_ptx)
# along with the gradient waveforms that achieve the kt_points.

# Ensure matplotlib is available for the example to run
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(torch.abs(rf_waveforms[0,:]).cpu().numpy().reshape(1,-1), aspect='auto', cmap='viridis')
    plt.title('Ch 1 RF Amplitudes vs kT-point')
    plt.xlabel('kT-point index')
    plt.ylabel('Channel 1')
    plt.colorbar()

    if num_channels > 1:
        plt.subplot(1,2,2)
        plt.imshow(torch.abs(rf_waveforms[1,:]).cpu().numpy().reshape(1,-1), aspect='auto', cmap='viridis')
        plt.title('Ch 2 RF Amplitudes vs kT-point')
        plt.xlabel('kT-point index')
        plt.ylabel('Channel 2')
        plt.colorbar()

    plt.tight_layout()
    plt.show()
except ImportError:
    print("Matplotlib not found, skipping example plot.")
except Exception as e:
    print(f"Error during plotting: {e}")
```

## Notes

-   The output `rf_waveforms_per_channel` represents the complex RF amplitudes (e.g., in Volts, if B1 maps are in Tesla/Volt) to be played out for each channel *while the k-space trajectory is at the corresponding kT-point*. To form a time-continuous RF pulse for simulation or execution, these amplitudes would typically modulate a basis function (e.g., a short rectangular pulse of duration `dt_per_kt_point` for each kT-point).
-   This designer does **not** design the gradient waveforms required to achieve the `kt_points` trajectory. These must be designed separately.
-   The STA approximation is most accurate for small flip angles. For larger flip angles, non-linear effects become significant, and more advanced designers (like `SpokesPulseDesigner` or `UniversalPulseDesigner` in this library, or iterative STA methods) might be required.
