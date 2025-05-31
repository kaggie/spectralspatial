# pTx Spokes Pulses for Tailored 3D Excitation

Parallel Transmit (pTx) "Spokes" pulses are an advanced RF pulse design technique used to achieve tailored 3D magnetization profiles (Mx, My, Mz). This is accomplished by jointly optimizing multi-channel RF waveforms and time-varying 3-axis magnetic field gradient waveforms. The term "spokes" refers to the discrete k-space trajectories often traversed during such pulses.

The `SpokesPulseDesigner` class in this library provides tools to design these complex pulses.

## Using `SpokesPulseDesigner`

The designer uses an iterative optimization algorithm to find the RF and gradient waveforms that best produce the desired 3D target magnetization pattern, considering subject-specific B1+ sensitivity maps, B0 off-resonance, and tissue properties.

### Key Initialization Parameters (`__init__`)

*   `num_channels` (int): Number of RF transmit channels.
*   `pulse_duration_s` (float): Total duration of the RF pulse segment (e.g., one spoke or the entire pulse).
*   `dt_s` (float): Time step for simulation in seconds.
*   `spatial_grid_m` (torch.Tensor): Defines spatial coordinates (x,y,z) for each voxel. Shape: `(Nx, Ny, Nz, 3)`, Units: meters.
*   `target_magnetization_profile` (torch.Tensor): Desired final magnetization `[Mx, My, Mz]` per voxel. Shape: `(Nx, Ny, Nz, 3)`.
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio.
*   `device` (str, optional): PyTorch device ('cpu' or 'cuda').

### Key Design Method Parameters (`design_spokes_pulse`)

*   `b1_maps_subject_tesla` (torch.Tensor): B1+ sensitivity maps for the subject. Shape: `(num_channels, Nx, Ny, Nz)`, Units: Tesla.
*   `b0_map_subject_hz` (torch.Tensor): B0 off-resonance map for the subject. Shape: `(Nx, Ny, Nz)`, Units: Hz.
*   `tissue_properties_subject` (dict): Tissue properties for the subject. Expected keys: `'T1'` (T1 map, s) and `'T2'` (T2 map, s), both tensors of shape `(Nx, Ny, Nz)`.
*   `initial_rf_waveforms` (torch.Tensor, optional): Initial guess for RF waveforms. Complex, shape: `(num_channels, N_timepoints)`.
*   `initial_gradient_waveforms_tm` (torch.Tensor, optional): Initial guess for gradient waveforms [Gx, Gy, Gz]. Real, shape: `(N_timepoints, 3)`, Units: T/m.
*   `num_iterations` (int, optional): Number of optimization iterations.
*   `learning_rate` (float, optional): Learning rate for the optimizer.
*   `optimizer_type` (str, optional): Type of optimizer (e.g., 'Adam', 'LBFGS').
*   `cost_mask` (torch.Tensor, optional): Voxel-wise mask for cost calculation. Shape `(Nx, Ny, Nz)`.
*   `cost_component_weights` (tuple, optional): Weights for `(Mx, My, Mz)` error components in the cost function.

### SAR and Gradient Constraints

The `SpokesPulseDesigner` can incorporate constraints related to Specific Absorption Rate (SAR) surrogates and total gradient power during optimization. This helps in designing pulses that are mindful of patient safety limits and hardware constraints.

These constraints are controlled via parameters in the `__init__` method:

**RF/SAR Constraints:**

*   `sar_constraint_mode` (str, default: `'none'`): Determines the type of SAR-related constraint for RF.
    *   `'none'`: No RF/SAR constraints are applied.
    *   `'local_b1_sq_peak'`: Penalizes if the peak squared B1+ magnitude (summed over channels, maximized over time and space) exceeds a limit. This serves as a surrogate for local SAR hotspots. Requires `local_b1_sq_limit_tesla_sq` to be set.
    *   `'total_rf_power'`: Penalizes if the total RF energy (approximated by `sum(|RF_waveforms|^2 * dt_s)`) exceeds a limit. This is a surrogate for global average SAR. Requires `total_rf_power_limit` to be set.
*   `sar_lambda` (float, default: `1.0`): The weighting factor (penalty multiplier) for the RF/SAR constraint term in the cost function.
*   `local_b1_sq_limit_tesla_sq` (float, optional): The limit for the peak local `|B1+|^2` in Tesla^2. Required if `sar_constraint_mode` is `'local_b1_sq_peak'`. (e.g., `(2.5e-6)**2` for a 2.5 µT B1+ peak limit).
*   `total_rf_power_limit` (float, optional): The limit for `sum(|RF_waveforms|^2 * dt_s)`. Required if `sar_constraint_mode` is `'total_rf_power'`.

**Gradient Constraints:**

*   `constrain_gradient_power` (bool, default: `False`): If `True`, enables a penalty on the total gradient power.
*   `gradient_power_lambda` (float, default: `1.0`): The weighting factor for the gradient power penalty.
*   `total_gradient_power_limit` (float, optional): The limit for `sum(|Gx(t)|^2 + |Gy(t)|^2 + |Gz(t)|^2) * dt_s`. Required if `constrain_gradient_power` is `True`. This helps to limit peripheral nerve stimulation (PNS) or hardware demands.

**Note on Approximations:**
The SAR constraints are based on surrogates (B1+ field magnitude for local SAR, total RF power for global SAR). Actual SAR depends on E-fields and detailed tissue models. The gradient power constraint is a general measure of gradient activity.

**Example: Initializing with Constraints**

```python
designer = SpokesPulseDesigner(
    # ... other initialization parameters ...
    sar_constraint_mode='local_b1_sq_peak',
    sar_lambda=1e8,  # Adjust based on cost function scale
    local_b1_sq_limit_tesla_sq=(2.0e-6)**2, # Limit B1+ peak to 2.0 µT

    constrain_gradient_power=True,
    gradient_power_lambda=1e4, # Adjust based on cost function scale
    total_gradient_power_limit=1e-5 # Example limit for sum(|G|^2*dt) in (T/m)^2*s
)
```
During the `design_spokes_pulse` optimization, if the calculated values exceed these limits, corresponding penalty terms are added to the cost function, guiding the optimizer towards solutions that satisfy the constraints.

### Example Usage Snippet

```python
from mri_pulse_library.simulators import SpokesPulseDesigner
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
import torch

# --- Define Design Parameters ---
num_channels = 2
pulse_duration = 1e-3 # 1 ms spoke duration
dt = 4e-6            # 4 us time step
device = 'cpu'

# Define spatial grid (e.g., 16x16x1 for a single slice)
Nx, Ny, Nz = 16, 16, 1
x_coords = torch.linspace(-0.05, 0.05, Nx, device=device) # -5cm to 5cm
y_coords = torch.linspace(-0.05, 0.05, Ny, device=device)
z_coords = torch.tensor([0.0], device=device)       # Single slice at z=0
grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # meters

# Define a target magnetization profile (e.g., excite Mxy in a central region)
target_M = torch.zeros((Nx, Ny, Nz, 3), device=device)
target_M[..., 2] = 1.0 # Default to Mz=1 (equilibrium)
center_x, center_y = Nx // 2, Ny // 2
radius_vox = Nx // 4
for r_idx in range(Nx):
    for c_idx in range(Ny):
        if (r_idx - center_x)**2 + (c_idx - center_y)**2 < radius_vox**2:
            target_M[r_idx, c_idx, 0, 0] = 0.707 # Target Mx component
            target_M[r_idx, c_idx, 0, 1] = 0.707 # Target My component
            target_M[r_idx, c_idx, 0, 2] = 0.0   # Target Mz component (fully tipped)

# --- Instantiate Designer ---
designer = SpokesPulseDesigner(
    num_channels=num_channels,
    pulse_duration_s=pulse_duration,
    dt_s=dt,
    spatial_grid_m=spatial_grid,
    target_magnetization_profile=target_M,
    device=device
)

# --- Prepare Dummy Subject Data (replace with actual data) ---
# B1 maps: (num_channels, Nx, Ny, Nz), complex, Tesla
b1_maps = torch.rand(num_channels, Nx, Ny, Nz, dtype=torch.complex64, device=device) * 1e-7
# B0 map: (Nx, Ny, Nz), Hz
b0_map = (torch.rand(Nx, Ny, Nz, device=device) - 0.5) * 50 # -25 to 25 Hz
# T1/T2 maps: (Nx, Ny, Nz), seconds
t1_map = torch.ones(Nx, Ny, Nz, device=device) * 1.0
t2_map = torch.ones(Nx, Ny, Nz, device=device) * 0.08
tissue_props = {'T1': t1_map, 'T2': t2_map}

# --- Run Pulse Design ---
# Note: For real applications, more iterations and careful LR tuning are needed.
optimized_rf, optimized_gradients = designer.design_spokes_pulse(
    b1_maps_subject_tesla=b1_maps,
    b0_map_subject_hz=b0_map,
    tissue_properties_subject=tissue_props,
    num_iterations=10, # Example: 10 iterations
    learning_rate=0.01
)

print(f"Optimized RF waveforms shape: {optimized_rf.shape}")
print(f"Optimized Gradient waveforms shape: {optimized_gradients.shape}")
# Further analysis, export, or use of these waveforms would follow.
```

### Output

The `design_spokes_pulse` method returns a tuple containing:
1.  Optimized RF waveforms (`torch.Tensor`, complex, shape: `(num_channels, N_timepoints)`).
2.  Optimized Gradient waveforms (`torch.Tensor`, real, shape: `(N_timepoints, 3)`, units: T/m).

These waveforms are designed to work together to achieve the specified target magnetization profile.
