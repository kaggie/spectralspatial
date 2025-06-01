# Universal pTx Pulses

Universal Parallel Transmit (pTx) pulses are designed to produce a consistent RF excitation (e.g., a uniform flip angle) across a target volume for a population of subjects, despite variations in B1+ sensitivity maps and B0 off-resonance fields that occur between different subjects or scan sessions.

The `UniversalPulseDesigner` class in this library implements an optimization process to design such pulses.

## Using `UniversalPulseDesigner`

The designer iteratively refines a set of RF waveforms (one for each transmit channel) to minimize the difference between the simulated flip angle profile and a target flip angle profile, averaged over a database of B1/B0 maps.

### Key Initialization Parameters (`__init__`)

*   `num_channels` (int): Number of transmit channels.
*   `pulse_duration_s` (float): Total duration of the RF pulse in seconds.
*   `dt_s` (float): Time step for simulation in seconds.
*   `spatial_grid_m` (torch.Tensor): Defines spatial coordinates (x,y,z) for each voxel. Shape: `(Nx, Ny, Nz, 3)`, Units: meters.
*   `target_flip_angle_profile_rad` (torch.Tensor): Desired flip angle at each voxel. Shape: `(Nx, Ny, Nz)`, Units: radians.
*   `gyromagnetic_ratio_hz_t` (float, optional): Gyromagnetic ratio.
*   `device` (str, optional): PyTorch device ('cpu' or 'cuda').

### Key Design Method Parameters (`design_pulse`)

*   `b1_maps_database` (list): List of B1+ sensitivity maps (Tesla) for each subject/dataset. Each element is a tensor of shape `(num_channels, Nx, Ny, Nz)`.
*   `b0_maps_database` (list): List of B0 off-resonance maps (Hz) for each subject/dataset. Each element is a tensor of shape `(Nx, Ny, Nz)`.
*   `tissue_properties_database` (list): List of tissue property maps. Each element is a dict `{'T1': T1_map (s), 'T2': T2_map (s)}`, where maps are tensors of shape `(Nx, Ny, Nz)`.
*   `initial_rf_waveforms` (torch.Tensor, optional): Initial guess for RF waveforms. Shape: `(num_channels, N_timepoints)`, complex.
*   `num_iterations` (int, optional): Number of optimization iterations.
*   `learning_rate` (float, optional): Learning rate for the optimizer.
*   `optimizer_type` (str, optional): Type of optimizer ('Adam' or 'LBFGS').

### SAR Constraints

The `UniversalPulseDesigner` can incorporate constraints related to Specific Absorption Rate (SAR) surrogates during optimization. This helps in designing pulses that are mindful of patient safety limits.

These constraints are controlled via parameters in the `__init__` method:

*   `sar_constraint_mode` (str, default: `'none'`): Determines the type of SAR-related constraint.
    *   `'none'`: No SAR constraints are applied.
    *   `'local_b1_sq_peak'`: Penalizes if the peak squared B1+ magnitude (summed over channels, maximized over time and space) exceeds a limit. This serves as a surrogate for local SAR hotspots. Requires `local_b1_sq_limit_tesla_sq` to be set.
    *   `'total_rf_power'`: Penalizes if the total RF energy (approximated by `sum(|RF_waveforms|^2 * dt_s)`) exceeds a limit. This is a surrogate for global average SAR. Requires `total_rf_power_limit` to be set.
*   `sar_lambda` (float, default: `1.0`): The weighting factor (penalty multiplier) for the SAR constraint term in the cost function. Higher values enforce the constraint more strictly.
*   `local_b1_sq_limit_tesla_sq` (float, optional): The limit for the peak local `|B1+|^2` in Tesla^2. Required if `sar_constraint_mode` is `'local_b1_sq_peak'`. For example, a B1+ peak limit of 3 µT would correspond to a limit of `(3e-6)**2 = 9e-12` T^2.
*   `total_rf_power_limit` (float, optional): The limit for the sum of squared RF waveform amplitudes multiplied by `dt_s`. This is a surrogate for total RF energy. Required if `sar_constraint_mode` is `'total_rf_power'`. The units depend on how RF waveforms are scaled; if RF waveforms are unitless scaling factors for B1 maps in Tesla, this limit is on `sum(|scales|^2 * dt_s)`. If RF waveforms are directly in Tesla, then it's `sum(|Tesla|^2 * dt_s)`.

**Note on Approximations:**
The SAR constraints implemented are based on surrogates:
-   `local_b1_sq_peak` uses the squared magnitude of the B1+ field as an indicator for potential local SAR hotspots. Actual local SAR depends on electric fields and tissue properties, which are more complex to model directly in the optimization loop.
-   `total_rf_power` relates to the overall energy deposited by the RF pulse, which is a factor in global average SAR.

**Example: Initializing with a Local B1 Peak Constraint**

```python
designer = UniversalPulseDesigner(
    # ... other initialization parameters ...
    sar_constraint_mode='local_b1_sq_peak',
    sar_lambda=1e7,  # Adjust this based on cost function scale
    local_b1_sq_limit_tesla_sq=(2.5e-6)**2 # Limit B1+ peak to 2.5 µT
)
```
During the `design_pulse` optimization, if the calculated peak local `|B1+|^2` (for any subject in the database) exceeds this limit, a penalty term is added to the cost function, guiding the optimizer towards solutions that satisfy the constraint.

### Example Usage Snippet

```python
from mri_pulse_library.simulators import UniversalPulseDesigner
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
import torch

# --- Define Design Parameters ---
num_channels = 8
pulse_duration = 10e-3 # 10 ms
dt = 4e-6             # 4 us time step
device = 'cpu'

# Define spatial grid (e.g., 32x32x1 for a single slice)
Nx, Ny, Nz = 32, 32, 1
x_coords = torch.linspace(-0.1, 0.1, Nx, device=device) # -10cm to 10cm
y_coords = torch.linspace(-0.1, 0.1, Ny, device=device)
z_coords = torch.tensor([0.0], device=device)       # Single slice at z=0
grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # meters

# Define target flip angle profile (e.g., uniform 30 degrees)
target_flip_angle_deg = 30.0
target_flip_angle_rad = torch.ones((Nx, Ny, Nz), device=device) * (target_flip_angle_deg * torch.pi / 180.0)

# --- Instantiate Designer ---
designer = UniversalPulseDesigner(
    num_channels=num_channels,
    pulse_duration_s=pulse_duration,
    dt_s=dt,
    spatial_grid_m=spatial_grid,
    target_flip_angle_profile_rad=target_flip_angle_rad,
    device=device
)

# --- Prepare Dummy Subject Databases (replace with actual data) ---
num_subjects = 2 # Example with 2 subjects
b1_maps_db = []
b0_maps_db = []
tissue_props_db = []

for _ in range(num_subjects):
    # Dummy B1 maps: (num_channels, Nx, Ny, Nz), complex, Tesla
    b1_map = torch.rand(num_channels, Nx, Ny, Nz, dtype=torch.complex64, device=device) * 1e-7 # Example peak ~0.1uT
    b1_maps_db.append(b1_map)

    # Dummy B0 maps: (Nx, Ny, Nz), Hz
    b0_map = (torch.rand(Nx, Ny, Nz, device=device) - 0.5) * 100 # -50 to 50 Hz
    b0_maps_db.append(b0_map)

    # Dummy T1/T2 maps: (Nx, Ny, Nz), seconds
    t1_map = torch.ones(Nx, Ny, Nz, device=device) * 0.8  # 800 ms
    t2_map = torch.ones(Nx, Ny, Nz, device=device) * 0.08 # 80 ms
    tissue_props_db.append({'T1': t1_map, 'T2': t2_map})

# --- Run Pulse Design ---
optimized_rf_waveforms = designer.design_pulse(
    b1_maps_database=b1_maps_db,
    b0_maps_database=b0_maps_db,
    tissue_properties_database=tissue_props_db,
    num_iterations=20, # Example iterations
    learning_rate=0.005
)

print(f"Optimized RF waveforms shape: {optimized_rf_waveforms.shape}")
# Further analysis/export of optimized_rf_waveforms would follow
```

### Output

The `design_pulse` method returns a `torch.Tensor` containing the optimized complex RF waveforms for each channel, with shape `(num_channels, N_timepoints)`. These waveforms can then be exported or used in pulse sequence simulations.
