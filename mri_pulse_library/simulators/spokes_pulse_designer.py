"""Designer for parallel transmit (pTx) spokes pulses, which aim to achieve
arbitrary 3D target magnetization profiles by jointly optimizing RF and
gradient waveforms.
"""
import torch
import torch.nn as nn
import torch.optim as optim # Will be used in later steps
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
from mri_pulse_library.core.bloch_sim import bloch_simulate_ptx

class SpokesPulseDesigner:
    """
    Designs parallel transmit (pTx) RF and gradient waveforms for "spokes"
    pulses. These pulses are tailored to achieve a user-defined 3D target
    magnetization profile (Mx, My, Mz) across a specified spatial volume.

    The design process involves an iterative optimization that adjusts the RF
    waveforms for multiple transmit channels and the corresponding 3-axis
    gradient waveforms. This optimization is based on Bloch simulations that
    account for subject-specific B1+ sensitivity maps, B0 off-resonance
    fields, and tissue relaxation properties (T1, T2).
    """
    def __init__(self,
                 num_channels: int,
                 pulse_duration_s: float,
                 dt_s: float,
                 spatial_grid_m: torch.Tensor,
                 target_magnetization_profile: torch.Tensor,
                 gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                 device: str = 'cpu'):
        """
        Initializes the SpokesPulseDesigner.

        Args:
            num_channels (int): Number of RF transmit channels.
            pulse_duration_s (float): Total duration of the RF pulse segment (e.g., one spoke).
            dt_s (float): Time step for simulation in seconds.
            spatial_grid_m (torch.Tensor): Defines spatial coordinates (x,y,z) for each voxel.
                                           Shape: (Nx, Ny, Nz, 3). Units: meters.
            target_magnetization_profile (torch.Tensor): Desired final magnetization [Mx, My, Mz]
                                                       per voxel. Shape: (Nx, Ny, Nz, 3).
            gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio.
                                                      Defaults to GAMMA_HZ_PER_T_PROTON.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if not isinstance(num_channels, int) or num_channels <= 0:
            raise ValueError("num_channels must be a positive integer.")
        if not isinstance(pulse_duration_s, (float, int)) or pulse_duration_s <= 0:
            raise ValueError("pulse_duration_s must be a positive number.")
        if not isinstance(dt_s, (float, int)) or dt_s <= 0:
            raise ValueError("dt_s must be a positive number.")
        if pulse_duration_s < dt_s:
            raise ValueError("pulse_duration_s cannot be less than dt_s.")

        self.num_channels = num_channels
        self.pulse_duration_s = float(pulse_duration_s)
        self.dt_s = float(dt_s)
        self.device = torch.device(device)
        self.dtype = torch.float32 # Default dtype

        if not isinstance(spatial_grid_m, torch.Tensor) or spatial_grid_m.ndim != 4 or spatial_grid_m.shape[-1] != 3:
            raise ValueError("spatial_grid_m must be a torch.Tensor of shape (Nx, Ny, Nz, 3).")
        self.spatial_grid_m = spatial_grid_m.to(self.device, dtype=self.dtype)
        self.Nx, self.Ny, self.Nz, _ = self.spatial_grid_m.shape

        expected_target_shape = (self.Nx, self.Ny, self.Nz, 3)
        if not isinstance(target_magnetization_profile, torch.Tensor) or target_magnetization_profile.shape != expected_target_shape:
            raise ValueError(f"target_magnetization_profile must be a torch.Tensor of shape {expected_target_shape}.")
        self.target_magnetization_profile = target_magnetization_profile.to(self.device, dtype=self.dtype)

        self.gyromagnetic_ratio_hz_t = float(gyromagnetic_ratio_hz_t)
        self.N_timepoints = int(round(self.pulse_duration_s / self.dt_s))
        if self.N_timepoints <= 0:
            raise ValueError("Number of timepoints must be positive. Check pulse_duration_s and dt_s.")

        self.rf_waveforms_per_channel = None # To be nn.Parameter; Shape: (num_channels, N_timepoints), complex
        self.gradient_waveforms_tm = None    # To be nn.Parameter; Shape: (N_timepoints, 3), real

    def _calculate_magnetization_error_cost(
            self,
            simulated_M_profile: torch.Tensor, # Shape (Nx, Ny, Nz, 3)
            target_M_profile: torch.Tensor,    # Shape (Nx, Ny, Nz, 3)
            cost_mask: torch.Tensor = None,    # Shape (Nx, Ny, Nz), optional
            component_weights: tuple = (1.0, 1.0, 1.0) # Weights for (Mx, My, Mz)
        ) -> torch.Tensor:
        """
        Calculates the cost based on the difference between simulated and target magnetization.

        Args:
            simulated_M_profile (torch.Tensor): Simulated magnetization [Mx, My, Mz] per voxel.
            target_M_profile (torch.Tensor): Target magnetization [Mx, My, Mz] per voxel.
            cost_mask (torch.Tensor, optional): Boolean or float tensor to weigh/ignore
                                                voxels in cost calculation. Shape (Nx, Ny, Nz).
            component_weights (tuple, optional): Weights for (Mx, My, Mz) error components.

        Returns:
            torch.Tensor: Scalar cost value.
        """
        if simulated_M_profile.shape != target_M_profile.shape:
            raise ValueError("Simulated and target magnetization profiles must have the same shape.")
        if simulated_M_profile.shape[-1] != 3:
            raise ValueError("Magnetization profiles must have 3 components (Mx, My, Mz) in the last dimension.")

        error = simulated_M_profile - target_M_profile

        # Ensure component_weights is a tensor on the correct device
        weights = torch.tensor(component_weights, device=self.device, dtype=self.dtype).view(1, 1, 1, 3)

        # Weighted squared error for each component, then sum components
        # (error * weights_expanded_to_match_error_dims_if_needed_or_direct_mult_on_last_dim)
        # error is (Nx,Ny,Nz,3), weights is (1,1,1,3) -> broadcasts
        weighted_error_sq_components = (error**2) * weights
        weighted_error_sq_sum = torch.sum(weighted_error_sq_components, dim=-1) # Sum over M_x, M_y, M_z components

        if cost_mask is not None:
            if cost_mask.shape != (self.Nx, self.Ny, self.Nz):
                raise ValueError(f"cost_mask must have shape ({self.Nx}, {self.Ny}, {self.Nz}).")
            masked_weighted_error_sq_sum = weighted_error_sq_sum * cost_mask.to(self.device, dtype=self.dtype)

            # Normalize by sum of active mask elements and sum of component weights
            # to make cost somewhat independent of mask size and weight magnitudes
            num_active_elements = torch.sum(cost_mask)
            if num_active_elements > 0:
                cost = torch.sum(masked_weighted_error_sq_sum) / (num_active_elements * torch.sum(weights))
            else: # Avoid division by zero if mask is all zeros
                cost = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            cost = torch.mean(weighted_error_sq_sum) # Mean over all voxels and implicitly components due to sum

        return cost

    def design_spokes_pulse(self,
                            b1_maps_subject_tesla: torch.Tensor, # (num_channels, Nx, Ny, Nz)
                            b0_map_subject_hz: torch.Tensor,     # (Nx, Ny, Nz)
                            tissue_properties_subject: dict,     # {'T1': T1_map, 'T2': T2_map}
                            initial_rf_waveforms: torch.Tensor = None,    # (num_channels, N_timepoints), complex
                            initial_gradient_waveforms_tm: torch.Tensor = None, # (N_timepoints, 3), real
                            num_iterations: int = 10,
                            learning_rate: float = 0.01,
                            optimizer_type: str = 'Adam',
                            cost_mask: torch.Tensor = None,
                            cost_component_weights: tuple = (1.0, 1.0, 1.0)):
        """
        Designs RF and gradient waveforms for a pTx spokes pulse by optimizing them
        to achieve the target magnetization profile for a given subject.

        Args:
            b1_maps_subject_tesla (torch.Tensor): B1+ sensitivity maps for the subject.
                                                 Shape: (num_channels, Nx, Ny, Nz), Units: Tesla.
            b0_map_subject_hz (torch.Tensor): B0 off-resonance map for the subject.
                                             Shape: (Nx, Ny, Nz), Units: Hz.
            tissue_properties_subject (dict): Tissue properties for the subject.
                                              Expected keys: 'T1' (T1 map, shape (Nx,Ny,Nz), units: s)
                                              and 'T2' (T2 map, shape (Nx,Ny,Nz), units: s).
            initial_rf_waveforms (torch.Tensor, optional): Initial guess for RF waveforms.
                                                          Must be a complex tensor.
                                                          Shape: (num_channels, N_timepoints).
                                                          If None, random initialization is used.
            initial_gradient_waveforms_tm (torch.Tensor, optional): Initial guess for gradient
                                                                  waveforms [Gx, Gy, Gz].
                                                                  Should be a real tensor.
                                                                  Shape: (N_timepoints, 3). Units: T/m.
                                                                  If None, random initialization is used.
            num_iterations (int, optional): Number of optimization iterations. Defaults to 10.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            optimizer_type (str, optional): Type of optimizer (e.g., 'Adam', 'LBFGS').
                                           Defaults to 'Adam'.
            cost_mask (torch.Tensor, optional): Mask for cost calculation. Applied to spatial dimensions.
                                                Shape (Nx, Ny, Nz). Defaults to None.
            cost_component_weights (tuple, optional): Weights for (Mx, My, Mz) error components
                                                     in the cost function. Defaults to (1.0, 1.0, 1.0).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Optimized RF waveforms (torch.Tensor, complex, shape: (num_channels, N_timepoints)).
                - Optimized Gradient waveforms (torch.Tensor, real, shape: (N_timepoints, 3), units: T/m).
        """
        # RF Waveform Initialization
        if initial_rf_waveforms is not None:
            if not torch.is_complex(initial_rf_waveforms):
                # Ensure complex type, float32 for real/imag parts before complex construction
                initial_rf_waveforms = initial_rf_waveforms.to(dtype=torch.float32)
                if initial_rf_waveforms.shape[-1] == 2: # Interpret last dim as real/imag
                     initial_rf_waveforms = torch.complex(initial_rf_waveforms[...,0], initial_rf_waveforms[...,1])
                else: # Assume it's real and should have zero imag part
                     initial_rf_waveforms = torch.complex(initial_rf_waveforms, torch.zeros_like(initial_rf_waveforms))

            if initial_rf_waveforms.shape != (self.num_channels, self.N_timepoints):
                raise ValueError(f"initial_rf_waveforms must have shape ({self.num_channels}, {self.N_timepoints}).")
            self.rf_waveforms_per_channel = nn.Parameter(initial_rf_waveforms.clone().to(device=self.device, dtype=torch.complex64))
        elif self.rf_waveforms_per_channel is None: # Initialize randomly only if not provided and not already set
            rf_real = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            rf_imag = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            self.rf_waveforms_per_channel = nn.Parameter(torch.complex(rf_real, rf_imag))
        # If self.rf_waveforms_per_channel is already an nn.Parameter, it's used as is.

        # Gradient Waveform Initialization
        if initial_gradient_waveforms_tm is not None:
            if initial_gradient_waveforms_tm.shape != (self.N_timepoints, 3):
                raise ValueError(f"initial_gradient_waveforms_tm must have shape ({self.N_timepoints}, 3).")
            self.gradient_waveforms_tm = nn.Parameter(initial_gradient_waveforms_tm.clone().to(device=self.device, dtype=self.dtype))
        elif self.gradient_waveforms_tm is None: # Initialize randomly only if not provided and not already set
            self.gradient_waveforms_tm = nn.Parameter(torch.randn(self.N_timepoints, 3, device=self.device, dtype=self.dtype) * 1e-5) # Small initial gradients
        # If self.gradient_waveforms_tm is already an nn.Parameter, it's used as is.

        # Ensure target magnetization profile is on the correct device
        target_M_on_device = self.target_magnetization_profile.to(self.device)

        # Optimizer setup
        parameters_to_optimize = [self.rf_waveforms_per_channel, self.gradient_waveforms_tm]
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(parameters_to_optimize, lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS(parameters_to_optimize, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'Adam' or 'LBFGS'.")

        # Prepare subject data (move to device, validate shapes - as in Step 4)
        b1_maps_subject_dev = b1_maps_subject_tesla.to(self.device)
        b0_map_subject_dev = b0_map_subject_hz.to(self.device)
        t1_map_dev = tissue_properties_subject['T1'].to(self.device, dtype=self.dtype)
        t2_map_dev = tissue_properties_subject['T2'].to(self.device, dtype=self.dtype)

        expected_b1_shape = (self.num_channels, self.Nx, self.Ny, self.Nz)
        expected_map_shape = (self.Nx, self.Ny, self.Nz)
        if b1_maps_subject_dev.shape != expected_b1_shape:
            raise ValueError(f"b1_maps_subject_tesla shape mismatch. Expected {expected_b1_shape}, got {b1_maps_subject_dev.shape}")
        if b0_map_subject_dev.shape != expected_map_shape:
            raise ValueError(f"b0_map_subject_hz shape mismatch. Expected {expected_map_shape}, got {b0_map_subject_dev.shape}")
        if t1_map_dev.shape != expected_map_shape:
            raise ValueError(f"T1 map shape mismatch. Expected {expected_map_shape}, got {t1_map_dev.shape}")
        if t2_map_dev.shape != expected_map_shape:
            raise ValueError(f"T2 map shape mismatch. Expected {expected_map_shape}, got {t2_map_dev.shape}")

        initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=self.dtype)
        initial_M_subject[..., 2] = 1.0

        # Optimization loop
        for i in range(num_iterations):
            if optimizer_type.lower() == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    final_magnetization_profile = bloch_simulate_ptx(
                        rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                        b1_sensitivity_maps=b1_maps_subject_dev,
                        dt_s=self.dt_s,
                        b0_map_hz=b0_map_subject_dev,
                        T1_map_s=t1_map_dev,
                        T2_map_s=t2_map_dev,
                        gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t,
                        initial_magnetization=initial_M_subject,
                        spatial_grid_m=self.spatial_grid_m,
                        gradient_waveforms_tm=self.gradient_waveforms_tm,
                        return_all_timepoints=False
                    )
                    current_cost = self._calculate_magnetization_error_cost(
                        final_magnetization_profile, target_M_on_device, cost_mask, cost_component_weights
                    )
                    current_cost.backward()
                    return current_cost

                loss = optimizer.step(closure) # For LBFGS, step is called with closure
                cost_val = loss.item()
            else: # Adam or similar
                optimizer.zero_grad()
                final_magnetization_profile = bloch_simulate_ptx(
                    rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                    b1_sensitivity_maps=b1_maps_subject_dev,
                    dt_s=self.dt_s,
                    b0_map_hz=b0_map_subject_dev,
                    T1_map_s=t1_map_dev,
                    T2_map_s=t2_map_dev,
                    gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t,
                    initial_magnetization=initial_M_subject,
                    spatial_grid_m=self.spatial_grid_m,
                    gradient_waveforms_tm=self.gradient_waveforms_tm,
                    return_all_timepoints=False
                )
                current_cost = self._calculate_magnetization_error_cost(
                    final_magnetization_profile, target_M_on_device, cost_mask, cost_component_weights
                )
                current_cost.backward()
                optimizer.step()
                cost_val = current_cost.item()

            print(f"Iteration {i+1}/{num_iterations}, Cost: {cost_val}")

        return self.rf_waveforms_per_channel.detach().clone(), self.gradient_waveforms_tm.detach().clone()

if __name__ == '__main__':
    print("SpokesPulseDesigner class defined.")
    # Parameters
    n_ch = 2
    dur_s = 0.5e-3 # Shorter spoke for faster example
    dt_s_val = 4e-6
    n_vox_x, n_vox_y, n_vox_z = 4, 4, 1 # Even smaller grid for faster example

    # Dummy spatial grid
    coords = [torch.linspace(-0.05, 0.05, n) for n in (n_vox_x, n_vox_y, n_vox_z)]
    grid_x, grid_y, grid_z = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    # Dummy target magnetization (e.g., excite Mxy in a central circle, Mz elsewhere)
    target_M = torch.zeros((n_vox_x, n_vox_y, n_vox_z, 3))
    target_M[..., 2] = 1.0 # Default to Mz=1
    center_x, center_y = n_vox_x // 2, n_vox_y // 2
    radius_sq = (min(n_vox_x, n_vox_y) // 4)**2
    for i in range(n_vox_x):
        for j in range(n_vox_y):
            if (i - center_x)**2 + (j - center_y)**2 < radius_sq:
                target_M[i, j, 0, 0] = 0.707 # Target Mx
                target_M[i, j, 0, 1] = 0.707 # Target My
                target_M[i, j, 0, 2] = 0.0   # Target Mz (flipped down)

    try:
        designer = SpokesPulseDesigner(
            num_channels=n_ch,
            pulse_duration_s=dur_s,
            dt_s=dt_s_val,
            spatial_grid_m=spatial_grid,
            target_magnetization_profile=target_M,
            device='cpu'
        )
        print(f"SpokesPulseDesigner initialized: {designer.N_timepoints} timepoints.")

        # Test cost function
        sim_M = torch.rand_like(target_M) * 2 - 1 # Random sim M between -1 and 1
        cost = designer._calculate_magnetization_error_cost(sim_M, target_M)
        print(f"Example cost calculation: {cost.item()}")

        # Dummy data for design_spokes_pulse
        dummy_b1 = torch.rand(n_ch, n_vox_x, n_vox_y, n_vox_z, dtype=torch.complex64) * 1e-7 # T
        dummy_b0 = torch.rand(n_vox_x, n_vox_y, n_vox_z) * 5 # Hz
        dummy_t1 = torch.ones(n_vox_x, n_vox_y, n_vox_z) * 0.8 # s
        dummy_t2 = torch.ones(n_vox_x, n_vox_y, n_vox_z) * 0.05 # s
        dummy_tissue = {'T1': dummy_t1, 'T2': dummy_t2}

        # Call design_spokes_pulse
        opt_rf, opt_grad = designer.design_spokes_pulse(
            b1_maps_subject_tesla=dummy_b1,
            b0_map_subject_hz=dummy_b0,
            tissue_properties_subject=dummy_tissue,
            num_iterations=5, # Small number of iterations for quick test
            learning_rate=0.01, # Adam default often 0.001, LBFGS default often 1.0
            optimizer_type='Adam' # or 'LBFGS'
        )
        print(f"design_spokes_pulse finished.")
        print(f"Optimized RF shape: {opt_rf.shape}, Mean abs: {torch.mean(torch.abs(opt_rf)).item()}")
        print(f"Optimized Grad shape: {opt_grad.shape}, Mean abs: {torch.mean(torch.abs(opt_grad)).item()}")

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")
