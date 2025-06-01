import torch
import torch.nn as nn
import torch.optim as optim
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
from mri_pulse_library.core.bloch_sim import bloch_simulate_ptx

class SpokesPulseDesigner:
    def __init__(self,
                 num_channels: int,
                 pulse_duration_s: float,
                 dt_s: float,
                 spatial_grid_m: torch.Tensor,
                 target_magnetization_profile: torch.Tensor,
                 gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                 device: str = 'cpu',
                 sar_constraint_mode: str = 'none',
                 sar_lambda: float = 1.0,
                 local_b1_sq_limit_tesla_sq: float = None,
                 total_rf_power_limit: float = None,
                 constrain_gradient_power: bool = False,
                 gradient_power_lambda: float = 1.0,
                 total_gradient_power_limit: float = None):
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
            sar_constraint_mode (str, optional): SAR constraint type: 'none',
                                                 'local_b1_sq_peak', or 'total_rf_power'.
                                                 Defaults to 'none'.
            sar_lambda (float, optional): Penalty weight for RF/SAR constraints. Defaults to 1.0.
            local_b1_sq_limit_tesla_sq (float, optional): Limit for peak local |B1(t,v)|^2 (Tesla^2).
                                                          Required if mode is 'local_b1_sq_peak'.
            total_rf_power_limit (float, optional): Limit for sum_channels sum_time |RF_ch(t)|^2 * dt_s.
                                                    Required if mode is 'total_rf_power'.
            constrain_gradient_power (bool, optional): Whether to constrain gradient power.
                                                       Defaults to False.
            gradient_power_lambda (float, optional): Penalty weight for gradient power constraint.
                                                     Defaults to 1.0.
            total_gradient_power_limit (float, optional): Limit for sum_time sum_axes |G_axis(t)|^2 * dt_s.
                                                          Required if constrain_gradient_power is True.
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
        self.dtype = torch.float32

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

        self.rf_waveforms_per_channel = None
        self.gradient_waveforms_tm = None

        self.sar_constraint_mode = sar_constraint_mode.lower()
        self.sar_lambda = sar_lambda
        self.local_b1_sq_limit_tesla_sq = local_b1_sq_limit_tesla_sq
        self.total_rf_power_limit = total_rf_power_limit
        self.constrain_gradient_power = constrain_gradient_power
        self.gradient_power_lambda = gradient_power_lambda
        self.total_gradient_power_limit = total_gradient_power_limit

        if self.sar_constraint_mode == 'local_b1_sq_peak' and self.local_b1_sq_limit_tesla_sq is None:
            raise ValueError("local_b1_sq_limit_tesla_sq must be provided for 'local_b1_sq_peak' SAR mode.")
        if self.sar_constraint_mode == 'total_rf_power' and self.total_rf_power_limit is None:
            raise ValueError("total_rf_power_limit must be provided for 'total_rf_power' SAR mode.")
        if self.sar_constraint_mode not in ['none', 'local_b1_sq_peak', 'total_rf_power']:
            raise ValueError(f"Invalid sar_constraint_mode: {self.sar_constraint_mode}")
        if self.constrain_gradient_power and self.total_gradient_power_limit is None:
            raise ValueError("total_gradient_power_limit must be provided if constrain_gradient_power is True.")

    def _calculate_magnetization_error_cost(
            self,
            simulated_M_profile: torch.Tensor,
            target_M_profile: torch.Tensor,
            cost_mask: torch.Tensor = None,
            component_weights: tuple = (1.0, 1.0, 1.0)
        ) -> torch.Tensor:
        if simulated_M_profile.shape != target_M_profile.shape:
            raise ValueError("Simulated and target magnetization profiles must have the same shape.")
        if simulated_M_profile.shape[-1] != 3:
            raise ValueError("Magnetization profiles must have 3 components (Mx, My, Mz) in the last dimension.")
        error = simulated_M_profile - target_M_profile
        weights = torch.tensor(component_weights, device=self.device, dtype=self.dtype).view(1, 1, 1, 3)
        weighted_error_sq_components = (error**2) * weights
        weighted_error_sq_sum = torch.sum(weighted_error_sq_components, dim=-1)
        if cost_mask is not None:
            if cost_mask.shape != (self.Nx, self.Ny, self.Nz):
                raise ValueError(f"cost_mask must have shape ({self.Nx}, {self.Ny}, {self.Nz}).")
            masked_weighted_error_sq_sum = weighted_error_sq_sum * cost_mask.to(self.device, dtype=self.dtype)
            num_active_elements = torch.sum(cost_mask)
            cost = torch.sum(masked_weighted_error_sq_sum) / (num_active_elements * torch.sum(weights)) if num_active_elements > 0 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            cost = torch.mean(weighted_error_sq_sum)
        return cost

    def design_spokes_pulse(self,
                            b1_maps_subject_tesla: torch.Tensor,
                            b0_map_subject_hz: torch.Tensor,
                            tissue_properties_subject: dict,
                            initial_rf_waveforms: torch.Tensor = None,
                            initial_gradient_waveforms_tm: torch.Tensor = None,
                            num_iterations: int = 10,
                            learning_rate: float = 0.01,
                            optimizer_type: str = 'Adam',
                            cost_mask: torch.Tensor = None,
                            cost_component_weights: tuple = (1.0, 1.0, 1.0)):
        """
        Designs RF and gradient waveforms for a pTx spokes pulse.

        Args:
            b1_maps_subject_tesla (torch.Tensor): B1+ sensitivity maps for the subject.
                                                 Shape `(num_channels, Nx, Ny, Nz)`.
            b0_map_subject_hz (torch.Tensor): B0 off-resonance map. Shape `(Nx, Ny, Nz)`.
            tissue_properties_subject (dict): T1 and T2 maps. Keys: 'T1', 'T2'.
            initial_rf_waveforms (torch.Tensor, optional): Initial RF guess. Complex.
                                                          Shape `(num_channels, N_timepoints)`.
            initial_gradient_waveforms_tm (torch.Tensor, optional): Initial gradient guess. Real.
                                                                  Shape `(N_timepoints, 3)`. Units: T/m.
            num_iterations (int, optional): Optimization iterations. Defaults to 10.
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.01.
            optimizer_type (str, optional): 'Adam' or 'LBFGS'. Defaults to 'Adam'.
            cost_mask (torch.Tensor, optional): Voxel-wise cost mask. Shape `(Nx, Ny, Nz)`.
            cost_component_weights (tuple, optional): Weights for (Mx,My,Mz) in cost.
                                                     Defaults to (1.0, 1.0, 1.0).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Optimized RF (complex) and Gradient (real) waveforms.
        """
        if initial_rf_waveforms is not None:
            if not torch.is_complex(initial_rf_waveforms):
                initial_rf_waveforms = initial_rf_waveforms.to(dtype=torch.float32)
                if initial_rf_waveforms.shape[-1] == 2:
                     initial_rf_waveforms = torch.complex(initial_rf_waveforms[...,0], initial_rf_waveforms[...,1])
                else:
                     initial_rf_waveforms = torch.complex(initial_rf_waveforms, torch.zeros_like(initial_rf_waveforms))
            if initial_rf_waveforms.shape != (self.num_channels, self.N_timepoints):
                raise ValueError(f"initial_rf_waveforms must have shape ({self.num_channels}, {self.N_timepoints}).")
            self.rf_waveforms_per_channel = nn.Parameter(initial_rf_waveforms.clone().to(device=self.device, dtype=torch.complex64))
        elif self.rf_waveforms_per_channel is None:
            rf_real = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            rf_imag = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            self.rf_waveforms_per_channel = nn.Parameter(torch.complex(rf_real, rf_imag))

        if initial_gradient_waveforms_tm is not None:
            if initial_gradient_waveforms_tm.shape != (self.N_timepoints, 3):
                raise ValueError(f"initial_gradient_waveforms_tm must have shape ({self.N_timepoints}, 3).")
            self.gradient_waveforms_tm = nn.Parameter(initial_gradient_waveforms_tm.clone().to(device=self.device, dtype=self.dtype))
        elif self.gradient_waveforms_tm is None:
            self.gradient_waveforms_tm = nn.Parameter(torch.randn(self.N_timepoints, 3, device=self.device, dtype=self.dtype) * 1e-5)

        target_M_on_device = self.target_magnetization_profile.to(self.device)
        parameters_to_optimize = [self.rf_waveforms_per_channel, self.gradient_waveforms_tm]
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(parameters_to_optimize, lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS(parameters_to_optimize, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'Adam' or 'LBFGS'.")

        b1_maps_subject_dev = b1_maps_subject_tesla.to(self.device)
        b0_map_subject_dev = b0_map_subject_hz.to(self.device)
        t1_map_dev = tissue_properties_subject['T1'].to(self.device, dtype=self.dtype)
        t2_map_dev = tissue_properties_subject['T2'].to(self.device, dtype=self.dtype)

        expected_b1_shape = (self.num_channels, self.Nx, self.Ny, self.Nz)
        expected_map_shape = (self.Nx, self.Ny, self.Nz)
        if b1_maps_subject_dev.shape != expected_b1_shape: raise ValueError(f"B1 maps shape error.")
        if b0_map_subject_dev.shape != expected_map_shape: raise ValueError(f"B0 map shape error.")
        if t1_map_dev.shape != expected_map_shape: raise ValueError(f"T1 map shape error.")
        if t2_map_dev.shape != expected_map_shape: raise ValueError(f"T2 map shape error.")

        initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=self.dtype)
        initial_M_subject[..., 2] = 1.0

        for i in range(num_iterations):
            cost_val = 0.0
            def closure(): # Used by LBFGS
                optimizer.zero_grad()
                b1_total_field_sq_t_v = None
                sim_outputs = bloch_simulate_ptx(
                    rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                    b1_sensitivity_maps=b1_maps_subject_dev, dt_s=self.dt_s, b0_map_hz=b0_map_subject_dev,
                    T1_map_s=t1_map_dev, T2_map_s=t2_map_dev,
                    gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t, initial_magnetization=initial_M_subject,
                    spatial_grid_m=self.spatial_grid_m, gradient_waveforms_tm=self.gradient_waveforms_tm,
                    return_all_timepoints=False,
                    return_b1_total_field_sq_t_v=(self.sar_constraint_mode == 'local_b1_sq_peak'))
                if self.sar_constraint_mode == 'local_b1_sq_peak':
                    final_magnetization_profile, b1_total_field_sq_t_v = sim_outputs
                else:
                    final_magnetization_profile = sim_outputs

                current_mag_cost = self._calculate_magnetization_error_cost(
                    final_magnetization_profile, target_M_on_device, cost_mask, cost_component_weights)

                additional_penalty = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                if self.sar_constraint_mode == 'local_b1_sq_peak':
                    if b1_total_field_sq_t_v is None: raise RuntimeError("B1 field not returned for SAR.")
                    peak_b1_sq = torch.max(b1_total_field_sq_t_v)
                    additional_penalty += self.sar_lambda * torch.relu(peak_b1_sq - self.local_b1_sq_limit_tesla_sq)**2
                elif self.sar_constraint_mode == 'total_rf_power': # elif ensures modes are exclusive for sar_lambda use
                    current_rf_power = torch.sum(torch.abs(self.rf_waveforms_per_channel)**2) * self.dt_s
                    additional_penalty += self.sar_lambda * torch.relu(current_rf_power - self.total_rf_power_limit)**2

                if self.constrain_gradient_power:
                    current_grad_power = torch.sum(self.gradient_waveforms_tm**2) * self.dt_s
                    additional_penalty += self.gradient_power_lambda * torch.relu(current_grad_power - self.total_gradient_power_limit)**2

                total_cost = current_mag_cost + additional_penalty
                total_cost.backward()
                return total_cost

            if optimizer_type.lower() == 'lbfgs':
                loss = optimizer.step(closure)
                cost_val = loss.item()
            else: # Adam
                cost_val = closure().item() # Call closure to compute cost and grads
                optimizer.step() # Adam step

            print(f"Iteration {i+1}/{num_iterations}, Cost (Mag [+SAR/Grad]): {cost_val:.6e}")
        return self.rf_waveforms_per_channel.detach().clone(), self.gradient_waveforms_tm.detach().clone()

if __name__ == '__main__':
    print("SpokesPulseDesigner class defined.")
    num_channels = 2
    pulse_duration = 0.5e-3
    dt_s_val = 4e-6
    n_vox_x, n_vox_y, n_vox_z = 4, 4, 1
    device = 'cpu'

    coords = [torch.linspace(-0.05, 0.05, n_vox_x), torch.linspace(-0.05, 0.05, n_vox_y), torch.tensor([0.0])]
    grid_x, grid_y, grid_z = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(device)

    target_M = torch.zeros((n_vox_x, n_vox_y, n_vox_z, 3), device=device)
    target_M[..., 2] = 1.0
    center_x, center_y = n_vox_x // 2, n_vox_y // 2
    radius_vox = n_vox_x // 4
    for r_idx in range(n_vox_x):
        for c_idx in range(n_vox_y):
            if (r_idx - center_x)**2 + (c_idx - center_y)**2 < radius_vox**2:
                target_M[r_idx, c_idx, 0, 0] = 0.707
                target_M[r_idx, c_idx, 0, 1] = 0.707
                target_M[r_idx, c_idx, 0, 2] = 0.0

    try:
        designer = SpokesPulseDesigner(
            num_channels=num_channels, pulse_duration_s=pulse_duration, dt_s=dt_s_val,
            spatial_grid_m=spatial_grid, target_magnetization_profile=target_M, device=device,
            sar_constraint_mode='local_b1_sq_peak', sar_lambda=1e8,
            local_b1_sq_limit_tesla_sq=(2e-6)**2,
            constrain_gradient_power=True, gradient_power_lambda=1e3,
            total_gradient_power_limit=1e-5
        )
        print(f"SpokesPulseDesigner initialized with constraints: {designer.N_timepoints} timepoints.")

        dummy_b1 = (torch.rand(num_channels, n_vox_x, n_vox_y, n_vox_z, dtype=torch.complex64, device=device) * 1e-7) + 1e-8
        dummy_b0 = (torch.rand(n_vox_x, n_vox_y, n_vox_z, device=device) - 0.5) * 50
        dummy_t1 = torch.ones(n_vox_x, n_vox_y, n_vox_z, device=device) * 0.8
        dummy_t2 = torch.ones(n_vox_x, n_vox_y, n_vox_z, device=device) * 0.05
        dummy_tissue = {'T1': dummy_t1, 'T2': dummy_t2}

        opt_rf, opt_grad = designer.design_spokes_pulse(
            b1_maps_subject_tesla=dummy_b1, b0_map_subject_hz=dummy_b0,
            tissue_properties_subject=dummy_tissue, num_iterations=3, learning_rate=0.01 # Reduced iterations

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
