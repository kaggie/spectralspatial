import torch
import torch.nn as nn
import torch.optim as optim
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
from mri_pulse_library.core.bloch_sim import bloch_simulate_ptx

class UniversalPulseDesigner:
    def __init__(self,
                 num_channels: int,
                 pulse_duration_s: float,
                 dt_s: float,
                 spatial_grid_m: torch.Tensor,
                 target_flip_angle_profile_rad: torch.Tensor,
                 gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                 device: str = 'cpu',
                 sar_constraint_mode: str = 'none',
                 sar_lambda: float = 1.0,
                 local_b1_sq_limit_tesla_sq: float = None,
                 total_rf_power_limit: float = None):
        """
        Initializes the UniversalPulseDesigner.

        Args:
            num_channels (int): Number of transmit channels.
            pulse_duration_s (float): Total duration of the RF pulse in seconds.
            dt_s (float): Time step for simulation in seconds.
            spatial_grid_m (torch.Tensor): Defines spatial coordinates (x,y,z) for each voxel.
                                           Shape: (Nx, Ny, Nz, 3), Units: meters.
            target_flip_angle_profile_rad (torch.Tensor): Desired flip angle at each voxel.
                                                      Shape: (Nx, Ny, Nz), Units: radians.
            gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio.
                                                      Defaults to GAMMA_HZ_PER_T_PROTON.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.
            sar_constraint_mode (str, optional): 'none', 'local_b1_sq_peak', or 'total_rf_power'.
                                                 Defaults to 'none'.
            sar_lambda (float, optional): Penalty weight for SAR constraints. Defaults to 1.0.
            local_b1_sq_limit_tesla_sq (float, optional): Limit for peak local |B1(t,v)|^2 (Tesla^2).
                                                          Required if mode is 'local_b1_sq_peak'.
            total_rf_power_limit (float, optional): Limit for sum_channels sum_time |RF_ch(t)|^2 * dt_s.
                                                    Required if mode is 'total_rf_power'.
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
        self.dtype = torch.float32

        self.device = torch.device(device)

        if not isinstance(spatial_grid_m, torch.Tensor) or spatial_grid_m.ndim != 4 or spatial_grid_m.shape[-1] != 3:
            raise ValueError("spatial_grid_m must be a torch.Tensor of shape (Nx, Ny, Nz, 3).")
        self.spatial_grid_m = spatial_grid_m.to(self.device, dtype=self.dtype)

        self.Nx, self.Ny, self.Nz, _ = self.spatial_grid_m.shape

        expected_target_shape = (self.Nx, self.Ny, self.Nz)
        if not isinstance(target_flip_angle_profile_rad, torch.Tensor) or target_flip_angle_profile_rad.shape != expected_target_shape:
            raise ValueError(f"target_flip_angle_profile_rad must be a torch.Tensor of shape {expected_target_shape}.")
        self.target_flip_angle_profile_rad = target_flip_angle_profile_rad.to(self.device, dtype=self.dtype)

        self.gyromagnetic_ratio_hz_t = float(gyromagnetic_ratio_hz_t)

        self.N_timepoints = int(round(self.pulse_duration_s / self.dt_s))
        if self.N_timepoints <= 0:
            raise ValueError("Number of timepoints must be positive. Check pulse_duration_s and dt_s.")

        self.rf_waveforms_per_channel = None

        self.sar_constraint_mode = sar_constraint_mode.lower()
        self.sar_lambda = sar_lambda
        self.local_b1_sq_limit_tesla_sq = local_b1_sq_limit_tesla_sq
        self.total_rf_power_limit = total_rf_power_limit

        if self.sar_constraint_mode == 'local_b1_sq_peak' and self.local_b1_sq_limit_tesla_sq is None:
            raise ValueError("local_b1_sq_limit_tesla_sq must be provided for 'local_b1_sq_peak' SAR mode.")
        if self.sar_constraint_mode == 'total_rf_power' and self.total_rf_power_limit is None:
            raise ValueError("total_rf_power_limit must be provided for 'total_rf_power' SAR mode.")
        if self.sar_constraint_mode not in ['none', 'local_b1_sq_peak', 'total_rf_power']:
            raise ValueError(f"Invalid sar_constraint_mode: {self.sar_constraint_mode}")

    def _calculate_flip_angle(self, magnetization_xyz: torch.Tensor) -> torch.Tensor:
        if magnetization_xyz.shape[-1] != 3:
            raise ValueError("Last dimension of magnetization_xyz must be 3 (Mx, My, Mz).")
        m_xy = torch.norm(magnetization_xyz[..., :2], dim=-1, p=2)
        m_z = magnetization_xyz[..., 2]

        flip_angle_rad = torch.atan2(m_xy, m_z)
        return flip_angle_rad

    def design_pulse(self,
                     b1_maps_database: list,
                     b0_maps_database: list,
                     tissue_properties_database: list,
                     initial_rf_waveforms: torch.Tensor = None,
                     num_iterations: int = 10,
                     learning_rate: float = 0.01,
                     optimizer_type: str = 'Adam'):
        """
        Designs the universal RF pulse waveforms using an optimization process.

        Args:
            b1_maps_database (list): List of B1+ sensitivity maps (Tesla) for each subject/dataset.
                                     Each element is a tensor of shape (num_channels, Nx, Ny, Nz).
            b0_maps_database (list): List of B0 off-resonance maps (Hz) for each subject/dataset.
                                     Each element is a tensor of shape (Nx, Ny, Nz).
            tissue_properties_database (list): List of tissue property maps for each subject/dataset.
                                              Each element is a dict {'T1': T1_map (s), 'T2': T2_map (s)},
                                              where maps are tensors of shape (Nx, Ny, Nz).
            initial_rf_waveforms (torch.Tensor, optional): Initial guess for RF waveforms.
                                                          Must be a complex tensor.
                                                          Shape: (num_channels, N_timepoints).
                                                          If None, random initialization is used.
            num_iterations (int, optional): Number of optimization iterations. Defaults to 10.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            optimizer_type (str, optional): Type of optimizer ('Adam', 'LBFGS', etc.). Defaults to 'Adam'.

        Returns:
            torch.Tensor: Optimized RF waveforms. Shape: (num_channels, N_timepoints), complex.
        """
        if initial_rf_waveforms is None:
            real_part = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            imag_part = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=self.dtype) * 0.01
            self.rf_waveforms_per_channel = nn.Parameter(torch.complex(real_part, imag_part))
        else:
            if not torch.is_complex(initial_rf_waveforms):
                 initial_rf_waveforms = initial_rf_waveforms.to(dtype=torch.complex64)
            if initial_rf_waveforms.shape != (self.num_channels, self.N_timepoints):
                raise ValueError(f"initial_rf_waveforms must have shape ({self.num_channels}, {self.N_timepoints}).")
            self.rf_waveforms_per_channel = nn.Parameter(initial_rf_waveforms.clone().to(self.device))

        target_fa_on_device = self.target_flip_angle_profile_rad.to(self.device)

        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam([self.rf_waveforms_per_channel], lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS([self.rf_waveforms_per_channel], lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'Adam' or 'LBFGS'.")

        if not b1_maps_database or not b0_maps_database or not tissue_properties_database:
            raise ValueError("Databases cannot be empty.")
        if not (len(b1_maps_database) == len(b0_maps_database) == len(tissue_properties_database)):
            raise ValueError("All databases must have the same number of subject entries.")
        num_subjects = len(b1_maps_database)

        expected_b1_shape = (self.num_channels, self.Nx, self.Ny, self.Nz)
        expected_map_shape = (self.Nx, self.Ny, self.Nz)

        def closure_lbfgs():
            optimizer.zero_grad()
            current_total_cost_lbfgs = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for subj_idx_lbfgs in range(num_subjects):
                subject_b1_map = b1_maps_database[subj_idx_lbfgs].to(self.device)
                subject_b0_map = b0_maps_database[subj_idx_lbfgs].to(self.device)
                subject_tissue_props = tissue_properties_database[subj_idx_lbfgs]
                subject_t1_map = subject_tissue_props['T1'].to(self.device, dtype=self.dtype)
                subject_t2_map = subject_tissue_props['T2'].to(self.device, dtype=self.dtype)

                if subject_b1_map.shape != expected_b1_shape: raise ValueError(f"LBFGS: Subj {subj_idx_lbfgs} B1 map shape error.")
                if subject_b0_map.shape != expected_map_shape: raise ValueError(f"LBFGS: Subj {subj_idx_lbfgs} B0 map shape error.")
                if subject_t1_map.shape != expected_map_shape: raise ValueError(f"LBFGS: Subj {subj_idx_lbfgs} T1 map shape error.")
                if subject_t2_map.shape != expected_map_shape: raise ValueError(f"LBFGS: Subj {subj_idx_lbfgs} T2 map shape error.")

                initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=self.dtype)
                initial_M_subject[..., 2] = 1.0

                b1_total_field_sq_t_v_subject_lbfgs = None
                bloch_sim_outputs_lbfgs = bloch_simulate_ptx(
                    rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                    b1_sensitivity_maps=subject_b1_map, dt_s=self.dt_s, b0_map_hz=subject_b0_map,
                    T1_map_s=subject_t1_map, T2_map_s=subject_t2_map,
                    gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t, initial_magnetization=initial_M_subject,
                    spatial_grid_m=self.spatial_grid_m, gradient_waveforms_tm=None, return_all_timepoints=False,
                    return_b1_total_field_sq_t_v=(self.sar_constraint_mode == 'local_b1_sq_peak'))
                if self.sar_constraint_mode == 'local_b1_sq_peak':
                    final_magnetization_lbfgs, b1_total_field_sq_t_v_subject_lbfgs = bloch_sim_outputs_lbfgs
                else:
                    final_magnetization_lbfgs = bloch_sim_outputs_lbfgs

                simulated_fa_subject_lbfgs = self._calculate_flip_angle(final_magnetization_lbfgs)
                cost_subject_lbfgs = torch.mean((simulated_fa_subject_lbfgs - target_fa_on_device)**2)
                subject_sar_penalty_lbfgs = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                if self.sar_constraint_mode == 'local_b1_sq_peak':
                    if b1_total_field_sq_t_v_subject_lbfgs is None: raise RuntimeError("B1 field not returned for LBFGS SAR.")
                    peak_b1_sq_subj_lbfgs = torch.max(b1_total_field_sq_t_v_subject_lbfgs)
                    subject_sar_penalty_lbfgs = self.sar_lambda * torch.relu(peak_b1_sq_subj_lbfgs - self.local_b1_sq_limit_tesla_sq)**2
                current_total_cost_lbfgs += (cost_subject_lbfgs + subject_sar_penalty_lbfgs)

            avg_cost_lbfgs = current_total_cost_lbfgs / num_subjects
            if self.sar_constraint_mode == 'total_rf_power':
                total_rf_power = torch.sum(torch.abs(self.rf_waveforms_per_channel)**2) * self.dt_s
                power_penalty_lbfgs = self.sar_lambda * torch.relu(total_rf_power - self.total_rf_power_limit)**2
                avg_cost_lbfgs += power_penalty_lbfgs
            avg_cost_lbfgs.backward()
            return avg_cost_lbfgs

        for i in range(num_iterations):
            cost_val_iter = 0.0
            if optimizer_type.lower() == 'lbfgs':
                loss = optimizer.step(closure_lbfgs)
                cost_val_iter = loss.item()
            else: # Adam
                optimizer.zero_grad()
                current_total_cost_adam = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for subj_idx_adam in range(num_subjects):
                    subject_b1_map = b1_maps_database[subj_idx_adam].to(self.device)
                    subject_b0_map = b0_maps_database[subj_idx_adam].to(self.device)
                    subject_tissue_props = tissue_properties_database[subj_idx_adam]
                    subject_t1_map = subject_tissue_props['T1'].to(self.device, dtype=self.dtype)
                    subject_t2_map = subject_tissue_props['T2'].to(self.device, dtype=self.dtype)

                    if subject_b1_map.shape != expected_b1_shape: raise ValueError(f"Adam: Subj {subj_idx_adam} B1 map shape error.")
                    if subject_b0_map.shape != expected_map_shape: raise ValueError(f"Adam: Subj {subj_idx_adam} B0 map shape error.")
                    if subject_t1_map.shape != expected_map_shape: raise ValueError(f"Adam: Subj {subj_idx_adam} T1 map shape error.")
                    if subject_t2_map.shape != expected_map_shape: raise ValueError(f"Adam: Subj {subj_idx_adam} T2 map shape error.")

                    initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=self.dtype)
                    initial_M_subject[..., 2] = 1.0
                    b1_total_field_sq_t_v_subject_adam = None
                    bloch_sim_outputs_adam = bloch_simulate_ptx(
                        rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                        b1_sensitivity_maps=subject_b1_map, dt_s=self.dt_s, b0_map_hz=subject_b0_map,
                        T1_map_s=subject_t1_map, T2_map_s=subject_t2_map,
                        gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t, initial_magnetization=initial_M_subject,
                        spatial_grid_m=self.spatial_grid_m, gradient_waveforms_tm=None, return_all_timepoints=False,
                        return_b1_total_field_sq_t_v=(self.sar_constraint_mode == 'local_b1_sq_peak'))
                    if self.sar_constraint_mode == 'local_b1_sq_peak':
                        final_magnetization_adam, b1_total_field_sq_t_v_subject_adam = bloch_sim_outputs_adam
                    else:
                        final_magnetization_adam = bloch_sim_outputs_adam
                    simulated_fa_subject_adam = self._calculate_flip_angle(final_magnetization_adam)
                    cost_subject_adam = torch.mean((simulated_fa_subject_adam - target_fa_on_device)**2)
                    subject_sar_penalty_adam = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    if self.sar_constraint_mode == 'local_b1_sq_peak':
                        if b1_total_field_sq_t_v_subject_adam is None: raise RuntimeError("B1 field not returned for Adam SAR.")
                        peak_b1_sq_subj_adam = torch.max(b1_total_field_sq_t_v_subject_adam)
                        subject_sar_penalty_adam = self.sar_lambda * torch.relu(peak_b1_sq_subj_adam - self.local_b1_sq_limit_tesla_sq)**2
                    current_total_cost_adam += (cost_subject_adam + subject_sar_penalty_adam)

                avg_cost_adam = current_total_cost_adam / num_subjects
                if self.sar_constraint_mode == 'total_rf_power':
                    total_rf_power = torch.sum(torch.abs(self.rf_waveforms_per_channel)**2) * self.dt_s
                    power_penalty_adam = self.sar_lambda * torch.relu(total_rf_power - self.total_rf_power_limit)**2
                    avg_cost_adam += power_penalty_adam
                avg_cost_adam.backward()
                optimizer.step()
                cost_val_iter = avg_cost_adam.item()

            print(f"Iteration {i+1}/{num_iterations}, Cost (FA [+SAR]): {cost_val_iter:.6e}")
        return self.rf_waveforms_per_channel.detach().clone()

if __name__ == '__main__':
    print("UniversalPulseDesigner class defined.")
    num_channels = 2
    pulse_duration = 2e-3
    dt = 4e-6
    device = 'cpu'
    Nx, Ny, Nz = 8, 8, 1
    x_coords = torch.linspace(-0.05, 0.05, Nx, device=device)
    y_coords = torch.linspace(-0.05, 0.05, Ny, device=device)
    z_coords = torch.tensor([0.0], device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    target_flip_angle_deg = 30.0
    target_fa_rad = torch.ones((Nx, Ny, Nz), device=device) * (target_flip_angle_deg * torch.pi / 180.0)
    try:
        designer = UniversalPulseDesigner(
            num_channels=num_channels, pulse_duration_s=pulse_duration, dt_s=dt,
            spatial_grid_m=spatial_grid, target_flip_angle_profile_rad=target_fa_rad, device=device,
            sar_constraint_mode='local_b1_sq_peak',
            sar_lambda=1e8,
            local_b1_sq_limit_tesla_sq=(3e-6)**2
        )
        print(f"Designer initialized with SAR: {designer.N_timepoints} timepoints, Mode: {designer.sar_constraint_mode}")

        designer_power_constrained = UniversalPulseDesigner(
            num_channels=num_channels, pulse_duration_s=pulse_duration, dt_s=dt,
            spatial_grid_m=spatial_grid, target_flip_angle_profile_rad=target_fa_rad, device=device,
            sar_constraint_mode='total_rf_power',
            sar_lambda=1e-2,
            total_rf_power_limit = (num_channels * designer.N_timepoints * (0.005)**2 * dt) # Example limit with dt
        )
        print(f"Designer initialized with SAR: {designer_power_constrained.N_timepoints} timepoints, Mode: {designer_power_constrained.sar_constraint_mode}")

        num_subjects = 1
        b1_maps_db = [(torch.rand(num_channels, Nx, Ny, Nz, dtype=torch.complex64, device=device)*1e-7 + 1e-8) for _ in range(num_subjects)]
        b0_maps_db = [(torch.rand(Nx,Ny,Nz,device=device)-0.5)*50 for _ in range(num_subjects)]
        tissue_props_db = [{'T1':torch.ones(Nx,Ny,Nz,device=device)*0.8,'T2':torch.ones(Nx,Ny,Nz,device=device)*0.08} for _ in range(num_subjects)]

        optimized_rf = designer.design_pulse(
            b1_maps_database=b1_maps_db, b0_maps_database=b0_maps_db,
            tissue_properties_database=tissue_props_db, num_iterations=3, learning_rate=0.01)
        print(f"design_pulse finished. RF shape: {optimized_rf.shape}, Mean abs: {torch.mean(torch.abs(optimized_rf)).item()}")

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
