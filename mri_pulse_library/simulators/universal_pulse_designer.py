import torch
import torch.nn as nn
import torch.optim as optim
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
from mri_pulse_library.core.bloch_sim import bloch_simulate_ptx

class UniversalPulseDesigner:
    """
    Designs universal parallel transmit (pTx) RF pulse waveforms.

    This class uses an optimization process to create RF waveforms for multiple
    transmit channels that aim to achieve a target flip angle profile across a
    specified spatial grid, while being robust to variations typically found
    across different subjects (e.g., B1 sensitivity and B0 off-resonance maps).
    """
    def __init__(self,
                 num_channels: int,
                 pulse_duration_s: float,
                 dt_s: float,
                 spatial_grid_m: torch.Tensor, # Shape (Nx, Ny, Nz, 3)
                 target_flip_angle_profile_rad: torch.Tensor, # Shape (Nx, Ny, Nz)
                 gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                 device: str = 'cpu'):
        """
        Initializes the UniversalPulseDesigner.

        Args:
            num_channels (int): Number of transmit channels.
            pulse_duration_s (float): Total duration of the RF pulse in seconds.
            dt_s (float): Time step for simulation in seconds.
            spatial_grid_m (torch.Tensor): Defines the spatial coordinates for each voxel.
                                           Shape: (Nx, Ny, Nz, 3), Units: meters.
            target_flip_angle_profile_rad (torch.Tensor): Desired flip angle at each voxel.
                                                      Shape: (Nx, Ny, Nz), Units: radians.
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

        if not isinstance(spatial_grid_m, torch.Tensor) or spatial_grid_m.ndim != 4 or spatial_grid_m.shape[-1] != 3:
            raise ValueError("spatial_grid_m must be a torch.Tensor of shape (Nx, Ny, Nz, 3).")
        self.spatial_grid_m = spatial_grid_m.to(self.device)

        self.Nx, self.Ny, self.Nz, _ = self.spatial_grid_m.shape

        expected_target_shape = (self.Nx, self.Ny, self.Nz)
        if not isinstance(target_flip_angle_profile_rad, torch.Tensor) or target_flip_angle_profile_rad.shape != expected_target_shape:
            raise ValueError(f"target_flip_angle_profile_rad must be a torch.Tensor of shape {expected_target_shape}.")
        self.target_flip_angle_profile_rad = target_flip_angle_profile_rad.to(self.device)

        self.gyromagnetic_ratio_hz_t = float(gyromagnetic_ratio_hz_t)

        self.N_timepoints = int(round(self.pulse_duration_s / self.dt_s))
        if self.N_timepoints <= 0:
            raise ValueError("Number of timepoints must be positive. Check pulse_duration_s and dt_s.")

        # Initialize RF waveforms as None; they will be created in the design method
        self.rf_waveforms_per_channel = None # Shape: (num_channels, N_timepoints), complex

    def _calculate_flip_angle(self, magnetization_xyz: torch.Tensor) -> torch.Tensor:
        """
        Calculates the flip angle from the magnetization vector [Mx, My, Mz].

        Args:
            magnetization_xyz (torch.Tensor): Magnetization vector(s).
                                             Shape: (..., 3).

        Returns:
            torch.Tensor: Flip angle(s) in radians. Shape: (...).
        """
        if magnetization_xyz.shape[-1] != 3:
            raise ValueError("Last dimension of magnetization_xyz must be 3 (Mx, My, Mz).")

        # Magnitude of transverse magnetization (Mxy)
        m_xy = torch.norm(magnetization_xyz[..., :2], dim=-1, p=2)
        m_z = magnetization_xyz[..., 2]

        flip_angle_rad = torch.atan2(m_xy, m_z)
        return flip_angle_rad

    def design_pulse(self,
                     b1_maps_database: list, # List of tensors (num_subj, num_channels, Nx, Ny, Nz) Tesla
                     b0_maps_database: list, # List of tensors (num_subj, Nx, Ny, Nz) Hz
                     tissue_properties_database: list, # List of dicts [{'T1': T1_map, 'T2': T2_map}]
                     initial_rf_waveforms: torch.Tensor = None, # Optional starting point
                     num_iterations: int = 10,
                     learning_rate: float = 0.01,
                     optimizer_type: str = 'Adam'):
        """
        Designs the universal RF pulse waveforms using an optimization process.
        (This is a placeholder and will be implemented in detail in subsequent steps)

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
            real_part = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=torch.float32) * 0.01
            imag_part = torch.randn(self.num_channels, self.N_timepoints, device=self.device, dtype=torch.float32) * 0.01
            self.rf_waveforms_per_channel = nn.Parameter(torch.complex(real_part, imag_part))
        else:
            # Ensure initial_rf_waveforms is complex and on the correct device
            if not torch.is_complex(initial_rf_waveforms):
                 # Attempt to cast if it's real, assuming it should be complex
                 initial_rf_waveforms = initial_rf_waveforms.to(dtype=torch.complex64)
            if initial_rf_waveforms.shape != (self.num_channels, self.N_timepoints):
                raise ValueError(f"initial_rf_waveforms must have shape ({self.num_channels}, {self.N_timepoints}).")
            self.rf_waveforms_per_channel = nn.Parameter(initial_rf_waveforms.clone().to(self.device))

        # Ensure target flip angle is on the correct device
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

        for i in range(num_iterations):
            def closure_lbfgs(): # For LBFGS
                optimizer.zero_grad()
                current_total_cost = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for subj_idx in range(num_subjects):
                    subject_b1_map = b1_maps_database[subj_idx].to(self.device)
                    subject_b0_map = b0_maps_database[subj_idx].to(self.device)
                    subject_tissue_props = tissue_properties_database[subj_idx]
                    subject_t1_map = subject_tissue_props['T1'].to(self.device)
                    subject_t2_map = subject_tissue_props['T2'].to(self.device)

                    # Basic shape validation for subject maps
                    expected_b1_shape = (self.num_channels, self.Nx, self.Ny, self.Nz)
                    expected_map_shape = (self.Nx, self.Ny, self.Nz)
                    if subject_b1_map.shape != expected_b1_shape: raise ValueError(f"Subj {subj_idx} B1 map shape error.")
                    if subject_b0_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} B0 map shape error.")
                    if subject_t1_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} T1 map shape error.")
                    if subject_t2_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} T2 map shape error.")

                    initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=torch.float32)
                    initial_M_subject[..., 2] = 1.0

                    final_magnetization = bloch_simulate_ptx(
                        rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                        b1_sensitivity_maps=subject_b1_map,
                        dt_s=self.dt_s,
                        b0_map_hz=subject_b0_map,
                        T1_map_s=subject_t1_map,
                        T2_map_s=subject_t2_map,
                        gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t,
                        initial_magnetization=initial_M_subject,
                        return_all_timepoints=False
                    )
                    simulated_fa_subject = self._calculate_flip_angle(final_magnetization)
                    cost_subject = torch.mean((simulated_fa_subject - target_fa_on_device)**2)
                    current_total_cost += cost_subject

                avg_cost_lbfgs_closure = current_total_cost / num_subjects
                avg_cost_lbfgs_closure.backward()
                return avg_cost_lbfgs_closure

            if optimizer_type.lower() == 'lbfgs':
                avg_cost = optimizer.step(closure_lbfgs) # LBFGS step returns loss
            else: # Adam
                optimizer.zero_grad()
                current_total_cost = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for subj_idx in range(num_subjects):
                    subject_b1_map = b1_maps_database[subj_idx].to(self.device)
                    subject_b0_map = b0_maps_database[subj_idx].to(self.device)
                    subject_tissue_props = tissue_properties_database[subj_idx]
                    subject_t1_map = subject_tissue_props['T1'].to(self.device)
                    subject_t2_map = subject_tissue_props['T2'].to(self.device)

                    # Basic shape validation (can be refactored to avoid repetition if LBFGS structure changes)
                    expected_b1_shape = (self.num_channels, self.Nx, self.Ny, self.Nz)
                    expected_map_shape = (self.Nx, self.Ny, self.Nz)
                    if subject_b1_map.shape != expected_b1_shape: raise ValueError(f"Subj {subj_idx} B1 map shape error.")
                    if subject_b0_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} B0 map shape error.")
                    if subject_t1_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} T1 map shape error.")
                    if subject_t2_map.shape != expected_map_shape: raise ValueError(f"Subj {subj_idx} T2 map shape error.")

                    initial_M_subject = torch.zeros((self.Nx, self.Ny, self.Nz, 3), device=self.device, dtype=torch.float32)
                    initial_M_subject[..., 2] = 1.0

                    final_magnetization = bloch_simulate_ptx(
                        rf_waveforms_per_channel=self.rf_waveforms_per_channel,
                        b1_sensitivity_maps=subject_b1_map,
                        dt_s=self.dt_s,
                        b0_map_hz=subject_b0_map,
                        T1_map_s=subject_t1_map,
                        T2_map_s=subject_t2_map,
                        gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t,
                        initial_magnetization=initial_M_subject,
                        return_all_timepoints=False
                    )
                    simulated_fa_subject = self._calculate_flip_angle(final_magnetization)
                    cost_subject = torch.mean((simulated_fa_subject - target_fa_on_device)**2)
                    current_total_cost += cost_subject

                avg_cost = current_total_cost / num_subjects
                avg_cost.backward()
                optimizer.step()

            print(f"Iteration {i+1}/{num_iterations}, Avg Cost: {avg_cost.item()}")

        return self.rf_waveforms_per_channel.detach().clone()

if __name__ == '__main__':
    # Example Usage (Illustrative)
    print("UniversalPulseDesigner class defined.")
    # Parameters
    n_ch = 2
    dur_s = 2e-3 # Shorter duration for faster example
    dt = 4e-6
    n_vox_x, n_vox_y, n_vox_z = 8, 8, 1 # Smaller grid for faster example

    # Create dummy spatial grid and target flip angle profile
    x_coords = torch.linspace(-0.1, 0.1, n_vox_x)
    y_coords = torch.linspace(-0.1, 0.1, n_vox_y)
    z_coords = torch.tensor([0.0]) # Single slice for simplicity

    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    spatial_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1) # Shape (Nx, Ny, Nz, 3)

    # Target: uniform 90-degree flip angle (pi/2)
    target_fa_rad = torch.ones((n_vox_x, n_vox_y, n_vox_z)) * (torch.pi / 2)

    try:
        designer = UniversalPulseDesigner(
            num_channels=n_ch,
            pulse_duration_s=dur_s,
            dt_s=dt,
            spatial_grid_m=spatial_grid,
            target_flip_angle_profile_rad=target_fa_rad,
            device='cpu'
        )
        print(f"Designer initialized: {designer.N_timepoints} timepoints")

        # Dummy data for design_pulse
        num_subjects = 1
        dummy_b1_maps = [torch.rand(n_ch, n_vox_x, n_vox_y, n_vox_z, dtype=torch.complex64) * 1e-6 for _ in range(num_subjects)] # Tesla
        dummy_b0_maps = [torch.rand(n_vox_x, n_vox_y, n_vox_z) * 10 for _ in range(num_subjects)] # Hz
        dummy_t1 = torch.ones(n_vox_x, n_vox_y, n_vox_z) * 1.0 # s
        dummy_t2 = torch.ones(n_vox_x, n_vox_y, n_vox_z) * 0.1 # s
        dummy_tissue_props = [{'T1': dummy_t1, 'T2': dummy_t2} for _ in range(num_subjects)]

        # Call design_pulse to get optimized RF waveforms
        optimized_rf_waveforms = designer.design_pulse(
            b1_maps_database=dummy_b1_maps,
            b0_maps_database=dummy_b0_maps,
            tissue_properties_database=dummy_tissue_props,
            num_iterations=5, # Small number of iterations for quick test
            learning_rate=0.01,
            optimizer_type='Adam' # or 'LBFGS'
        )
        print(f"design_pulse finished.")
        print(f"Optimized RF waveforms shape: {optimized_rf_waveforms.shape}")
        print(f"Mean absolute value of optimized RF: {torch.mean(torch.abs(optimized_rf_waveforms)).item()}")

        # Optionally, test with LBFGS (can be slower for this type of problem if not tuned)
        # print("\nTesting with LBFGS...")
        # designer_lbfgs = UniversalPulseDesigner(
        #     num_channels=n_ch,
        #     pulse_duration_s=dur_s,
        #     dt_s=dt,
        #     spatial_grid_m=spatial_grid,
        #     target_flip_angle_profile_rad=target_fa_rad,
        #     device='cpu'
        # )
        # optimized_rf_lbfgs = designer_lbfgs.design_pulse(
        #     b1_maps_database=dummy_b1_maps,
        #     b0_maps_database=dummy_b0_maps,
        #     tissue_properties_database=dummy_tissue_props,
        #     num_iterations=3, # Very few iterations for LBFGS as it can be slow per iter
        #     learning_rate=0.1, # LBFGS often needs different LR
        #     optimizer_type='LBFGS'
        # )
        # print(f"LBFGS design_pulse finished.")
        # print(f"Optimized RF waveforms shape (LBFGS): {optimized_rf_lbfgs.shape}")
        # print(f"Mean absolute value of optimized RF (LBFGS): {torch.mean(torch.abs(optimized_rf_lbfgs)).item()}")

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
