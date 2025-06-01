import torch
import numpy as np # For np.eye if torch.eye needs specific dtype handling for complex
from mri_pulse_library.core.constants import GAMMA_RAD_PER_S_PER_T_PROTON, M0_PROTON

class STAPTxDesigner:
    """
    Designs multi-channel RF pulses for specified kT-points using the Small Tip Angle (STA)
    approximation to achieve a target transverse magnetization pattern.
    """
    def __init__(self, verbose: bool = True):
        """
        Initializes the STAPTxDesigner.

        Args:
            verbose (bool, optional): If True, prints design information and warnings.
                                      Defaults to True.
        """
        self.verbose = verbose
        # Effective gyromagnetic ratio for STA (gamma * M0 * dt)
        # M0 is often taken as 1.0 if target_mxy_pattern is already scaled or relative.
        # Here, self.M0 is available from constants.
        self.gamma_rad_s_t = GAMMA_RAD_PER_S_PER_T_PROTON # rad/s/T
        self.M0 = M0_PROTON # Typically 1.0

    def design_kta_pulse(
        self,
        b1_maps: torch.Tensor,
        target_mxy_pattern: torch.Tensor, # Should be complex Mx + iMy
        kt_points: torch.Tensor,
        spatial_grid_m: torch.Tensor,
        dt_per_kt_point: float,
        regularization_factor: float = 1e-3,
        target_spatial_mask: torch.Tensor = None
    ):
        """
        Designs RF pulse waveforms for each channel at each kT-point using STA.
        Solves: min || S * rf_vec - mxy_target_vec ||^2 + lambda * ||rf_vec||^2
        where S is the STA system matrix.

        Args:
            b1_maps (torch.Tensor): Complex B1+ sensitivity maps (Tesla/Volt or arbitrary units).
                                    Shape: (num_channels, Nx, Ny, Nz) or (num_channels, N_voxels_total).
                                    Must be on the same device as other tensors.
            target_mxy_pattern (torch.Tensor): Desired complex transverse magnetization (Mx + iMy).
                                               Shape: (Nx, Ny, Nz) or (N_voxels_total). Must be complex.
            kt_points (torch.Tensor): k-space points for the transmit trajectory (rad/m).
                                      Shape: (Num_kT_points, N_spatial_dims), where N_spatial_dims is typically 3.
            spatial_grid_m (torch.Tensor): Spatial coordinates (x,y,z) for each voxel (meters).
                                           Shape: (Nx, Ny, Nz, 3) or (N_voxels_total, 3).
            dt_per_kt_point (float): Effective duration/dwell time for each kT-point (seconds).
            regularization_factor (float, optional): Lambda for L2 regularization on RF amplitudes.
                                                     Defaults to 1e-3. Must be non-negative.
            target_spatial_mask (torch.Tensor, optional): Boolean mask defining the ROI.
                                                          Shape: (Nx, Ny, Nz) or (N_voxels_total).
                                                          If None, all voxels in target_mxy_pattern are used.
                                                          Defaults to None.

        Returns:
            torch.Tensor: Complex RF pulse values (units corresponding to B1_maps units, e.g., Volts)
                          for each channel at each kT-point.
                          Shape: (num_channels, Num_kT_points).
        """
        # --- Determine Device and Dtype ---
        device = b1_maps.device
        complex_dtype = b1_maps.dtype if b1_maps.is_complex() else torch.complex64
        real_dtype = torch.float32 # Default real dtype for intermediate calculations if needed

        # --- Input Validation ---
        if not isinstance(b1_maps, torch.Tensor) or not b1_maps.is_complex():
            raise ValueError("b1_maps must be a complex PyTorch Tensor.")
        if b1_maps.ndim not in [2, 4]: # (num_ch, Nvox) or (num_ch, Nx, Ny, Nz)
            raise ValueError("b1_maps must have 2 or 4 dimensions.")
        num_channels = b1_maps.shape[0]

        if not isinstance(target_mxy_pattern, torch.Tensor) or not target_mxy_pattern.is_complex():
            raise ValueError("target_mxy_pattern must be a complex PyTorch Tensor.")
        if target_mxy_pattern.ndim not in [1, 3]: # (Nvox) or (Nx, Ny, Nz)
            raise ValueError("target_mxy_pattern must have 1 or 3 dimensions.")

        if not isinstance(kt_points, torch.Tensor) or kt_points.ndim != 2:
            raise ValueError("kt_points must be a 2D PyTorch Tensor (Num_kT_points, N_spatial_dims).")
        num_kt_points = kt_points.shape[0]
        num_spatial_dims_k = kt_points.shape[1]

        if not isinstance(spatial_grid_m, torch.Tensor) or spatial_grid_m.ndim not in [2, 4]:
            raise ValueError("spatial_grid_m must be a 2D or 4D PyTorch Tensor.")
        if spatial_grid_m.shape[-1] != num_spatial_dims_k:
            raise ValueError(f"Last dimension of spatial_grid_m ({spatial_grid_m.shape[-1]}) must match " +
                             f"last dimension of kt_points ({num_spatial_dims_k}).")

        if dt_per_kt_point <= 0:
            raise ValueError("dt_per_kt_point must be positive.")
        if regularization_factor < 0:
            raise ValueError("regularization_factor cannot be negative.")

        # --- Prepare Data (Reshape, Mask, Select ROI) ---
        original_spatial_shape = None
        if b1_maps.ndim == 4: # (num_ch, Nx, Ny, Nz)
            original_spatial_shape = b1_maps.shape[1:]
            b1_maps_flat = b1_maps.reshape(num_channels, -1) # (num_ch, N_vox_total)
        else: # (num_ch, N_vox_total)
            b1_maps_flat = b1_maps

        N_vox_total = b1_maps_flat.shape[1]

        if target_mxy_pattern.ndim == 3: # (Nx, Ny, Nz)
            if original_spatial_shape and target_mxy_pattern.shape != original_spatial_shape:
                raise ValueError("Spatial shape of target_mxy_pattern must match b1_maps.")
            target_mxy_flat = target_mxy_pattern.reshape(-1)
        else: # (N_vox_total)
            target_mxy_flat = target_mxy_pattern
        if target_mxy_flat.shape[0] != N_vox_total:
            raise ValueError("Total number of voxels in target_mxy_pattern must match b1_maps.")

        if spatial_grid_m.ndim == 4: # (Nx, Ny, Nz, N_spatial_dims)
            if original_spatial_shape and spatial_grid_m.shape[:3] != original_spatial_shape:
                raise ValueError("Spatial shape of spatial_grid_m must match b1_maps.")
            spatial_grid_flat = spatial_grid_m.reshape(-1, num_spatial_dims_k) # (N_vox_total, N_spatial_dims)
        else: # (N_vox_total, N_spatial_dims)
            spatial_grid_flat = spatial_grid_m
        if spatial_grid_flat.shape[0] != N_vox_total:
            raise ValueError("Total number of voxels in spatial_grid_m must match b1_maps.")

        if target_spatial_mask is None:
            mask_flat = torch.ones(N_vox_total, dtype=torch.bool, device=device)
        elif target_spatial_mask.ndim == 3: # (Nx, Ny, Nz)
            if original_spatial_shape and target_spatial_mask.shape != original_spatial_shape:
                raise ValueError("Spatial shape of target_spatial_mask must match b1_maps.")
            mask_flat = target_spatial_mask.reshape(-1).to(dtype=torch.bool)
        else: # (N_vox_total)
            mask_flat = target_spatial_mask.to(dtype=torch.bool)
        if mask_flat.shape[0] != N_vox_total:
            raise ValueError("Total number of voxels in target_spatial_mask must match b1_maps.")

        roi_indices = torch.where(mask_flat)[0]
        if len(roi_indices) == 0:
            if self.verbose:
                self._log("Warning: target_spatial_mask is empty. Returning zero RF waveforms.")
            return torch.zeros((num_channels, num_kt_points), dtype=complex_dtype, device=device)

        b1_maps_roi = b1_maps_flat[:, roi_indices]
        target_mxy_roi = target_mxy_flat[roi_indices]
        spatial_grid_roi = spatial_grid_flat[roi_indices, :]
        N_vox_roi = len(roi_indices)

        if self.verbose:
            self._log(f"Designing for {N_vox_roi} voxels in ROI.")
            self._log(f"Number of channels: {num_channels}, Number of kT-points: {num_kt_points}")

        C_val = -1j * self.gamma_rad_s_t * self.M0 * dt_per_kt_point # M0 is used here
        C = torch.tensor(C_val, dtype=complex_dtype, device=device)

        phase_arg = torch.matmul(spatial_grid_roi, kt_points.T)
        phase_encoding_matrix = torch.exp(1j * phase_arg)

        S_v_ch_k = C * b1_maps_roi.T.unsqueeze(2) * phase_encoding_matrix.unsqueeze(1)
        S = S_v_ch_k.reshape(N_vox_roi, -1)

        S_H = S.conj().T
        S_H_S = S_H @ S

        lambda_val_tensor = torch.tensor(regularization_factor, dtype=real_dtype, device=device)

        identity_matrix = torch.eye(S.shape[1], device=device, dtype=complex_dtype)
        lhs = S_H_S + lambda_val_tensor * identity_matrix

        rhs = S_H @ target_mxy_roi.to(complex_dtype)

        try:
            rf_vec = torch.linalg.solve(lhs, rhs)
        except torch.linalg.LinAlgError: # PyTorch specific error for singular matrices
            if self.verbose:
                self._log("Warning: torch.linalg.solve failed. Using torch.linalg.lstsq on normal equations as fallback.")
            # lstsq on normal equations (A*x=b where A=lhs, x=rf_vec, b=rhs)
            # This is less stable than lstsq on original problem but a common fallback from direct solve.
            try:
                rf_vec_lstsq_result = torch.linalg.lstsq(lhs, rhs)
                rf_vec = rf_vec_lstsq_result.solution
                # TODO: Check rf_vec_lstsq_result.residuals if needed for convergence/quality
            except torch.linalg.LinAlgError:
                 if self.verbose:
                    self._log("Error: torch.linalg.lstsq on normal equations also failed. Returning zero RF waveforms.")
                 return torch.zeros((num_channels, num_kt_points), dtype=complex_dtype, device=device)

        rf_waveforms = rf_vec.reshape(num_channels, num_kt_points)

        if self.verbose:
            self._log("STA kT-point pulse design complete.")
            if rf_waveforms.numel() > 0: # Check if tensor is not empty
                 self._log(f"RF Waveforms shape: {rf_waveforms.shape}, Max abs: {torch.max(torch.abs(rf_waveforms)):.2e}")

        return rf_waveforms

    def _log(self, message):
        if self.verbose:
            print(f"[STAPTxDesigner] {message}")

if __name__ == '__main__':
    print("STAPTxDesigner class with core logic.")
    designer = STAPTxDesigner(verbose=True)

    num_ch_test = 2
    N_kt_test = 10
    Nx_s, Ny_s, Nz_s = 16, 16, 1
    N_spatial_dims_test = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dummy_b1 = (torch.rand(num_ch_test, Nx_s, Ny_s, Nz_s, dtype=torch.complex64, device=device) - 0.5 + \
               1j*(torch.rand(num_ch_test, Nx_s, Ny_s, Nz_s, dtype=torch.complex64, device=device) - 0.5)) * 2e-6

    target_mxy = torch.zeros(Nx_s, Ny_s, Nz_s, dtype=torch.complex64, device=device)
    center_x, center_y = Nx_s // 2, Ny_s // 2
    radius = Nx_s // 4
    for r_idx in range(Nx_s):
        for c_idx in range(Ny_s):
            if (r_idx - center_x)**2 + (c_idx - center_y)**2 < radius**2:
                target_mxy[r_idx, c_idx, 0] = 0.01 + 0j # Target a small Mxy, M0 is 1.0

    dummy_kts = (torch.rand(N_kt_test, N_spatial_dims_test, device=device) - 0.5) * 2 * np.pi / (0.05) # kmax ~ 2pi / (5cm_fov)

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.1, 0.1, Nx_s, device=device, dtype=torch.float32),
        torch.linspace(-0.1, 0.1, Ny_s, device=device, dtype=torch.float32),
        indexing='ij'
    )

    spatial_grid_to_use = torch.stack((grid_x.unsqueeze(2).expand(-1,-1,Nz_s),
                                       grid_y.unsqueeze(2).expand(-1,-1,Nz_s)), dim=-1)
    if N_spatial_dims_test == 3:
         dummy_grid_z_vals = torch.linspace(-0.005, 0.005, Nz_s, device=device, dtype=torch.float32)
         dummy_grid_z = dummy_grid_z_vals.view(1,1,Nz_s,1).expand(Nx_s,Ny_s,-1,-1)
         spatial_grid_to_use = torch.cat((spatial_grid_to_use,dummy_grid_z),dim=-1)

    mask = torch.ones(Nx_s, Ny_s, Nz_s, dtype=torch.bool, device=device)

    try:
        rf_designed = designer.design_kta_pulse(
            b1_maps=dummy_b1,
            target_mxy_pattern=target_mxy,
            kt_points=dummy_kts,
            spatial_grid_m=spatial_grid_to_use,
            dt_per_kt_point=4e-6, # 4us per kT-point
            regularization_factor=1e-5,
            target_spatial_mask=mask
        )
        print(f"STA Design successful. RF shape: {rf_designed.shape}")
        if rf_designed.numel() > 0:
             print(f"RF Max abs: {torch.max(torch.abs(rf_designed))}")
    except Exception as e:
        print(f"Error in STAPTxDesigner example execution: {e}")
        import traceback
        traceback.print_exc()
