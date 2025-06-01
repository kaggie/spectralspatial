import unittest
import torch
import numpy as np # For np.pi, np.isclose, etc.
from mri_pulse_library.ptx.sta_designer import STAPTxDesigner
from mri_pulse_library.core.constants import GAMMA_RAD_PER_S_PER_T_PROTON, M0_PROTON
# For creating a base pulse in spectral content test, if not using hard pulse from setUp
from mri_pulse_library.rf_pulses.simple.sinc_pulse import generate_sinc_pulse
from mri_pulse_library.rf_pulses.simple.hard_pulse import generate_hard_pulse


# Helper to get spectrum (simplified)
def get_spectrum(signal, dt):
    if len(signal) == 0:
        return np.array([]), np.array([])
    n = len(signal)
    # Frequencies are from -Fs/2 to Fs/2 if using fftshift
    # Fs = 1/dt
    freq = np.fft.fftfreq(n, d=dt)
    spectrum = np.fft.fft(signal)
    return np.fft.fftshift(freq), np.fft.fftshift(np.abs(spectrum))

class TestSTAPTxDesigner(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.designer = STAPTxDesigner(verbose=False) # Turn off prints for tests

        # Common parameters
        self.num_channels = 2
        self.nx, self.ny, self.nz = 4, 4, 1 # Small spatial grid
        self.num_vox_total = self.nx * self.ny * self.nz
        self.num_kt_points = 5
        self.num_spatial_dims = 2 # Using 2D k-space (kx, ky) and grid (x,y) for simplicity

        # B1 maps (num_channels, Nx, Ny, Nz) - complex
        # Ensure B1 maps are on the correct device
        self.b1_maps = (torch.rand(self.num_channels, self.nx, self.ny, self.nz, dtype=torch.complex64, device=self.device) - 0.5 +                            1j * (torch.rand(self.num_channels, self.nx, self.ny, self.nz, dtype=torch.complex64, device=self.device) - 0.5)) * 2e-6 # Example scale Tesla/Volt
        self.b1_maps += (1e-7 + 1j*1e-7) # Avoid all zeros by adding a small complex offset

        # Target Mxy pattern (Nx, Ny, Nz) - complex
        self.target_mxy = torch.zeros(self.nx, self.ny, self.nz, dtype=torch.complex64, device=self.device)
        self.target_mxy[1:3, 1:3, 0] = 0.05 + 0.05j # Excite a small central region

        # kT-points (Num_kT_points, N_spatial_dims) - rad/m
        self.kt_points = (torch.rand(self.num_kt_points, self.num_spatial_dims, device=self.device, dtype=torch.float32) - 0.5) * 2 * np.pi / 0.05

        # Spatial grid (Nx, Ny, Nz, N_spatial_dims) - meters
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-0.1, 0.1, self.nx, device=self.device, dtype=torch.float32),
            torch.linspace(-0.1, 0.1, self.ny, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self.spatial_grid = torch.stack(
            (grid_x.unsqueeze(2).expand(-1,-1,self.nz),
             grid_y.unsqueeze(2).expand(-1,-1,self.nz)),
            dim=-1
        )

        if self.num_spatial_dims == 3:
            grid_z_vals = torch.linspace(-0.01, 0.01, self.nz, device=self.device, dtype=torch.float32).view(1,1,self.nz,1).expand(self.nx,self.ny,-1,-1)
            self.spatial_grid = torch.cat((self.spatial_grid, grid_z_vals), dim=-1)

        self.dt_kt = 2e-5
        self.reg = 1e-4

        # Base RF for multiband tests (used in some tests, not STAPTxDesigner directly)
        base_rf_hard_np, _, _ = generate_hard_pulse(
            flip_angle_deg=10, duration_s=0.001, dt_s=1e-5
        )
        self.base_rf_hard_torch = torch.tensor(base_rf_hard_np.astype(np.complex64), device=self.device)


    def test_basic_invocation_and_output_shape(self):
        rf_designed = self.designer.design_kta_pulse(
            b1_maps=self.b1_maps,
            target_mxy_pattern=self.target_mxy,
            kt_points=self.kt_points,
            spatial_grid_m=self.spatial_grid,
            dt_per_kt_point=self.dt_kt,
            regularization_factor=self.reg
        )
        self.assertIsInstance(rf_designed, torch.Tensor)
        self.assertTrue(rf_designed.is_complex())
        self.assertEqual(rf_designed.shape, (self.num_channels, self.num_kt_points))
        self.assertEqual(rf_designed.device, self.device)

    def test_single_channel_voxel_ktpoint_k0(self):
        num_ch = 1
        b1_val_scalar = 0.5 + 0.5j
        b1_map_single = torch.full((num_ch, 1, 1, 1), b1_val_scalar, dtype=torch.complex64, device=self.device) * 1e-6

        target_mxy_scalar = 0.1 + 0.0j
        target_mxy_single = torch.full((1,1,1), target_mxy_scalar, dtype=torch.complex64, device=self.device)

        kt_point_k0 = torch.tensor([[0.0, 0.0]], device=self.device, dtype=torch.float32)
        spatial_coord = torch.tensor([[[[0.0, 0.0]]]], device=self.device, dtype=torch.float32)
        mask = torch.ones(1,1,1, dtype=torch.bool, device=self.device)
        dt = 1e-4
        reg_small = 1e-10

        rf_designed = self.designer.design_kta_pulse(
            b1_maps=b1_map_single,
            target_mxy_pattern=target_mxy_single,
            kt_points=kt_point_k0,
            spatial_grid_m=spatial_coord,
            dt_per_kt_point=dt,
            regularization_factor=reg_small,
            target_spatial_mask=mask
        )

        C_val = -1j * GAMMA_RAD_PER_S_PER_T_PROTON * M0_PROTON * dt
        exp_term = 1.0 # exp(j * k.r) where k=0

        S_element = C_val * (b1_val_scalar * 1e-6) * exp_term # b1_val_scalar needs to be in T/V or T/arb_unit
        S_H_S_element = np.abs(S_element)**2
        lhs_val = S_H_S_element + reg_small
        rhs_val = np.conjugate(S_element) * target_mxy_scalar
        expected_rf_val = rhs_val / lhs_val

        self.assertEqual(rf_designed.shape, (num_ch, 1))
        self.assertTrue(torch.isclose(rf_designed[0,0], torch.tensor(expected_rf_val, dtype=torch.complex64, device=self.device), atol=1e-6))

    def test_regularization_effect(self):
        rf_low_reg = self.designer.design_kta_pulse(
            self.b1_maps, self.target_mxy, self.kt_points, self.spatial_grid, self.dt_kt, 1e-6
        )
        rf_high_reg = self.designer.design_kta_pulse(
            self.b1_maps, self.target_mxy, self.kt_points, self.spatial_grid, self.dt_kt, 1e-0
        )
        norm_low_reg = torch.linalg.norm(rf_low_reg)
        norm_high_reg = torch.linalg.norm(rf_high_reg)

        if norm_low_reg > 1e-9 :
            self.assertLessEqual(norm_high_reg.item(), norm_low_reg.item() + 1e-6,
                                 "Norm of RF with higher regularization should generally be smaller or equal.")

    def test_target_spatial_mask_effect(self):
        rf_full_mask = self.designer.design_kta_pulse(
            self.b1_maps, self.target_mxy, self.kt_points, self.spatial_grid, self.dt_kt, self.reg,
            target_spatial_mask=None
        )

        partial_mask = torch.zeros_like(self.target_mxy.squeeze(), dtype=torch.bool) # ensure mask is 3D if target_mxy is
        partial_mask[self.nx//2, self.ny//2, 0] = True

        target_mxy_for_partial_mask = self.target_mxy.clone()
        if torch.abs(target_mxy_for_partial_mask[self.nx//2, self.ny//2, 0]) < 1e-9:
             target_mxy_for_partial_mask[self.nx//2, self.ny//2, 0] = 0.01 + 0.01j

        rf_partial_mask = self.designer.design_kta_pulse(
            self.b1_maps, target_mxy_for_partial_mask, self.kt_points, self.spatial_grid, self.dt_kt, self.reg,
            target_spatial_mask=partial_mask.reshape(self.nx, self.ny, self.nz) # ensure correct shape for mask
        )
        # It's hard to predict an exact relation, but they should not be identical if mask is effective
        # Sum of squares of RF might be smaller for partial mask if target is mostly outside
        self.assertFalse(torch.allclose(rf_full_mask, rf_partial_mask, atol=1e-5),
                         "RF solution should differ when target spatial mask changes significantly if target is different.")

    def test_input_validation_errors(self):
        wrong_target_mxy = torch.rand(self.nx+1, self.ny, self.nz, dtype=torch.complex64, device=self.device)
        with self.assertRaisesRegex(ValueError, "Spatial shape of target_mxy_pattern must match b1_maps"):
            self.designer.design_kta_pulse(self.b1_maps, wrong_target_mxy, self.kt_points, self.spatial_grid, self.dt_kt)

        wrong_kt_points = torch.rand(self.num_kt_points, self.num_spatial_dims + 1, device=self.device)
        with self.assertRaisesRegex(ValueError, "Last dimension of spatial_grid_m .* must match .* kt_points"):
             self.designer.design_kta_pulse(self.b1_maps, self.target_mxy, wrong_kt_points, self.spatial_grid, self.dt_kt)

        with self.assertRaisesRegex(ValueError, "dt_per_kt_point must be positive"):
            self.designer.design_kta_pulse(self.b1_maps, self.target_mxy, self.kt_points, self.spatial_grid, -0.01)

        with self.assertRaisesRegex(ValueError, "b1_maps must be a complex PyTorch Tensor"):
            self.designer.design_kta_pulse(self.b1_maps.real(), self.target_mxy, self.kt_points, self.spatial_grid, self.dt_kt)

    def test_empty_roi_mask(self):
        empty_mask = torch.zeros_like(self.target_mxy, dtype=torch.bool, device=self.device)
        rf_empty_roi = self.designer.design_kta_pulse(
            self.b1_maps, self.target_mxy, self.kt_points, self.spatial_grid, self.dt_kt,
            target_spatial_mask=empty_mask
        )
        self.assertTrue(torch.allclose(rf_empty_roi, torch.zeros_like(rf_empty_roi, device=self.device), atol=1e-9),
                        "RF should be zero for an empty ROI mask.")
        self.assertEqual(rf_empty_roi.shape, (self.num_channels, self.num_kt_points))

if __name__ == '__main__':
    unittest.main()
