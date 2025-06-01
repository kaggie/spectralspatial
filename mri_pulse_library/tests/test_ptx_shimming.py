import unittest
import numpy as np
from mri_pulse_library.ptx.shimming import calculate_static_shims

class TestStaticPTxShimming(unittest.TestCase):

    def setUp(self):
        # Common parameters for tests
        self.num_channels_default = 2
        self.nx, self.ny, self.nz = 4, 4, 2 # Small volume
        self.num_voxels_total = self.nx * self.ny * self.nz

        # Default B1 maps (num_channels, Nx, Ny, Nz) - complex
        self.b1_maps_vol = (np.random.rand(self.num_channels_default, self.nx, self.ny, self.nz) - 0.5 +                            1j * (np.random.rand(self.num_channels_default, self.nx, self.ny, self.nz) - 0.5)) * 2.0

        # Default mask (Nx, Ny, Nz) - boolean
        self.mask_vol = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        self.mask_vol[1:3, 1:3, 0:1] = True # A small ROI

        # Flattened versions for testing alternative inputs
        self.b1_maps_flat = self.b1_maps_vol.reshape(self.num_channels_default, -1)
        self.mask_flat = self.mask_vol.flatten()

    def test_basic_invocation_and_output_structure(self):
        shims = calculate_static_shims(self.b1_maps_vol, self.mask_vol)
        self.assertIsInstance(shims, np.ndarray)
        self.assertTrue(np.iscomplexobj(shims))
        self.assertEqual(shims.shape, (self.num_channels_default,))

        shims_flat_input = calculate_static_shims(self.b1_maps_flat, self.mask_flat)
        self.assertEqual(shims_flat_input.shape, (self.num_channels_default,))

    def test_return_achieved_b1(self):
        shims, achieved_b1 = calculate_static_shims(
            self.b1_maps_vol, self.mask_vol, return_achieved_b1=True
        )
        self.assertIsInstance(achieved_b1, np.ndarray)
        self.assertTrue(np.iscomplexobj(achieved_b1))
        self.assertEqual(achieved_b1.shape, (np.sum(self.mask_vol),))

        # Verify A @ w = achieved_b1
        roi_indices = np.where(self.mask_flat)[0]
        A = self.b1_maps_flat[:, roi_indices].T
        expected_achieved_b1 = A @ shims
        self.assertTrue(np.allclose(achieved_b1, expected_achieved_b1))

    def test_single_channel_shimming(self):
        b1_single_ch_vol = self.b1_maps_vol[0:1, ...] # Take first channel
        mask_roi_voxels_complex = b1_single_ch_vol[0][self.mask_vol] # B1 values in ROI for the single channel

        target_amp = 1.5

        shims = calculate_static_shims(b1_single_ch_vol, self.mask_vol,
                                       target_b1_amplitude=target_amp, regularization_factor=1e-8)
        self.assertEqual(shims.shape, (1,))

        if len(mask_roi_voxels_complex) > 0 and np.mean(np.abs(mask_roi_voxels_complex)) > 1e-6 :
            A_roi = mask_roi_voxels_complex.flatten()
            b_roi = np.full(len(A_roi), target_amp, dtype=float)

            expected_shim_val_numerator = np.sum(A_roi.conj() * b_roi)
            expected_shim_val_denominator = np.sum(np.abs(A_roi)**2) + 1e-8
            expected_shim_val = expected_shim_val_numerator / expected_shim_val_denominator

            self.assertAlmostEqual(shims[0], expected_shim_val, delta=1e-5 if np.abs(expected_shim_val) < 1 else np.abs(expected_shim_val)*0.01)

    def test_perfectly_homogenizable_case(self):
        num_ch = 2
        b1_ch1 = np.ones((self.nx, self.ny, self.nz), dtype=np.complex128)
        b1_ch2 = np.zeros((self.nx, self.ny, self.nz), dtype=np.complex128)
        b1_perfect = np.stack([b1_ch1, b1_ch2], axis=0)

        mask_simple = np.ones((self.nx, self.ny, self.nz), dtype=bool)
        target_val = 1.0

        shims, achieved = calculate_static_shims(b1_perfect, mask_simple,
                                                 target_b1_amplitude=target_val,
                                                 regularization_factor=1e-8,
                                                 return_achieved_b1=True)
        self.assertAlmostEqual(shims[0], 1.0, delta=1e-5)
        # shims[1] can be anything if b1_ch2 is exactly zero, as it won't affect cost.
        # So, we check its effect on achieved B1.
        self.assertTrue(np.allclose(np.abs(achieved), target_val, atol=1e-5))
        self.assertTrue(np.allclose(np.angle(achieved), 0, atol=1e-5))

        b1_ch2_ones = np.ones((self.nx, self.ny, self.nz), dtype=np.complex128)
        b1_both_ones = np.stack([b1_ch1, b1_ch2_ones], axis=0)
        target_val_2 = 2.0
        shims2, achieved2 = calculate_static_shims(b1_both_ones, mask_simple,
                                                   target_b1_amplitude=target_val_2,
                                                   regularization_factor=0.01,
                                                   return_achieved_b1=True)
        self.assertAlmostEqual(shims2[0], 1.0, delta=0.1)
        self.assertAlmostEqual(shims2[1], 1.0, delta=0.1)
        self.assertTrue(np.allclose(np.abs(achieved2), target_val_2, atol=0.1))

    def test_regularization_effect(self):
        target_amp = 1.0
        shims_no_reg = calculate_static_shims(self.b1_maps_vol, self.mask_vol,
                                            target_b1_amplitude=target_amp, regularization_factor=1e-10)
        norm_shims_no_reg = np.linalg.norm(shims_no_reg)

        shims_with_reg = calculate_static_shims(self.b1_maps_vol, self.mask_vol,
                                              target_b1_amplitude=target_amp, regularization_factor=1.0)
        norm_shims_with_reg = np.linalg.norm(shims_with_reg)

        if norm_shims_no_reg > 1e-9 :
            self.assertLessEqual(norm_shims_with_reg, norm_shims_no_reg + 1e-6)

    def test_input_validation_errors(self):
        with self.assertRaises(ValueError):
            calculate_static_shims(self.b1_maps_vol.real, self.mask_vol)
        with self.assertRaises(ValueError):
            calculate_static_shims(self.b1_maps_vol, self.mask_vol.astype(int))
        wrong_mask = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        with self.assertRaises(ValueError):
            calculate_static_shims(self.b1_maps_vol, wrong_mask)
        wrong_mask_flat = np.zeros(self.num_voxels_total + 1, dtype=bool)
        with self.assertRaises(ValueError):
            calculate_static_shims(self.b1_maps_flat, wrong_mask_flat)
        with self.assertRaises(ValueError):
            calculate_static_shims(self.b1_maps_vol, self.mask_vol, regularization_factor=-0.1)
        with self.assertRaises(ValueError): # b1_maps too few dims
            calculate_static_shims(self.b1_maps_vol[0,0,0,:], self.mask_vol)

    def test_empty_roi(self):
        empty_mask = np.zeros_like(self.mask_vol, dtype=bool)
        shims = calculate_static_shims(self.b1_maps_vol, empty_mask)
        self.assertTrue(np.all(shims == 0))
        self.assertEqual(shims.shape, (self.num_channels_default,))

        shims_ret, achieved_ret = calculate_static_shims(self.b1_maps_vol, empty_mask, return_achieved_b1=True)
        self.assertTrue(np.all(shims_ret == 0))
        self.assertEqual(achieved_ret.shape, (0,))

    def test_volumetric_vs_flattened_consistency(self):
        shims_vol = calculate_static_shims(self.b1_maps_vol, self.mask_vol, regularization_factor=0.05)
        shims_flat = calculate_static_shims(self.b1_maps_flat, self.mask_flat, regularization_factor=0.05)
        self.assertTrue(np.allclose(shims_vol, shims_flat),
                        "Shims from volumetric and pre-flattened inputs should be consistent.")

        shims_vol_ret, ach_vol = calculate_static_shims(self.b1_maps_vol, self.mask_vol,
                                                        regularization_factor=0.05, return_achieved_b1=True)
        shims_flat_ret, ach_flat = calculate_static_shims(self.b1_maps_flat, self.mask_flat,
                                                          regularization_factor=0.05, return_achieved_b1=True)
        self.assertTrue(np.allclose(shims_vol_ret, shims_flat_ret))
        if ach_vol.size > 0 and ach_flat.size > 0 : # Only compare if not empty due to empty mask in some test setup
            self.assertTrue(np.allclose(ach_vol, ach_flat))

if __name__ == '__main__':
    unittest.main()
