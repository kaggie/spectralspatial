import numpy as np

def calculate_static_shims(
    b1_maps: np.ndarray,
    target_mask: np.ndarray,
    target_b1_amplitude: float = 1.0,
    regularization_factor: float = 1e-2,
    return_achieved_b1: bool = False
    ):
    """
    Calculates static B1+ shim weights for multiple transmit channels to homogenize
    the effective B1+ field within a target Region of Interest (ROI).

    The optimization aims to find complex shim weights 'w' that minimize:
    || A * w - b ||^2_2 + lambda * ||w||^2_2
    where A is the system matrix from b1_maps in the ROI, b is the target
    effective B1+ vector, and lambda is the regularization_factor.
    This is solved via the normal equations:
    (A_H * A + lambda * I) * w = A_H * b

    Args:
        b1_maps (np.ndarray): Complex B1+ sensitivity maps for each channel.
            Expected shape: (num_channels, Nx, Ny, Nz) or (num_channels, Nvoxels_total)
            if target_mask is also flattened.
            Units should be consistent (e.g., uT/Volt or arbitrary units).
        target_mask (np.ndarray): Boolean mask defining the ROI.
            Expected shape: (Nx, Ny, Nz) or (Nvoxels_total).
            True values indicate voxels within the ROI.
        target_b1_amplitude (float, optional): The desired uniform effective B1+
            amplitude within the ROI. The phase of the target is implicitly 0 (real).
            Defaults to 1.0 (arbitrary unit).
        regularization_factor (float, optional): Lambda value for L2 regularization
            on the shim weights. Helps to control shim power and improve robustness.
            Defaults to 1e-2. A value of 0 means no regularization.
        return_achieved_b1 (bool, optional): If True, also returns the achieved
                                           effective B1+ field within the ROI.
                                           Defaults to False.

    Returns:
        np.ndarray or tuple:
            - shim_weights (np.ndarray): Complex array of optimal shim weights
              for each channel. Shape: (num_channels,).
            - achieved_b1_roi (np.ndarray, optional): If `return_achieved_b1` is True,
              this is the complex effective B1+ field within the ROI after applying
              the shims. Shape: (Nvoxels_in_ROI,).
    """

    # --- Input Validation ---
    if not isinstance(b1_maps, np.ndarray) or not np.iscomplexobj(b1_maps):
        raise ValueError("b1_maps must be a complex numpy array.")
    if not isinstance(target_mask, np.ndarray) or target_mask.dtype != bool:
        raise ValueError("target_mask must be a boolean numpy array.")
    if b1_maps.ndim < 2:
        raise ValueError("b1_maps must have at least 2 dimensions (num_channels, Nvoxels_or_spatial_dims...).")
    if regularization_factor < 0:
        raise ValueError("regularization_factor cannot be negative.")

    num_channels = b1_maps.shape[0]

    # --- Prepare Data ---
    # Handle volumetric or flattened inputs
    if b1_maps.ndim > 2: # Volumetric, e.g., (num_channels, Nx, Ny, Nz)
        if target_mask.ndim == 1:
             raise ValueError(f"If b1_maps are volumetric {b1_maps.shape}, target_mask cannot be flat {target_mask.shape} unless b1_maps are (num_channels, Nvoxels_total).")
        if b1_maps.shape[1:] != target_mask.shape:
            raise ValueError(f"Spatial dimensions of volumetric b1_maps {b1_maps.shape[1:]} must match target_mask {target_mask.shape}.")

        b1_maps_flat_all_voxels = b1_maps.reshape(num_channels, -1)
        mask_flat = target_mask.flatten()
    elif b1_maps.ndim == 2: # Assumed pre-flattened: (num_channels, Nvoxels_total)
        if target_mask.ndim != 1 or b1_maps.shape[1] != target_mask.shape[0]:
             raise ValueError(f"If b1_maps are (num_channels, Nvoxels_total) {b1_maps.shape}, target_mask must be (Nvoxels_total,) {target_mask.shape} and Nvoxels must match.")
        b1_maps_flat_all_voxels = b1_maps
        mask_flat = target_mask
    else:
        raise ValueError("Unsupported shapes for b1_maps. Must be (num_channels, Nx,Ny,Nz) or (num_channels, Nvoxels_total).")

    roi_indices = np.where(mask_flat)[0]
    if len(roi_indices) == 0:
        # print("Warning: target_mask is empty. Returning zero shim weights.") # Consider if print is desired or just return
        shim_weights_solution = np.zeros(num_channels, dtype=np.complex128)
        if return_achieved_b1:
            return shim_weights_solution, np.array([], dtype=np.complex128)
        else:
            return shim_weights_solution

    A = b1_maps_flat_all_voxels[:, roi_indices].T
    b = np.full(len(roi_indices), target_b1_amplitude, dtype=np.float64)

    # --- Solve for Shim Weights ---
    A_H = A.conj().T
    A_H_A = A_H @ A

    # Ensure lambda_I is complex if A_H_A is complex, which it will be.
    lambda_I = regularization_factor * np.eye(num_channels, dtype=A_H_A.dtype)

    lhs = A_H_A + lambda_I
    rhs = A_H @ b

    try:
        shim_weights_solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if solve fails
        try:
            lhs_pinv = np.linalg.pinv(lhs) # pinv can handle singular matrices
            shim_weights_solution = lhs_pinv @ rhs
        except np.linalg.LinAlgError:
            # This case is highly unlikely if pinv is used, but as a last resort:
            shim_weights_solution = np.zeros(num_channels, dtype=np.complex128)

    if return_achieved_b1:
        achieved_b1_roi = A @ shim_weights_solution
        return shim_weights_solution, achieved_b1_roi
    else:
        return shim_weights_solution

if __name__ == '__main__':
    # Example Usage (for basic testing during development)
    print("Static pTx shimming module with core logic.")

    num_ch_test = 4
    Nx_test, Ny_test, Nz_test = 10, 10, 1
    # num_voxels_total_test = Nx_test * Ny_test * Nz_test # Not directly used in example

    b1_maps_test_vol = (np.random.rand(num_ch_test, Nx_test, Ny_test, Nz_test) - 0.5 +                        1j * (np.random.rand(num_ch_test, Nx_test, Ny_test, Nz_test) - 0.5)) * 2.0

    mask_test_vol = np.zeros((Nx_test, Ny_test, Nz_test), dtype=bool)
    mask_test_vol[3:7, 3:7, :] = True

    print(f"Test B1 maps shape: {b1_maps_test_vol.shape}")
    print(f"Test mask shape: {mask_test_vol.shape}, ROI size: {np.sum(mask_test_vol)}")

    try:
        print("\nTesting with volumetric inputs:")
        shims_vol, b1_eff_vol = calculate_static_shims(
            b1_maps_test_vol,
            mask_test_vol,
            target_b1_amplitude=1.0,
            regularization_factor=0.1,
            return_achieved_b1=True
        )
        print(f"Calculated shims (volumetric): {shims_vol}")
        if b1_eff_vol is not None and b1_eff_vol.size > 0: # Check if b1_eff_vol is not empty
            print(f"Achieved B1 in ROI (volumetric) mean: {np.mean(np.abs(b1_eff_vol)):.3f}, std: {np.std(np.abs(b1_eff_vol)):.3f}")

        b1_maps_test_flat = b1_maps_test_vol.reshape(num_ch_test, -1)
        mask_test_flat = mask_test_vol.flatten()

        print("\nTesting with flattened inputs:")
        shims_flat, b1_eff_flat = calculate_static_shims(
            b1_maps_test_flat,
            mask_test_flat,
            target_b1_amplitude=1.0,
            regularization_factor=0.1,
            return_achieved_b1=True
        )
        print(f"Calculated shims (flattened): {shims_flat}")
        if b1_eff_flat is not None and b1_eff_flat.size > 0: # Check if b1_eff_flat is not empty
            print(f"Achieved B1 in ROI (flattened) mean: {np.mean(np.abs(b1_eff_flat)):.3f}, std: {np.std(np.abs(b1_eff_flat)):.3f}")

        assert np.allclose(shims_vol, shims_flat), "Shims from volumetric and flat inputs should be close."
        if b1_eff_vol is not None and b1_eff_flat is not None and b1_eff_vol.size > 0 and b1_eff_flat.size > 0:
            assert np.allclose(b1_eff_vol, b1_eff_flat), "Achieved B1 from volumetric and flat inputs should be close."
        print("\nConsistency check passed.")

        print("\nTesting with empty mask:")
        empty_mask = np.zeros_like(mask_test_vol, dtype=bool)
        shims_empty_ret = calculate_static_shims(b1_maps_test_vol, empty_mask, return_achieved_b1=True)
        shims_empty, b1_empty = shims_empty_ret
        assert np.all(shims_empty == 0), "Shims for empty mask should be zero."
        assert b1_empty.size == 0, "Achieved B1 for empty mask should be empty."
        print("Empty mask test passed.")

    except ValueError as e:
        print(f"Error during example test: {e}")
    except Exception as e_gen:
        print(f"General error during example test: {e_gen}")
