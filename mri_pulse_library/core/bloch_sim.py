import numpy as np
import torch

def bloch_simulate(rf, grad, dt, gamma=4257.0, b0=0.0, mx0=0.0, my0=0.0, mz0=1.0,
                   spatial_positions=None, freq_offsets=None, return_all=False,
                   T1=1.0, T2=0.1):
    """
    Simulate the Bloch equations for given RF and gradient waveforms.
    
    Args:
        rf (array-like): Complex RF waveform (Gauss), shape (N,)
        grad (array-like): Gradient waveform (G/cm), shape (N,)
        dt (float): Time step (s)
        gamma (float): Hz/Gyromagnetic ratio (Hz/G)
        b0 (float): Off-resonance (Hz), default 0
        mx0, my0, mz0 (float): Initial magnetization components
        spatial_positions (array-like): Positions (cm) for spatial simulation, shape (P,)
        freq_offsets (array-like): Frequency offsets (Hz) for spectral simulation, shape (F,)
        return_all (bool): If True, return all time points, else final M
        T1 (float): Longitudinal relaxation time (s), default 1.0 s.
        T2 (float): Transverse relaxation time (s), default 0.1 s.

    Returns:
        If both spatial_positions and freq_offsets are None:
            (Mx, My, Mz) final values or time series (if return_all)
        If spatial_positions or freq_offsets provided:
            Mx, My, Mz arrays over grid
    """
    rf = torch.as_tensor(rf, dtype=torch.complex64)
    grad = torch.as_tensor(grad, dtype=torch.float32)
    N = rf.shape[0]
    device = rf.device

    # Set up spatial and frequency grids
    if spatial_positions is not None:
        x = torch.as_tensor(spatial_positions, dtype=torch.float32, device=device)
    else:
        x = torch.tensor([0.0], dtype=torch.float32, device=device)
    if freq_offsets is not None:
        df = torch.as_tensor(freq_offsets, dtype=torch.float32, device=device)
    else:
        df = torch.tensor([0.0], dtype=torch.float32, device=device)

    Nx, Nf = len(x), len(df)
    M = torch.zeros(Nx, Nf, 3, dtype=torch.float32, device=device)
    M[..., 0] = mx0
    M[..., 1] = my0
    M[..., 2] = mz0

    if return_all:
        Mout = torch.zeros(N, Nx, Nf, 3, dtype=torch.float32, device=device)

    for t in range(N):
        # Calculate effective B1 and Bz at each position/freq
        b1 = rf[t]  # Gauss, complex
        gx = grad[t]  # G/cm
        # Bz = Gx * x + B0 + freq offset
        Bz = gx * x[:, None] + b0 + df[None, :]
        # Convert B1 to real/imag
        b1x = b1.real
        b1y = b1.imag

        # Effective field (rad/s)
        bx = gamma * 2 * np.pi * b1x
        by = gamma * 2 * np.pi * b1y
        bz = gamma * 2 * np.pi * Bz

        # Construct rotation vector (rad/s)
        omega = torch.stack([bx.expand(Nx, Nf), by.expand(Nx, Nf), bz], dim=-1)
        omega_norm = torch.norm(omega, dim=-1) + 1e-20
        dtheta = omega_norm * dt

        # Rodrigues' rotation formula
        k = omega / omega_norm.unsqueeze(-1)
        cost = torch.cos(dtheta)
        sint = torch.sin(dtheta)
        kx, ky, kz = k[...,0], k[...,1], k[...,2]

        # Current M
        mx, my, mz = M[...,0], M[...,1], M[...,2]
        dot = kx*mx + ky*my + kz*mz
        crossx = ky*mz - kz*my
        crossy = kz*mx - kx*mz
        crossz = kx*my - ky*mx

        M[...,0] = mx*cost + crossx*sint + kx*dot*(1-cost)
        M[...,1] = my*cost + crossy*sint + ky*dot*(1-cost)
        M[...,2] = mz*cost + crossz*sint + kz*dot*(1-cost)

        # Apply relaxation
        if T1 > 0 and T2 > 0:
            # M is (Nx, Nf, 3). apply_relaxation expects M (3,) and returns a new (3,) tensor.
            # Iterate over Nx and Nf dimensions.
            # M0 for apply_relaxation is implicitly handled by mz0 through the M matrix state.
            # Specifically, apply_relaxation has an M0 parameter, which defaults to 1.0.
            # This means it assumes relaxation is towards [0, 0, M0_equilibrium].
            # Our M vector's z-component (M[..., 2]) is the one that recovers towards M0_equilibrium.
            # The mz0 is the initial state of M[..., 2].
            M_after_rotation = M.clone() # Clone M after rotation before passing to relaxation
            for i in range(M.shape[0]): # Iterate over Nx
                for j in range(M.shape[1]): # Iterate over Nf
                    # Pass M0 from initial condition, effectively mz0 for the z-component recovery target
                    M[i, j, :] = apply_relaxation(M_after_rotation[i, j, :], dt, T1, T2, M0=mz0)

        if return_all:
            Mout[t] = M

    if return_all:
        return Mout.cpu().numpy()
    else:
        return M.cpu().numpy()

def small_tip_simulate(rf, grad, dt, gamma=4257.0, spatial_positions=None, freq_offsets=None):
    """
    Small-tip approximation: linear response. Returns transverse magnetization.

    Args:
        rf (array-like): Complex RF waveform (Gauss), shape (N,)
        grad (array-like): Gradient waveform (G/cm), shape (N,)
        dt (float): Time step (s)
        gamma (float): Hz/Gyromagnetic ratio (Hz/G)
        spatial_positions (array-like): Positions (cm)
        freq_offsets (array-like): Frequency offsets (Hz)

    Returns:
        Mxy (array): Shape (len(spatial_positions), len(freq_offsets))
    """
    rf = torch.as_tensor(rf, dtype=torch.complex64)
    grad = torch.as_tensor(grad, dtype=torch.float32)
    N = rf.shape[0]

    if spatial_positions is not None:
        x = torch.as_tensor(spatial_positions, dtype=torch.float32)
    else:
        x = torch.tensor([0.0], dtype=torch.float32)
    if freq_offsets is not None:
        df = torch.as_tensor(freq_offsets, dtype=torch.float32)
    else:
        df = torch.tensor([0.0], dtype=torch.float32)
    Nx, Nf = len(x), len(df)

    # Calculate phase accrual from gradients
    grad_cum = torch.cumsum(grad, dim=0) * dt  # G/cm * s = G*cm*s
    phase_space = torch.zeros((N, Nx, Nf), dtype=torch.float32)
    for t in range(N):
        phi = 2 * np.pi * gamma * (
            torch.outer(x, grad_cum[t].expand(Nf))  # spatial encoding
            + df * dt * t  # frequency encoding
        )
        phase_space[t] = phi

    # Integrate RF * phase
    Mxy = torch.zeros((Nx, Nf), dtype=torch.complex64)
    for t in range(N):
        Mxy += rf[t] * torch.exp(-1j * phase_space[t]) * dt
    return Mxy.cpu().numpy()

# --- New functions to be added ---

def rotate_magnetization(M, B_eff, dt, gyromagnetic_ratio_hz_t):
    """
    Rotates magnetization M due to effective field B_eff over time dt.
    Uses Rodrigues' rotation formula.

    Args:
        M (array-like or torch.Tensor): Magnetization vector [Mx, My, Mz].
        B_eff (array-like or torch.Tensor): Effective magnetic field vector [Bx, By, Bz] in Tesla.
        dt (float): Time step in seconds.
        gyromagnetic_ratio_hz_t (float): Gyromagnetic ratio in Hz/T.

    Returns:
        torch.Tensor: New magnetization vector after rotation.
    """
    M = torch.as_tensor(M, dtype=torch.float32)
    B_eff = torch.as_tensor(B_eff, dtype=torch.float32)

    if M.ndim == 0 or M.shape[0] != 3: # Check if M is not a scalar and has 3 elements
        raise ValueError("M must be a 3-element vector.")
    if B_eff.ndim == 0 or B_eff.shape[0] != 3: # Check if B_eff is not a scalar and has 3 elements
        raise ValueError("B_eff must be a 3-element vector.")


    gamma_rad_per_s_per_t = gyromagnetic_ratio_hz_t * 2 * torch.pi

    B_norm = torch.linalg.norm(B_eff)

    # If B_norm is very small (or zero), no rotation occurs
    # Using a small epsilon to prevent division by zero or issues with very small norms.
    if B_norm < 1e-20:
        return M

    axis = B_eff / B_norm
    angle = gamma_rad_per_s_per_t * B_norm * dt

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Rodrigues' rotation formula
    # M_new = M * cos_a + torch.cross(axis, M) * sin_a + axis * torch.dot(axis, M) * (1 - cos_a)
    # For M and axis as (3,) tensors, torch.sum(axis * M) is equivalent to dot product.
    dot_product = torch.sum(axis * M)
    M_new = M * cos_a + torch.cross(axis, M) * sin_a + axis * dot_product * (1 - cos_a)

    return M_new

def apply_relaxation(M, dt, T1, T2, M0=1.0):
    """
    Applies T1 and T2 relaxation to magnetization M over time dt.

    Args:
        M (array-like or torch.Tensor): Magnetization vector [Mx, My, Mz].
        dt (float): Time step in seconds.
        T1 (float): Longitudinal relaxation time in seconds.
        T2 (float): Transverse relaxation time in seconds.
        M0 (float): Equilibrium magnetization, defaults to 1.0.

    Returns:
        torch.Tensor: New magnetization vector after relaxation.
    """
    M = torch.as_tensor(M, dtype=torch.float32)

    if M.ndim == 0 or M.shape[0] != 3: # Check if M is not a scalar and has 3 elements
        raise ValueError("M must be a 3-element vector.")
    if T1 <= 0 or T2 <= 0: # T1 and T2 must be positive
        raise ValueError("T1 and T2 must be positive and non-zero.")
    if T2 > T1: # Physical constraint
        raise ValueError(f"T2 ({T2}s) cannot be greater than T1 ({T1}s).")
    if dt < 0:
        raise ValueError("dt cannot be negative.")

    # Ensure T1 and T2 are float tensors for ops with other tensors if they come from numpy/python float
    T1 = torch.as_tensor(T1, dtype=torch.float32, device=M.device)
    T2 = torch.as_tensor(T2, dtype=torch.float32, device=M.device)

    E1 = torch.exp(-dt / T1)
    E2 = torch.exp(-dt / T2)

    M_new_x = M[0] * E2
    M_new_y = M[1] * E2
    M_new_z = M[2] * E1 + M0 * (1 - E1) # M0 is equilibrium magnetization along z

    return torch.tensor([M_new_x, M_new_y, M_new_z], dtype=torch.float32, device=M.device)


def bloch_simulate_ptx(
    rf_waveforms_per_channel: torch.Tensor,
    b1_sensitivity_maps: torch.Tensor,
    dt_s: float,
    b0_map_hz: torch.Tensor,
    T1_map_s: torch.Tensor,
    T2_map_s: torch.Tensor,
    gyromagnetic_ratio_hz_t: float,
    initial_magnetization: torch.Tensor,
    spatial_grid_m: torch.Tensor,
    gradient_waveforms_tm: torch.Tensor = None,
    return_all_timepoints: bool = False
):
    """
    Simulates the Bloch equations for parallel transmit (pTx) RF pulses
    on a 3D spatial grid, including effects of dynamic gradients.

    Args:
        rf_waveforms_per_channel (torch.Tensor): Complex RF waveforms for each
            transmit channel. Shape: (num_channels, N_timepoints).
        b1_sensitivity_maps (torch.Tensor): Complex B1 sensitivity map for each
            channel at each spatial voxel. Shape: (num_channels, Nx, Ny, Nz). Units: Tesla (T).
        dt_s (float): Time step duration in seconds (uniform for all steps).
        b0_map_hz (torch.Tensor): B0 off-resonance map at each spatial voxel.
            Shape: (Nx, Ny, Nz). Units: Hertz (Hz).
        T1_map_s (torch.Tensor): T1 relaxation time map.
            Shape: (Nx, Ny, Nz). Units: seconds (s).
        T2_map_s (torch.Tensor): T2 relaxation time map.
            Shape: (Nx, Ny, Nz). Units: seconds (s).
        gyromagnetic_ratio_hz_t (float): Gyromagnetic ratio. Units: Hertz per Tesla (Hz/T).
        initial_magnetization (torch.Tensor): Initial magnetization state [Mx, My, Mz]
            for each voxel. Shape: (Nx, Ny, Nz, 3).
        spatial_grid_m (torch.Tensor): Defines the spatial coordinates (x,y,z) for each voxel.
                                       Shape: (Nx, Ny, Nz, 3). Units: meters (m).
        gradient_waveforms_tm (torch.Tensor, optional): Time-varying gradient waveforms
            [Gx(t), Gy(t), Gz(t)]. Shape: (N_timepoints, 3). Units: Tesla/meter (T/m).
            If None, gradients are assumed to be zero. Defaults to None.
        return_all_timepoints (bool, optional): If True, returns the magnetization
            state at all time points. Defaults to False.

    Returns:
        torch.Tensor: Magnetization state.
            If return_all_timepoints is True: Shape (N_timepoints, Nx, Ny, Nz, 3).
            Else: Shape (Nx, Ny, Nz, 3).
    """
    device = rf_waveforms_per_channel.device
    num_channels = rf_waveforms_per_channel.shape[0]
    N_timepoints = rf_waveforms_per_channel.shape[1]

    if not isinstance(spatial_grid_m, torch.Tensor) or spatial_grid_m.ndim != 4 or spatial_grid_m.shape[3] != 3:
        raise ValueError("spatial_grid_m must be a Tensor of shape (Nx, Ny, Nz, 3).")
    Nx, Ny, Nz = spatial_grid_m.shape[:3]

    if b1_sensitivity_maps.shape != (num_channels, Nx, Ny, Nz):
        raise ValueError(f"b1_sensitivity_maps shape mismatch. Expected ({num_channels}, {Nx}, {Ny}, {Nz}), got {b1_sensitivity_maps.shape}")

    expected_spatial_shape = (Nx, Ny, Nz)
    if b0_map_hz.shape != expected_spatial_shape:
        raise ValueError(f"b0_map_hz shape mismatch. Expected {expected_spatial_shape}, got {b0_map_hz.shape}")
    if T1_map_s.shape != expected_spatial_shape:
        raise ValueError(f"T1_map_s shape mismatch. Expected {expected_spatial_shape}, got {T1_map_s.shape}")
    if T2_map_s.shape != expected_spatial_shape:
        raise ValueError(f"T2_map_s shape mismatch. Expected {expected_spatial_shape}, got {T2_map_s.shape}")
    if initial_magnetization.shape != (Nx, Ny, Nz, 3):
        raise ValueError(f"initial_magnetization shape mismatch. Expected {(*expected_spatial_shape, 3)}, got {initial_magnetization.shape}")

    if gradient_waveforms_tm is not None:
        if not isinstance(gradient_waveforms_tm, torch.Tensor) or gradient_waveforms_tm.ndim != 2 or gradient_waveforms_tm.shape[0] != N_timepoints or gradient_waveforms_tm.shape[1] != 3:
            raise ValueError(f"gradient_waveforms_tm must be a Tensor of shape ({N_timepoints}, 3) or None.")
        gradient_waveforms_tm = gradient_waveforms_tm.to(device)

    M = initial_magnetization.clone().to(device)
    b1_sensitivity_maps = b1_sensitivity_maps.to(device)
    b0_map_hz = b0_map_hz.to(device)
    T1_map_s = T1_map_s.to(device)
    T2_map_s = T2_map_s.to(device)
    rf_waveforms_per_channel = rf_waveforms_per_channel.to(device)
    spatial_grid_m = spatial_grid_m.to(device)

    if return_all_timepoints:
        M_time_course = torch.zeros((N_timepoints, Nx, Ny, Nz, 3), dtype=M.dtype, device=device)

    b0_map_hz_flat = b0_map_hz.reshape(-1) # Keep in Hz for now

    M_flat = M.reshape(-1, 3)
    T1_map_s_flat = T1_map_s.reshape(-1)
    T2_map_s_flat = T2_map_s.reshape(-1)
    initial_M0_flat = initial_magnetization.reshape(-1, 3)[:, 2].clone().to(device)
    spatial_grid_m_flat = spatial_grid_m.reshape(-1, 3) # Shape (N_voxels, 3)

    b1_sens_flat = b1_sensitivity_maps.reshape(num_channels, -1) # Shape (num_channels, N_voxels)
    N_voxels = M_flat.shape[0]

    for t in range(N_timepoints):
        current_rf_amps = rf_waveforms_per_channel[:, t] # Shape (num_channels)
        # B1 eff = sum over channels ( rf_amp_channel[t] * B1_sens_channel_voxel )
        b1_eff_t_tesla_flat = torch.sum(current_rf_amps.unsqueeze(1) * b1_sens_flat, dim=0) # Shape (N_voxels)

        current_gradients_tm_t = None
        if gradient_waveforms_tm is not None:
            current_gradients_tm_t = gradient_waveforms_tm[t, :] # Shape (3,) [Gx, Gy, Gz] at time t

        for v_idx in range(N_voxels):
            b1_eff_voxel_t = b1_eff_t_tesla_flat[v_idx] # Complex scalar

            b_eff_x_tesla = b1_eff_voxel_t.real
            b_eff_y_tesla = b1_eff_voxel_t.imag

            # Calculate Bz component from B0 map and gradients
            # B0 map contribution (convert Hz to Tesla)
            base_b0_field_tesla = b0_map_hz_flat[v_idx] / gyromagnetic_ratio_hz_t
            total_b_eff_z_tesla = base_b0_field_tesla

            if current_gradients_tm_t is not None:
                vx, vy, vz = spatial_grid_m_flat[v_idx, 0], spatial_grid_m_flat[v_idx, 1], spatial_grid_m_flat[v_idx, 2]
                # Gradient induced field = Gx*x + Gy*y + Gz*z (all in Tesla)
                gradient_induced_field_tesla = (current_gradients_tm_t[0] * vx +
                                                current_gradients_tm_t[1] * vy +
                                                current_gradients_tm_t[2] * vz)
                total_b_eff_z_tesla = total_b_eff_z_tesla + gradient_induced_field_tesla # Add gradient field in Tesla

            B_eff_tesla_voxel = torch.stack([b_eff_x_tesla, b_eff_y_tesla, total_b_eff_z_tesla])

            M_voxel = M_flat[v_idx, :]
            # Assumes rotate_magnetization and apply_relaxation are defined in the same file
            M_rotated = rotate_magnetization(M_voxel, B_eff_tesla_voxel, dt_s, gyromagnetic_ratio_hz_t)
            M_relaxed = apply_relaxation(M_rotated, dt_s, T1_map_s_flat[v_idx], T2_map_s_flat[v_idx], M0=initial_M0_flat[v_idx])
            M_flat[v_idx, :] = M_relaxed

        if return_all_timepoints:
            M_time_course[t, ...] = M_flat.reshape(Nx, Ny, Nz, 3)

    if return_all_timepoints:
        return M_time_course
    else:
        return M_flat.reshape(Nx, Ny, Nz, 3)
