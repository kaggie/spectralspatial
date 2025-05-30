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
