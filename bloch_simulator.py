import numpy as np
import torch

def bloch_simulate(rf, grad, dt, gamma=4257.0, b0=0.0, mx0=0.0, my0=0.0, mz0=1.0, 
                   spatial_positions=None, freq_offsets=None, return_all=False):
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
