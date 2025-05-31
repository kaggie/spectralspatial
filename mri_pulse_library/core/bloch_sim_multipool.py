"""Core solver for multi-pool Bloch-McConnell equations."""
import torch
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON # Assuming this might be used or for consistency

def bloch_mcconnell_step(
    M_vector_flat_initial: torch.Tensor,
    dt_s: float,
    b1_complex_tesla: complex,
    b0_offset_hz: float,
    gyromagnetic_ratio_hz_t: float,
    pool_params: dict
) -> torch.Tensor:
    """
    Performs one time step of the Bloch-McConnell equations for a multi-pool system.

    The system is described by dM/dt = A * M + C, solved using the matrix exponential.
    M(dt) = expm(A*dt) * M(0) + (expm(A*dt) - I) * A_inv * C (analytical solution if A invertible)
    Alternatively, uses an augmented matrix approach: d/dt [M; 1] = [[A, C], [0, 0]] * [M; 1]
    which is more robust if A is singular.

    Args:
        M_vector_flat_initial (torch.Tensor): Initial magnetization vector, flattened.
            Shape: (num_pools * 3,), representing [Mx0,My0,Mz0, Mx1,My1,Mz1,...].
        dt_s (float): Time step duration in seconds.
        b1_complex_tesla (complex): Complex B1 RF field at the current time step (Tesla).
        b0_offset_hz (float): Global B0 off-resonance frequency in Hz. This is added
                              to the individual pool frequency offsets.
        gyromagnetic_ratio_hz_t (float): Gyromagnetic ratio in Hz/T.
        pool_params (dict): Dictionary containing multi-pool system parameters:
            'num_pools' (int): Number of pools.
            'M0_fractions' (torch.Tensor): Equilibrium magnetization fraction for each pool.
                                           Shape: (num_pools,). Sums to 1.0.
            'T1s' (torch.Tensor): T1 relaxation times for each pool (seconds). Shape: (num_pools,).
            'T2s' (torch.Tensor): T2 relaxation times for each pool (seconds). Shape: (num_pools,).
            'freq_offsets_hz' (torch.Tensor): Chemical shift for each pool relative
                                               to the reference frequency (Hz). Shape: (num_pools,).
            'exchange_rates_k_to_from' (torch.Tensor): Matrix of exchange rates (Hz).
                                                       k_to_from[i, j] is rate from pool j to pool i.
                                                       Shape: (num_pools, num_pools). Diagonal is 0.

    Returns:
        torch.Tensor: Final magnetization vector after the time step.
                      Shape: (num_pools * 3,).
    """
    device = M_vector_flat_initial.device
    dtype = M_vector_flat_initial.dtype # Typically float32 for simulations

    num_pools = pool_params['num_pools']
    M0_fractions = pool_params['M0_fractions'].to(device, dtype=dtype)
    T1s = pool_params['T1s'].to(device, dtype=dtype)
    T2s = pool_params['T2s'].to(device, dtype=dtype)
    pool_freq_offsets_hz = pool_params['freq_offsets_hz'].to(device, dtype=dtype)
    # k_to_from[i,j] is rate from j to i
    exchange_rates_k_to_from = pool_params['exchange_rates_k_to_from'].to(device, dtype=dtype)

    system_size = num_pools * 3
    A = torch.zeros((system_size, system_size), device=device, dtype=dtype)
    C_vector = torch.zeros(system_size, device=device, dtype=dtype)

    omega_1x_rad_s = gyromagnetic_ratio_hz_t * 2 * torch.pi * b1_complex_tesla.real
    omega_1y_rad_s = gyromagnetic_ratio_hz_t * 2 * torch.pi * b1_complex_tesla.imag

    for p in range(num_pools):
        # Indices for pool p's Mx, My, Mz components
        idx_mx, idx_my, idx_mz = 3 * p, 3 * p + 1, 3 * p + 2

        # Effective Larmor frequency for pool p (rad/s)
        omega_L_p_rad_s = 2 * torch.pi * (b0_offset_hz + pool_freq_offsets_hz[p])

        R1_p = 1.0 / T1s[p]
        R2_p = 1.0 / T2s[p]

        # --- Rotation terms (on-diagonal 3x3 blocks) ---
        # dMx/dt = ... + omega_L_p * My - omega_1y * Mz
        A[idx_mx, idx_my] += omega_L_p_rad_s
        A[idx_mx, idx_mz] += -omega_1y_rad_s
        # dMy/dt = ... - omega_L_p * Mx + omega_1x * Mz
        A[idx_my, idx_mx] += -omega_L_p_rad_s
        A[idx_my, idx_mz] += omega_1x_rad_s
        # dMz/dt = ... + omega_1y * Mx - omega_1x * My
        A[idx_mz, idx_mx] += omega_1y_rad_s
        A[idx_mz, idx_my] += -omega_1x_rad_s

        # --- Relaxation terms (on-diagonal 3x3 blocks) ---
        A[idx_mx, idx_mx] += -R2_p
        A[idx_my, idx_my] += -R2_p
        A[idx_mz, idx_mz] += -R1_p

        # Add to C vector for M0 recovery
        C_vector[idx_mz] += M0_fractions[p] * R1_p

        # --- Exchange terms ---
        # Sum of rates leaving pool p
        sum_k_out_of_p = torch.sum(exchange_rates_k_to_from[:, p]) # Sum over rows for column p (rates from p to other pools)

        A[idx_mx, idx_mx] += -sum_k_out_of_p
        A[idx_my, idx_my] += -sum_k_out_of_p
        A[idx_mz, idx_mz] += -sum_k_out_of_p

        # Rates coming into pool p from other pools q
        for q in range(num_pools):
            if p == q:
                continue
            k_p_from_q = exchange_rates_k_to_from[p, q] # Rate from q to p

            idx_qx, idx_qy, idx_qz = 3 * q, 3 * q + 1, 3 * q + 2
            A[idx_mx, idx_qx] += k_p_from_q
            A[idx_my, idx_qy] += k_p_from_q
            A[idx_mz, idx_qz] += k_p_from_q

    # Robust solution using augmented matrix for dM/dt = AM + C
    # M_final = expm(A_aug * dt) * [M_initial; 1]
    # where A_aug = [[A, C], [0, 0]]

    A_aug = torch.zeros((system_size + 1, system_size + 1), device=device, dtype=dtype)
    A_aug[:system_size, :system_size] = A
    A_aug[:system_size, system_size] = C_vector # C_vector is the last column of A_aug's top block
    # The last row of A_aug remains zeros.

    M_aug_initial = torch.zeros(system_size + 1, device=device, dtype=dtype)
    M_aug_initial[:system_size] = M_vector_flat_initial
    M_aug_initial[system_size] = 1.0 # Augment with 1

    expm_A_aug_dt = torch.matrix_exp(A_aug * dt_s)
    M_aug_final = torch.matmul(expm_A_aug_dt, M_aug_initial)

    M_final = M_aug_final[:system_size] # Extract the M part

    return M_final

if __name__ == '__main__':
    # Example Usage for bloch_mcconnell_step (Illustrative)
    print("Bloch-McConnell Step function defined.")

    # --- Example Parameters for a 2-Pool System (Water and CEST agent) ---
    dev = torch.device('cpu')
    num_pools = 2
    # Water: Pool 0, CEST Agent: Pool 1
    pool_params_example = {
        'num_pools': num_pools,
        'M0_fractions': torch.tensor([0.95, 0.05], device=dev),  # Water, CEST agent
        'T1s': torch.tensor([1.3, 1.0], device=dev),             # s
        'T2s': torch.tensor([0.05, 0.005], device=dev),           # s
        'freq_offsets_hz': torch.tensor([0.0, 2000.0], device=dev), # Hz (CEST agent at 2 kHz / ~3.3ppm at 14T, ~4.7ppm at 9.4T)
        'exchange_rates_k_to_from': torch.tensor([
            [0.0, 30.0],  # To water: from water (0), from CEST (k_wc = 30 Hz)
            [3.0, 0.0]    # To CEST: from water (k_cw = 3 Hz), from CEST (0)
        ], device=dev) # k_to_from[i,j] is rate from pool j to pool i
    }

    # Initial Magnetization (e.g., fully relaxed water, no CEST magnetization)
    M_initial_flat = torch.zeros(num_pools * 3, device=dev, dtype=torch.float32) # Explicitly float32 for example
    M_initial_flat[2] = pool_params_example['M0_fractions'][0] # Mz for water
    M_initial_flat[5] = pool_params_example['M0_fractions'][1] # Mz for CEST pool (can be 0 if starting from no solute Mz)


    dt = 0.001  # 1 ms time step
    b1_val_tesla = 1e-6  # 1 uT B1 field
    b1_complex = complex(b1_val_tesla, 0.0)
    b0_offset_val_hz = 0.0 # Global off-resonance
    gamma_val_hz_t = GAMMA_HZ_PER_T_PROTON # Using constant

    print(f"Initial M: {M_initial_flat}")

    # Simulate one step
    try:
        M_final_flat = bloch_mcconnell_step(
            M_initial_flat,
            dt,
            b1_complex,
            b0_offset_val_hz,
            gamma_val_hz_t,
            pool_params_example
        )
        print(f"Final M after {dt*1000} ms: {M_final_flat}")

        # Example: simulate a few steps to see evolution
        num_steps_sim = 10
        M_current = M_initial_flat.clone()
        print(f"\nSimulating {num_steps_sim} steps of {dt*1000} ms each:")
        for step_i in range(num_steps_sim):
            M_current = bloch_mcconnell_step(
                M_current, dt, b1_complex, b0_offset_val_hz, gamma_val_hz_t, pool_params_example
            )
            if (step_i + 1) % 2 == 0: # Print every 2 steps
                 print(f"Step {step_i+1}: Mz_water={M_current[2]:.4f}, Mz_cest={M_current[5]:.4f}")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
