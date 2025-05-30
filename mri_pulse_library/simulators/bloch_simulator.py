# File: mri_pulse_library/simulators/bloch_simulator.py
import numpy as np
import torch # For torch.as_tensor if needed, core simulator handles tensor conversion
from mri_pulse_library.core.bloch_sim import bloch_simulate as core_bloch_simulate
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON, GAMMA_HZ_PER_G_PROTON

# Conversion factor: 1 Tesla = 10,000 Gauss
T_TO_G = 10000.0
# Conversion factor: 1 T/m = 100 G/cm
TM_TO_GCM = 100.0


def simulate_hard_pulse_profile(rf_pulse_T, time_s,
                                B0_T=3.0, T1_s=1.0, T2_s=0.1,
                                mx0=0.0, my0=0.0, mz0=1.0,
                                freq_offsets_hz=np.array([0.0]),
                                gyromagnetic_ratio_hz_g=GAMMA_HZ_PER_G_PROTON):
    """
    Simulates the magnetization profile for a hard pulse over a range of frequency offsets.

    Args:
        rf_pulse_T (np.ndarray): RF pulse amplitude waveform (Tesla).
        time_s (np.ndarray): Time vector for the RF pulse (seconds).
        B0_T (float, optional): Main magnetic field strength (Tesla). Not directly used in b0 offset
                                for core simulator but kept for context. Defaults to 3.0.
        T1_s (float, optional): Longitudinal relaxation time (seconds). Defaults to 1.0.
        T2_s (float, optional): Transverse relaxation time (seconds). Defaults to 0.1.
        mx0 (float, optional): Initial Mx magnetization. Defaults to 0.0.
        my0 (float, optional): Initial My magnetization. Defaults to 0.0.
        mz0 (float, optional): Initial Mz magnetization. Defaults to 1.0.
        freq_offsets_hz (np.ndarray, optional): Array of frequency offsets to simulate (Hz).
                                               Defaults to np.array([0.0]).
        gyromagnetic_ratio_hz_g (float, optional): Gyromagnetic ratio in Hz/Gauss.
                                                  Defaults to GAMMA_HZ_PER_G_PROTON.

    Returns:
        np.ndarray: Magnetization profile (Nf, 3) where Nf is number of frequency offsets.
    """

    if not isinstance(rf_pulse_T, np.ndarray): rf_pulse_T = np.asarray(rf_pulse_T)
    if not isinstance(time_s, np.ndarray): time_s = np.asarray(time_s)
    if not isinstance(freq_offsets_hz, np.ndarray): freq_offsets_hz = np.asarray(freq_offsets_hz)


    if rf_pulse_T.size == 0 or time_s.size == 0:
        # print("Warning: Empty RF pulse or time vector provided.")
        return np.empty((len(freq_offsets_hz), 3)) if len(freq_offsets_hz) > 0 else np.empty((0,3))
    if rf_pulse_T.shape != time_s.shape:
        raise ValueError("rf_pulse_T and time_s must have the same shape.")

    dt_s = time_s[1] - time_s[0] if len(time_s) > 1 else (time_s[0] if len(time_s) == 1 else 1e-6)
    if dt_s <= 0 and len(time_s)>1 : raise ValueError("Time steps (dt_s) must be positive.")


    rf_pulse_G = rf_pulse_T * T_TO_G # Convert RF from Tesla to Gauss

    # For a hard pulse, gradient is typically zero (non-selective)
    grad_G_cm = np.zeros_like(rf_pulse_G) # G/cm

    M_all_freqs = core_bloch_simulate(
        rf=rf_pulse_G, grad=grad_G_cm, dt=dt_s,
        gamma=gyromagnetic_ratio_hz_g,
        b0=0.0, # Base off-resonance; specific offsets handled by freq_offsets arg
        mx0=mx0, my0=my0, mz0=mz0,
        T1=T1_s, T2=T2_s,
        spatial_positions=None,
        freq_offsets=freq_offsets_hz
    )
    # Output shape from core_bloch_simulate is (Nx, Nf, 3). Here Nx=1. So (1, Nf, 3).
    return np.squeeze(M_all_freqs, axis=0) # Shape (Nf, 3)


def simulate_slice_selective_profile(rf_pulse_T, time_s,
                                     slice_select_gradient_Tm,
                                     z_positions_m,
                                     B0_T=3.0, T1_s=1.0, T2_s=0.1,
                                     mx0=0.0, my0=0.0, mz0=1.0,
                                     freq_offsets_hz=np.array([0.0]),
                                     gyromagnetic_ratio_hz_g=GAMMA_HZ_PER_G_PROTON):
    """
    Simulates the slice profile for a slice-selective pulse.

    Args:
        rf_pulse_T (np.ndarray): RF pulse amplitude waveform (Tesla).
        time_s (np.ndarray): Time vector for the RF pulse (seconds).
        slice_select_gradient_Tm (float): Slice selection gradient strength (T/m).
        z_positions_m (np.ndarray): Spatial positions along slice direction (m) to simulate profile.
        B0_T (float, optional): Main magnetic field (Tesla). Defaults to 3.0.
        T1_s (float, optional): Longitudinal relaxation time (s). Defaults to 1.0.
        T2_s (float, optional): Transverse relaxation time (s). Defaults to 0.1.
        mx0 (float, optional): Initial Mx magnetization. Defaults to 0.0.
        my0 (float, optional): Initial My magnetization. Defaults to 0.0.
        mz0 (float, optional): Initial Mz magnetization. Defaults to 1.0.
        freq_offsets_hz (np.ndarray, optional): Spectral off-resonance values (Hz). Defaults to np.array([0.0]).
        gyromagnetic_ratio_hz_g (float, optional): Gyromagnetic ratio in Hz/Gauss. Defaults to GAMMA_HZ_PER_G_PROTON.

    Returns:
        np.ndarray: Magnetization profile (Nz, Nf, 3) where Nz is number of z positions,
                    Nf is number of frequency offsets.
    """
    if not isinstance(rf_pulse_T, np.ndarray): rf_pulse_T = np.asarray(rf_pulse_T)
    if not isinstance(time_s, np.ndarray): time_s = np.asarray(time_s)
    if not isinstance(z_positions_m, np.ndarray): z_positions_m = np.asarray(z_positions_m)
    if not isinstance(freq_offsets_hz, np.ndarray): freq_offsets_hz = np.asarray(freq_offsets_hz)

    if rf_pulse_T.size == 0 or time_s.size == 0:
        return np.empty((len(z_positions_m), len(freq_offsets_hz), 3)) if (len(z_positions_m)>0 and len(freq_offsets_hz)>0) else np.empty((0,0,3))
    if rf_pulse_T.shape != time_s.shape:
        raise ValueError("rf_pulse_T and time_s must have the same shape.")

    dt_s = time_s[1] - time_s[0] if len(time_s) > 1 else (time_s[0] if len(time_s) == 1 else 1e-6)
    if dt_s <= 0 and len(time_s)>1 : raise ValueError("Time steps (dt_s) must be positive.")

    rf_pulse_G = rf_pulse_T * T_TO_G

    grad_G_cm_scalar = slice_select_gradient_Tm * TM_TO_GCM
    grad_waveform_G_cm = np.ones_like(rf_pulse_G) * grad_G_cm_scalar

    z_positions_cm = z_positions_m * 100.0 # Convert m to cm

    M_profile = core_bloch_simulate(
        rf=rf_pulse_G, grad=grad_waveform_G_cm, dt=dt_s,
        gamma=gyromagnetic_ratio_hz_g,
        b0=0.0,
        mx0=mx0, my0=my0, mz0=mz0,
        T1=T1_s, T2=T2_s,
        spatial_positions=z_positions_cm,
        freq_offsets=freq_offsets_hz
    )
    return M_profile # Shape (Nz, Nf, 3)

# Specific wrappers based on pseudocode:

def simulate_hard_pulse(rf_pulse_T, time_s, B0_T=3.0, T1_s=1.0, T2_s=0.1, mx0=0.0, my0=0.0, mz0=1.0):
    M_final_matrix = simulate_hard_pulse_profile(
        rf_pulse_T, time_s, B0_T, T1_s, T2_s, mx0, my0, mz0,
        freq_offsets_hz=np.array([0.0])
    )
    # M_final_matrix is (1,3) for single freq_offset. Squeeze to (3,).
    return M_final_matrix[0,:] if M_final_matrix.shape[0] == 1 else M_final_matrix


def simulate_sinc_pulse(rf_pulse_T, time_s,
                        slice_select_gradient_Tm, z_positions_m,
                        B0_T=3.0, T1_s=1.0, T2_s=0.1, mx0=0.0, my0=0.0, mz0=1.0):
    M_profile = simulate_slice_selective_profile(
        rf_pulse_T, time_s, slice_select_gradient_Tm, z_positions_m,
        B0_T, T1_s, T2_s, mx0, my0, mz0,
        freq_offsets_hz=np.array([0.0])
    )
    # M_profile is (Nz, 1, 3). Squeeze the spectral dimension.
    return np.squeeze(M_profile, axis=1)


def simulate_gaussian_pulse(rf_pulse_T, time_s,
                            slice_select_gradient_Tm, z_positions_m,
                            B0_T=3.0, T1_s=1.0, T2_s=0.1, mx0=0.0, my0=0.0, mz0=1.0):
    M_profile = simulate_slice_selective_profile(
        rf_pulse_T, time_s, slice_select_gradient_Tm, z_positions_m,
        B0_T, T1_s, T2_s, mx0, my0, mz0,
        freq_offsets_hz=np.array([0.0])
    )
    return np.squeeze(M_profile, axis=1)


def simulate_spsp_pulse(rf_pulse_T, time_s,
                        slice_select_gradient_Tm, z_positions_m, freq_offsets_hz,
                        B0_T=3.0, T1_s=1.0, T2_s=0.1, mx0=0.0, my0=0.0, mz0=1.0):
    M_profile = simulate_slice_selective_profile(
        rf_pulse_T, time_s, slice_select_gradient_Tm, z_positions_m,
        B0_T, T1_s, T2_s, mx0, my0, mz0,
        freq_offsets_hz=freq_offsets_hz
    )
    return M_profile


def simulate_hs1_pulse(rf_pulse_T, time_s,
                       B0_T=3.0, T1_s=1.0, T2_s=0.1,
                       B1_variations=np.array([1.0]),
                       mx0=0.0, my0=0.0, mz0=1.0):
    if not isinstance(rf_pulse_T, np.ndarray): rf_pulse_T = np.asarray(rf_pulse_T)
    if not isinstance(B1_variations, np.ndarray): B1_variations = np.asarray(B1_variations)

    results_b1_var = []
    for b1_scale in B1_variations:
        rf_scaled_T = rf_pulse_T * b1_scale # Element-wise multiplication if rf_pulse_T is already an array

        M_final_single_b1_matrix = simulate_hard_pulse_profile(
            rf_scaled_T, time_s, B0_T, T1_s, T2_s, mx0, my0, mz0,
            freq_offsets_hz=np.array([0.0])
        )
        # M_final_single_b1_matrix is (1,3), squeeze to (3,)
        results_b1_var.append(M_final_single_b1_matrix[0,:] if M_final_single_b1_matrix.shape[0]==1 else M_final_single_b1_matrix)

    return np.array(results_b1_var)


def simulate_3d_multiband_spsp(rf_pulse_T, time_s,
                               slice_select_gradient_Tm, z_positions_m, freq_offsets_hz,
                               B0_T=3.0, T1_s=1.0, T2_s=0.1, mx0=0.0, my0=0.0, mz0=1.0):
    M_profile = simulate_slice_selective_profile(
        rf_pulse_T, time_s, slice_select_gradient_Tm, z_positions_m,
        B0_T, T1_s, T2_s, mx0, my0, mz0,
        freq_offsets_hz=freq_offsets_hz
    )
    return M_profile
