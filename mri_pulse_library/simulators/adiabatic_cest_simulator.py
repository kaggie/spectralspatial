"""Simulator for adiabatic pulses in multi-pool CEST (Chemical Exchange Saturation Transfer) systems."""
import torch
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON
from mri_pulse_library.core.bloch_sim_multipool import bloch_mcconnell_step
# from mri_pulse_library.rf_pulses.adiabatic.waveforms import generate_hs_pulse_waveforms # Example import

class AdiabaticCESTSimulator:
    """
    Simulates the response of multi-pool systems (e.g., for CEST or T1rho imaging)
    to adiabatic RF pulses over a range of B1 scaling factors and B0 offsets.

    This simulator utilizes a Bloch-McConnell solver to model the time evolution
    of magnetization in systems with chemical exchange.
    """
    def __init__(self,
                 gyromagnetic_ratio_hz_t: float = GAMMA_HZ_PER_T_PROTON,
                 device: str = 'cpu'):
        """
        Initializes the AdiabaticCESTSimulator.

        Args:
            gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
                                                      Defaults to GAMMA_HZ_PER_T_PROTON.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.gyromagnetic_ratio_hz_t = gyromagnetic_ratio_hz_t
        self.device = torch.device(device)
        self.dtype = torch.float32 # Default dtype for simulations

    def simulate_pulse_response(
        self,
        rf_amp_waveform_tesla: torch.Tensor,    # Shape (N_timepoints,)
        rf_freq_waveform_hz: torch.Tensor,      # Shape (N_timepoints,) - RF frequency relative to scanner 0 Hz
        pulse_duration_s: float,
        dt_s: float,                            # Time step for simulation
        target_b1_scales: torch.Tensor,         # Shape (Num_B1_Points,) - B1 scaling factors
        target_b0_offsets_hz: torch.Tensor,     # Shape (Num_B0_Points,) - Global B0 offsets
        tissue_params: dict,                    # Multi-pool parameters as defined in planning
        initial_M_vector_flat: torch.Tensor = None # Optional, shape (num_pools * 3)
    ) -> torch.Tensor:
        """
        Simulates the response of a multi-pool system to an adiabatic pulse
        across a range of B1 scaling factors and B0 offsets.

        Args:
            rf_amp_waveform_tesla (torch.Tensor): RF pulse amplitude modulation (Tesla).
                                                 Shape: (N_timepoints,).
            rf_freq_waveform_hz (torch.Tensor): RF pulse frequency modulation (Hz),
                                                relative to the scanner's base frequency.
                                                Shape: (N_timepoints,).
            pulse_duration_s (float): Total duration of the pulse in seconds.
            dt_s (float): Simulation time step in seconds. This value represents the
                          time interval between consecutive points in the input RF waveforms
                          (`rf_amp_waveform_tesla` and `rf_freq_waveform_hz`).
            target_b1_scales (torch.Tensor): 1D Tensor of B1 scaling factors to test.
            target_b0_offsets_hz (torch.Tensor): 1D Tensor of global B0 offsets (Hz) to test.
            tissue_params (dict): Dictionary of multi-pool tissue parameters. Expected keys:
                'num_pools', 'M0_fractions', 'T1s', 'T2s', 'freq_offsets_hz',
                'exchange_rates_k_to_from'. All values should be torch.Tensors.
            initial_M_vector_flat (torch.Tensor, optional): Initial magnetization state
                for all pools, flattened. Shape: (num_pools * 3,). If None, defaults
                to equilibrium based on M0_fractions.

        Returns:
            torch.Tensor: Resulting signal (e.g., Mz of the first pool - typically water)
                          after the pulse for each B1 scale and B0 offset.
                          Shape: (len(target_b1_scales), len(target_b0_offsets_hz)).
        """
        rf_amp_waveform_tesla = rf_amp_waveform_tesla.to(self.device, dtype=self.dtype)
        rf_freq_waveform_hz = rf_freq_waveform_hz.to(self.device, dtype=self.dtype)
        target_b1_scales = target_b1_scales.to(self.device, dtype=self.dtype)
        target_b0_offsets_hz = target_b0_offsets_hz.to(self.device, dtype=self.dtype)

        num_pools = tissue_params['num_pools']
        # Ensure all tissue_params tensors are on the correct device and dtype
        for key in ['M0_fractions', 'T1s', 'T2s', 'freq_offsets_hz', 'exchange_rates_k_to_from']:
            if key in tissue_params:
                tissue_params[key] = tissue_params[key].to(self.device, dtype=self.dtype)
            else:
                raise ValueError(f"tissue_params missing required key: {key}")


        N_timepoints_pulse = rf_amp_waveform_tesla.shape[0]
        if rf_freq_waveform_hz.shape[0] != N_timepoints_pulse:
            raise ValueError("RF amplitude and frequency waveforms must have the same number of timepoints.")

        # Check consistency of pulse duration, number of timepoints, and dt_s
        # It's assumed that the input waveforms are sampled at intervals of dt_s
        expected_duration = N_timepoints_pulse * dt_s
        if abs(pulse_duration_s - expected_duration) > 1e-7: # Allow small tolerance
             print(f"Warning: pulse_duration_s ({pulse_duration_s:.2e}) does not match N_timepoints_pulse * dt_s ({expected_duration:.2e}). Ensure waveforms are sampled at dt_s.")

        result_profile = torch.zeros(
            (len(target_b1_scales), len(target_b0_offsets_hz)),
            device=self.device, dtype=self.dtype
        )

        for b1_idx, b1_scale in enumerate(target_b1_scales):
            for b0_idx, global_b0_offset_hz_val in enumerate(target_b0_offsets_hz):
                if initial_M_vector_flat is None:
                    M_current_flat = torch.zeros(num_pools * 3, device=self.device, dtype=self.dtype)
                    for p in range(num_pools):
                        M_current_flat[3 * p + 2] = tissue_params['M0_fractions'][p]
                else:
                    M_current_flat = initial_M_vector_flat.clone().to(self.device, dtype=self.dtype)

                for t_idx in range(N_timepoints_pulse):
                    current_rf_amp_scaled = rf_amp_waveform_tesla[t_idx] * b1_scale

                    # b0_offset for bloch_mcconnell_step is the offset from the RF pulse's current frequency.
                    # rf_freq_waveform_hz[t_idx] is the RF's instantaneous frequency relative to scanner 0 Hz.
                    # global_b0_offset_hz_val is the subject's global off-resonance.
                    # So, the offset experienced by the spin system relative to the RF's frequency is:
                    # global_b0_offset_hz_val - rf_freq_waveform_hz[t_idx]
                    # This value is then used as the 'b0_offset_hz' for bloch_mcconnell_step,
                    # and pool_params['freq_offsets_hz'] are applied on top of this within the solver.
                    effective_b0_for_solver_hz = global_b0_offset_hz_val - rf_freq_waveform_hz[t_idx]

                    # b1_complex_tesla for bloch_mcconnell_step is the B1 field in the rotating frame
                    # defined by the RF pulse's own frequency.
                    # Assuming rf_amp_waveform_tesla is purely amplitude (real).
                    b1_complex_val = complex(current_rf_amp_scaled.item(), 0.0)

                    M_current_flat = bloch_mcconnell_step(
                        M_vector_flat_initial=M_current_flat,
                        dt_s=dt_s,
                        b1_complex_tesla=b1_complex_val,
                        b0_offset_hz=effective_b0_for_solver_hz,
                        gyromagnetic_ratio_hz_t=self.gyromagnetic_ratio_hz_t,
                        pool_params=tissue_params
                    )

                # Store Mz of the first pool (typically water)
                result_profile[b1_idx, b0_idx] = M_current_flat[2]

        return result_profile

if __name__ == '__main__':
    from mri_pulse_library.rf_pulses.adiabatic.waveforms import generate_hs_pulse_waveforms
    import traceback # For more detailed error messages
    print("Adiabatic CEST Simulator class defined.")

    # Example Parameters
    dev = torch.device('cpu')
    sim = AdiabaticCESTSimulator(device=dev)

    # Adiabatic Pulse (Hyperbolic Secant)
    duration = 15e-3  # 15 ms
    n_points_rf = 1000 # Number of points in RF waveform
    actual_dt_s = duration / n_points_rf # Simulation dt must match RF waveform sampling

    b1_max_ut = 10.0  # uT
    bw_khz = 6.0     # kHz
    beta_hs = 8.0

    time_vec, amp_vec_tesla, freq_vec_hz = generate_hs_pulse_waveforms(
        pulse_duration_s=duration,
        num_timepoints=n_points_rf,
        peak_b1_tesla=b1_max_ut * 1e-6, # Convert uT to T
        bandwidth_hz=bw_khz * 1e3,    # Convert kHz to Hz
        beta=beta_hs,
        device=dev
    )

    # Tissue Parameters (2-Pool: Water and CEST agent)
    # Using example values, ensure M0_fractions are relative if total M0 is implicitly 1.
    # Here, M0_water = 1.0, M0_cest = 0.01 (i.e. CEST pool is 1% of water pool size)
    m0_water_fraction = 1.0
    m0_cest_fraction_relative_to_water = 0.01

    pool_params_example = {
        'num_pools': 2,
        'M0_fractions': torch.tensor([m0_water_fraction, m0_cest_fraction_relative_to_water], device=dev),
        'T1s': torch.tensor([1.330, 1.0], device=dev),      # s (Lufkin, JMRI 2019, 3T values for WM)
        'T2s': torch.tensor([0.080, 0.010], device=dev),    # s (Lufkin, JMRI 2019)
        # Example: Amide protons at 3.5 ppm. At 3T (proton Larmor freq ~127.74 MHz):
        # Offset = 3.5 ppm * 127.74 MHz = 3.5 * 127.74 Hz = 447.09 Hz
        'freq_offsets_hz': torch.tensor([0.0, 447.09], device=dev),
        'exchange_rates_k_to_from': torch.tensor([
            [0.0, 25.0],  # k_aw: To water(a) from CEST(w) = 25 Hz (example rate)
            [0.0, 0.0]    # k_wa: To CEST(w) from water(a) - will be set by detailed balance
        ], device=dev)
    }
    # Detailed balance: k_wa * M0a = k_aw * M0w (here a=water, w=CEST pool)
    # k_wa = k_aw * M0w / M0a
    k_aw = pool_params_example['exchange_rates_k_to_from'][0,1].item() # Rate from CEST to water
    m0a_val = pool_params_example['M0_fractions'][0].item() # M0 of water
    m0w_val = pool_params_example['M0_fractions'][1].item() # M0 of CEST

    if m0a_val > 1e-9: # Avoid division by zero if water M0 is zero
        k_wa = k_aw * m0w_val / m0a_val
    else:
        k_wa = 0.0 # Or some other appropriate value if M0_water is effectively zero
    pool_params_example['exchange_rates_k_to_from'][1,0] = k_wa


    # Simulation Ranges
    b1_scales_test = torch.linspace(0.5, 1.5, 3, device=dev)    # Test 3 B1 scales
    b0_offsets_test_hz = torch.linspace(-100, 100, 5, device=dev) # Test 5 B0 offsets

    print(f"Simulating HS pulse response for {pool_params_example['num_pools']} pools...")
    print(f"Using {n_points_rf} timepoints for RF, dt = {actual_dt_s*1e6:.2f} us.")
    print(f"B1 scales: {b1_scales_test.tolist()}")
    print(f"B0 offsets (Hz): {b0_offsets_test_hz.tolist()}")
    print(f"CEST pool @ {pool_params_example['freq_offsets_hz'][1].item()} Hz, k_exchange_to_water={k_aw} Hz, k_exchange_to_cest={k_wa:.2f} Hz")

    try:
        saturation_profile = sim.simulate_pulse_response(
            rf_amp_waveform_tesla=amp_vec_tesla,
            rf_freq_waveform_hz=freq_vec_hz,
            pulse_duration_s=duration,
            dt_s=actual_dt_s,
            target_b1_scales=b1_scales_test,
            target_b0_offsets_hz=b0_offsets_test_hz,
            tissue_params=pool_params_example
        )
        print(f"\nSimulation complete. Saturation profile shape: {saturation_profile.shape}")
        print("Example saturation values (Mz of water pool):")
        # Format output for better readability
        for b1_idx, b1_val in enumerate(b1_scales_test):
            row_str = f"B1 scale {b1_val:.2f}: ["
            for b0_idx, b0_val in enumerate(b0_offsets_test_hz):
                row_str += f"{saturation_profile[b1_idx, b0_idx]:.4f}"
                if b0_idx < len(b0_offsets_test_hz) - 1:
                    row_str += ", "
            row_str += "]"
            print(row_str)

    except Exception as e:
        print(f"An error occurred during simulation example: {e}")
        traceback.print_exc()
