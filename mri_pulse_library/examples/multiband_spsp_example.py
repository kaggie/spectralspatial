# mri_pulse_library/examples/multiband_spsp_example.py
"""
Example script demonstrating the design and simulation of a
multi-band spectral-spatial (SPSP) RF pulse.
"""

# Example usage (to be filled in later):
# import numpy as np
# from mri_pulse_library import rf_pulses
# from mri_pulse_library import simulators
# from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

# def run_multiband_spsp_example():
#     # 1. Define SPSP parameters
#     duration = 10e-3 # s
#     total_flip_angle_deg = 90.0
#     n_subpulses = 32

    # Spatial parameters (e.g., 2 slices)
#     spatial_bandwidths_hz = [2000, 2000] # Hz per slice
#     slice_positions_m = [-0.01, 0.01] # 1cm apart
#     slice_select_gradient_Tm = 0.01 # T/m

    # Spectral parameters (e.g., water and fat, or two arbitrary bands)
#     spectral_bandwidths_hz = [200, 200] # Hz per spectral band (note: current SPSP design might not use this for shaping)
#     spectral_center_freqs_hz = [0, 440] # Hz offset from carrier (e.g., on-resonance and a fat peak at 3T)

    # 2. Generate the 3D multiband SPSP pulse
#     spsp_pulse_T, time_s = rf_pulses.spectral_spatial.generate_3d_multiband_spsp(
#         duration=duration,
#         spatial_bandwidths_hz=spatial_bandwidths_hz,
#         slice_select_gradient_Tm=slice_select_gradient_Tm,
#         slice_positions_m=slice_positions_m,
#         spectral_bandwidths_hz=spectral_bandwidths_hz,
#         spectral_center_freqs_hz=spectral_center_freqs_hz,
#         flip_angle_deg=total_flip_angle_deg,
#         n_subpulses=n_subpulses
#     )
#     print(f"Generated 3D multiband SPSP pulse with {len(spsp_pulse_T)} samples.")

    # 3. Simulate its profile
#     sim_z_positions_m = np.linspace(-0.03, 0.03, 100) # Simulate over a wider spatial range
#     sim_freq_offsets_hz = np.linspace(-600, 600, 200)  # Simulate over a wider spectral range

#     M_profile = simulators.simulate_3d_multiband_spsp(
#         rf_pulse_T=spsp_pulse_T,
#         time_s=time_s,
#         slice_select_gradient_Tm=slice_select_gradient_Tm,
#         z_positions_m=sim_z_positions_m,
#         freq_offsets_hz=sim_freq_offsets_hz
#     )
#     print(f"Simulated profile shape: {M_profile.shape}") # Should be (Nz, Nf, 3)

    # 4. Visualization (basic, actual plotting would require matplotlib)
    # print("Example Mz values at center of first slice, on-resonance peak:")
    # slice1_idx = np.argmin(np.abs(sim_z_positions_m - slice_positions_m[0]))
    # freq1_idx = np.argmin(np.abs(sim_freq_offsets_hz - spectral_center_freqs_hz[0]))
    # print(f"Mz(slice1, freq1): {M_profile[slice1_idx, freq1_idx, 2]:.3f}")

    # print("Example Mz values at center of second slice, second spectral peak:")
    # slice2_idx = np.argmin(np.abs(sim_z_positions_m - slice_positions_m[1]))
    # freq2_idx = np.argmin(np.abs(sim_freq_offsets_hz - spectral_center_freqs_hz[1]))
    # print(f"Mz(slice2, freq2): {M_profile[slice2_idx, freq2_idx, 2]:.3f}")

# if __name__ == "__main__":
#     run_multiband_spsp_example()
pass
