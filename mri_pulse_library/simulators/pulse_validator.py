# File: mri_pulse_library/simulators/pulse_validator.py
# Placeholder for pulse validation logic
# This module would compare simulated magnetization profiles
# (e.g., slice profiles, spectral selectivity) against desired targets.

import numpy as np

class PulseValidationMetrics:
    """
    A class to hold various validation metrics for an RF pulse.
    """
    def __init__(self):
        self.slice_thickness_m = None
        self.slice_ripple_percent = None
        self.transition_width_m = None
        self.spectral_fwhm_hz = None
        self.inversion_efficiency_percent = None
        # Add more metrics as needed

    def __str__(self):
        metrics_str = []
        if self.slice_thickness_m is not None:
            metrics_str.append(f"Slice Thickness (FWHM): {self.slice_thickness_m*1000:.2f} mm")
        if self.slice_ripple_percent is not None:
            metrics_str.append(f"Slice Ripple: {self.slice_ripple_percent:.2f}%")
        if self.transition_width_m is not None:
            metrics_str.append(f"Transition Width: {self.transition_width_m*1000:.2f} mm")
        if self.spectral_fwhm_hz is not None:
            metrics_str.append(f"Spectral FWHM: {self.spectral_fwhm_hz:.2f} Hz")
        if self.inversion_efficiency_percent is not None:
            metrics_str.append(f"Inversion Efficiency: {self.inversion_efficiency_percent:.2f}%")

        return "\n".join(metrics_str) if metrics_str else "No metrics calculated."


def analyze_slice_profile(mz_profile, z_positions_m, target_mz=-1.0):
    """
    Analyzes a 1D Mz slice profile to extract metrics like FWHM thickness and ripple.

    Args:
        mz_profile (np.ndarray): Array of Mz values along the z-axis.
        z_positions_m (np.ndarray): Corresponding z positions in meters.
        target_mz (float): The target Mz value for full excitation/inversion (e.g., -1 for inversion).

    Returns:
        dict: A dictionary containing 'fwhm_m' (Full Width at Half Maximum in meters)
              and 'ripple_percent'.
    """
    if mz_profile.size == 0 or z_positions_m.size == 0 or mz_profile.shape != z_positions_m.shape:
        # print("Warning: Invalid input for slice profile analysis.")
        return {"fwhm_m": 0, "ripple_percent": 0, "transition_width_m":0}

    # Normalize Mz profile if needed, assuming it's relative to M0=1
    # For inversion, profile goes from +1 to -1. For excitation, from +1 to 0 (for Mz).
    # This function assumes the profile is already Mz values.

    # FWHM calculation
    # Find where profile crosses half max/min. For inversion, half-way between Mz0 (1) and target_mz (-1) is 0.
    # For 90-deg excitation (Mz goes from 1 to 0), half-max is 0.5.
    # Let's define half-max based on the range of Mz values achieved.
    min_mz = np.min(mz_profile)
    max_mz_in_slice = np.max(mz_profile) # Could be different from initial Mz0 if not perfect pulse.

    if target_mz < 0: # Inversion-like pulse
        half_value = (1.0 + min_mz) / 2.0
        # Find points where profile crosses this value
        # Ensure it's sorted by z_positions_m
        sorted_indices = np.argsort(z_positions_m)
        z_sorted = z_positions_m[sorted_indices]
        mz_sorted = mz_profile[sorted_indices]

        above_half = mz_sorted > half_value # For inversion, we look for where it's *less* inverted than half
    else: # Excitation-like pulse (Mz from 1 to 0 or some positive value)
        half_value = (1.0 + min_mz) / 2.0 # Halfway between initial Mz=1 and minimum Mz reached
        sorted_indices = np.argsort(z_positions_m)
        z_sorted = z_positions_m[sorted_indices]
        mz_sorted = mz_profile[sorted_indices]
        above_half = mz_sorted < half_value # For excitation, where it's *more* excited than half

    fwhm_indices = np.where(np.diff(above_half))[0]
    fwhm_m = 0
    if len(fwhm_indices) >= 2:
        # Interpolate to find more accurate crossing points
        z1 = np.interp(half_value, [mz_sorted[fwhm_indices[0]], mz_sorted[fwhm_indices[0]+1]], [z_sorted[fwhm_indices[0]], z_sorted[fwhm_indices[0]+1]])
        z2 = np.interp(half_value, [mz_sorted[fwhm_indices[-1]], mz_sorted[fwhm_indices[-1]+1]], [z_sorted[fwhm_indices[-1]], z_sorted[fwhm_indices[-1]+1]])
        fwhm_m = abs(z2 - z1)

    # Ripple: std deviation within the passband (e.g., where Mz is close to target_mz)
    # Define passband, e.g., within FWHM or where profile is > 90% of peak excitation/inversion
    passband_thresh = min_mz + 0.1 * abs(min_mz - 1.0) if target_mz < 0 else min_mz + 0.9 * abs(1.0 - min_mz)

    if target_mz < 0: # Inversion
        in_passband = mz_sorted <= passband_thresh
    else: # Excitation
        in_passband = mz_sorted <= passband_thresh # Mz values are lower when excited

    ripple_percent = 0
    if np.any(in_passband):
        mz_passband = mz_sorted[in_passband]
        # Ripple relative to the mean/target value in the passband
        mean_passband_mz = np.mean(mz_passband)
        # ripple_percent = (np.std(mz_passband) / abs(mean_passband_mz)) * 100 if abs(mean_passband_mz) > 1e-6 else 0
        # Ripple as peak-to-peak variation relative to M0 (which is 1 or 2 if from -1 to 1)
        ripple_val = (np.max(mz_passband) - np.min(mz_passband)) / (1.0 - target_mz) * 100 # Pk-Pk as % of total Mz change
        ripple_percent = ripple_val


    # Transition width (e.g., 10% to 90% of peak)
    # For inversion from Mz=1 to Mz=-1:
    # 10% inversion: Mz = 1 - 0.1 * (1 - (-1)) = 0.8
    # 90% inversion: Mz = 1 - 0.9 * (1 - (-1)) = -0.8
    val_10_percent = 1.0 - 0.1 * (1.0 - min_mz) # e.g. if min_mz=-1, val_10 = 0.8
    val_90_percent = 1.0 - 0.9 * (1.0 - min_mz) # e.g. if min_mz=-1, val_90 = -0.8

    indices_10 = np.where(np.diff(mz_sorted < val_10_percent if target_mz < 0 else mz_sorted > val_10_percent))[0]
    indices_90 = np.where(np.diff(mz_sorted < val_90_percent if target_mz < 0 else mz_sorted > val_90_percent))[0]

    z10_edges = []
    z90_edges = []

    if len(indices_10) >=1 :
        for idx in indices_10:
             z10_edges.append(np.interp(val_10_percent, [mz_sorted[idx], mz_sorted[idx+1]], [z_sorted[idx], z_sorted[idx+1]]))
    if len(indices_90) >=1 :
        for idx in indices_90:
            z90_edges.append(np.interp(val_90_percent, [mz_sorted[idx], mz_sorted[idx+1]], [z_sorted[idx], z_sorted[idx+1]]))

    transition_width_m = 0
    if len(z10_edges)>=2 and len(z90_edges)>=2:
        # Assuming symmetric pulse, take average of left/right transition widths
        # Left transition: z90_edges[0] to z10_edges[0]
        # Right transition: z10_edges[1] to z90_edges[1]
        # Need to handle cases with multiple crossings carefully if profile is not simple box.
        # For a simple profile, expect two 10% crossings and two 90% crossings.
        # Width of left transition: abs(z_at_10%_left - z_at_90%_left)
        # Width of right transition: abs(z_at_90%_right - z_at_10%_right)
        # For now, simple:
        if fwhm_m > 0: # Only if a slice was actually selected
            # Take outer 10% and inner 90% points that define the slice edges
            try:
                left_transition = abs(sorted(z90_edges)[0] - sorted(z10_edges)[0])
                right_transition = abs(sorted(z10_edges)[-1] - sorted(z90_edges)[-1])
                transition_width_m = (left_transition + right_transition) / 2.0
            except IndexError:
                transition_width_m = 0 # Not enough points found

    return {"fwhm_m": fwhm_m, "ripple_percent": ripple_percent, "transition_width_m": transition_width_m}


def validate_pulse_performance(pulse_type, simulation_results, **kwargs):
    """
    Validates pulse performance based on simulation results.
    This is a placeholder and would need to be significantly expanded.

    Args:
        pulse_type (str): E.g., "hard", "sinc", "gaussian", "spsp", "hs1".
        simulation_results (np.ndarray): Output from the corresponding simulate_* function.
        **kwargs: Additional parameters needed for validation (e.g., z_positions_m for slice profile).

    Returns:
        PulseValidationMetrics: An object containing calculated metrics.
    """
    metrics = PulseValidationMetrics()
    # print(f"Validating {pulse_type} pulse...")

    if pulse_type in ["sinc", "gaussian", "spsp", "3d_multiband_spsp"] and "z_positions_m" in kwargs:
        # Assuming simulation_results are (Nz, Nf, 3) or (Nz, 3)
        # For simplicity, analyze the on-resonance slice profile (Nf index 0 or squeezed)
        mz_profile = simulation_results[:, 0, 2] if simulation_results.ndim == 3 else simulation_results[:, 2]
        z_positions_m = kwargs["z_positions_m"]

        analysis = analyze_slice_profile(mz_profile, z_positions_m)
        metrics.slice_thickness_m = analysis.get("fwhm_m", 0)
        metrics.slice_ripple_percent = analysis.get("ripple_percent", 0)
        metrics.transition_width_m = analysis.get("transition_width_m",0)

    elif pulse_type == "hs1":
        # Adiabatic pulses often check inversion efficiency over B1 variations
        # simulation_results is (Nb1, 3)
        mz_values = simulation_results[:, 2]
        # Assuming target is full inversion (Mz = -1 from Mz0 = 1)
        inversion_efficiency = (1 - np.mean(mz_values)) / 2 * 100 # Avg % inversion from Mz=1 to Mz=-1
        metrics.inversion_efficiency_percent = inversion_efficiency

    # Add more pulse type specific validations here

    # print(str(metrics))
    return metrics

# TODO: Implement more detailed validation metrics and comparisons.
# Examples:
# - Slice thickness based on FWHM of Mz profile.
# - Slice ripple (std dev of Mz in passband).
# - Transition width (e.g., 10%-90%).
# - Spectral selectivity (FWHM of spectral profile for SPSP).
# - Inversion/excitation efficiency for adiabatic/hard pulses.
# - B1 insensitivity for adiabatic pulses.
# - SAR calculations (would need B1rms, duration).
pass
