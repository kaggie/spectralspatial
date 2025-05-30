# File: mri_pulse_library/rf_pulses/simple/gaussian_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_gaussian_pulse(duration, bandwidth, flip_angle_deg, sigma_factor=2.5, gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON, dt=1e-6):
    """
    Generates a Gaussian RF pulse.

    Args:
        duration (float): Pulse duration in seconds (defines the time window).
        bandwidth (float): Frequency bandwidth in Hz. This is often interpreted as FWHM.
                           The relationship between bandwidth and sigma depends on convention.
                           Here, we use sigma_factor to define sigma from duration.
                           Bandwidth parameter is not directly used in this sigma definition,
                           but is kept for signature consistency if interpretation changes.
        flip_angle_deg (float): Desired flip angle in degrees.
        sigma_factor (float, optional): Controls Gaussian width. sigma = duration / sigma_factor.
                                      A smaller sigma_factor gives a wider Gaussian relative to duration.
                                      A larger sigma_factor gives a narrower Gaussian (more truncated by duration).
                                      Defaults to 2.5, which truncates at +/- 2.5 sigma.
        gyromagnetic_ratio_hz_t (float, optional): Gyromagnetic ratio in Hz/T.
                                                   Defaults to GAMMA_HZ_PER_T_PROTON.
        dt (float, optional): Time step in seconds. Defaults to 1e-6 s (1 µs).

    Returns:
        tuple: (rf_pulse, time_vector)
            rf_pulse (np.ndarray): RF waveform in Tesla.
            time_vector (np.ndarray): Time points in seconds, centered around 0.
    """
    if duration <= 0:
        return np.array([]), np.array([])
    # Bandwidth is not directly used if sigma is defined by duration and sigma_factor.
    # Could add a check: if bandwidth is for FWHM, FWHM = 2*sigma*sqrt(2*ln(2)).
    # sigma = FWHM / (2*sqrt(2*ln(2))) approx FWHM / 2.3548
    # If this interpretation is desired, sigma_factor usage would change.
    # For now, proceeding with sigma = duration / sigma_factor as per prompt's description.

    num_samples = int(round(duration / dt))
    if num_samples == 0 and duration > 0:
        num_samples = 1

    time = np.linspace(-duration/2, duration/2, num_samples, endpoint=True)

    if sigma_factor == 0:
        # A sigma_factor of 0 would mean sigma is infinite (flat pulse if duration is finite)
        # or undefined if duration is also zero.
        # Let's treat it as an unphysical input or default to a reasonable value.
        raise ValueError("sigma_factor cannot be zero.")

    sigma = duration / sigma_factor

    if sigma == 0: # This happens if duration is zero (already handled) or sigma_factor is infinite.
                   # If sigma_factor is very large, sigma becomes very small (narrow spike).
        if duration > 0: # Duration is non-zero, but sigma is effectively zero
            # Approximate a Dirac delta pulse: single point at center with area 1 (before scaling for flip angle)
            gaussian_env = np.zeros_like(time)
            if num_samples > 0:
                gaussian_env[num_samples // 2] = 1.0 # Arbitrary height, area normalization will handle it.
                                                 # Sum will be 1.0, dt_effective is dt.
                                                 # So current_area = 1.0 * dt if we define it this way.
                                                 # Or, more simply, if sigma is zero, B1 is applied at a single point.
                                                 # This case might need more careful definition for flip angle.
                                                 # Let's assume it implies a very sharp pulse, area will be small.
                                                 # If exp becomes problematic, handle it.
                                                 # A true zero sigma is problematic for exp.
                if len(time) > 0 : gaussian_env[len(time)//2] = 1.0
                else: gaussian_env = np.array([]) # Should not happen if num_samples > 0
        else: # duration is zero
            gaussian_env = np.array([]) # Should be caught by initial duration check
    else:
        gaussian_env = np.exp(-time**2 / (2 * sigma**2))

    if gaussian_env.size == 0 and duration > 0: # Should not happen with linspace if num_samples >=1
        return np.array([]), np.array([]) # Safety

    # Normalize for flip angle
    current_area = np.sum(gaussian_env) * dt # Integral of the shape
    flip_angle_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if abs(current_area * gyromagnetic_ratio_rad_s_t) < 1e-20:
        if flip_angle_rad == 0:
            B1_amplitude_scalar = 0.0
        else:
            # This could happen if gaussian_env is all zeros (e.g. extreme sigma_factor and duration)
            raise ValueError(f"Cannot achieve non-zero flip angle. Pulse shape integral is near zero (Area: {current_area}). Check duration and sigma_factor.")
    else:
        B1_amplitude_scalar = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * current_area)

    rf_pulse = B1_amplitude_scalar * gaussian_env # Resulting B1 in Tesla

    return rf_pulse, time
