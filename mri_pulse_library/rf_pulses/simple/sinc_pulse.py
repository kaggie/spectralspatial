# File: mri_pulse_library/rf_pulses/simple/sinc_pulse.py
import numpy as np
from mri_pulse_library.core.constants import GAMMA_HZ_PER_T_PROTON

def generate_sinc_pulse(duration, bandwidth, flip_angle_deg, n_lobes=3, gyromagnetic_ratio_hz_t=GAMMA_HZ_PER_T_PROTON, dt=1e-6):
    """
    Generates a sinc RF pulse.

    Args:
        duration (float): Pulse duration in seconds (center of pulse to last zero-crossing or truncation).
        bandwidth (float): Frequency bandwidth in Hz (typically full width of the main lobe).
        flip_angle_deg (float): Desired flip angle in degrees.
        n_lobes (int, optional): Number of sinc lobes to include on each side of the main lobe.
                                 Total lobes = 2 * n_lobes. Defaults to 3.
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
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive.")
    if n_lobes < 0: # n_lobes = 0 could mean just the main lobe, but definition is usually for side lobes.
        raise ValueError("Number of lobes (n_lobes) must be non-negative.")

    # Time vector generation: (np.arange(N) - (N-1)/2)*dt for symmetry
    # num_samples should ensure the duration is covered.
    # If duration is total (e.g. -duration/2 to duration/2), then num_samples = duration/dt + 1
    # Or, num_samples = round(duration/dt) and adjust time to be symmetric.
    num_samples = int(round(duration / dt))
    if num_samples == 0 and duration > 0: # Ensure at least one sample if duration is very small
        num_samples = 1

    # Create a time vector centered at 0
    # time = (np.arange(num_samples) - (num_samples - 1) / 2) * dt
    # This creates num_samples points. If num_samples is odd, 0 is included.
    # If num_samples is even, 0 is halfway between two points.
    # Example: N=5 -> [-2, -1, 0, 1, 2]*dt. Example: N=4 -> [-1.5, -0.5, 0.5, 1.5]*dt
    time = np.linspace(-duration/2, duration/2, num_samples, endpoint=True)


    # Sinc envelope: np.sinc(x) = sin(pi*x)/(pi*x)
    # The argument 'x' for np.sinc should represent time normalized by the lobe duration.
    # A common definition for a sinc pulse with 'n_lobes' (meaning n zero-crossings on each side of center):
    # The time for one lobe (distance between zero crossings) is t_lobe = duration / (2 * n_lobes) if duration is full extent.
    # Or, if bandwidth is FWZM, and it corresponds to n_lobes, then sinc argument is time * (n_lobes / duration_main_lobe)
    # The prompt uses `sinc_arg = (bandwidth / n_lobes) * time`. This seems to imply that `bandwidth / n_lobes` is a frequency.
    # Let's use a common definition: time-bandwidth product (TBW) = duration * bandwidth.
    # For a sinc pulse, TBW often refers to the number of lobes * 2, or related to it.
    # If bandwidth is FWZM of main lobe, and pulse has N zero crossings (N/2 lobes on each side):
    # sinc(t * N / duration) or sinc(t * bandwidth_nominal).
    # Let's assume `n_lobes` refers to the number of zero crossings from center to edge of pulse.
    # So, argument for np.sinc should be `time * n_lobes / (duration / 2)`.
    # This makes `np.sinc(n_lobes)` at `t = duration/2`.
    # Or, if bandwidth is defined as `k * n_lobes / duration`, then `sinc(time * bandwidth / k_factor)`.
    # The prompt's `sinc_arg = (bandwidth / n_lobes) * time` suggests `bandwidth/n_lobes` is the effective frequency for normalization.
    # If `n_lobes` is the number of side lobes (e.g., 3 means 3 on each side of main), then total zero crossings might be related.
    # A typical sinc pulse is sinc(t/T) where T is related to lobe width.
    # If `n_lobes` means the argument to sinc goes from -n_lobes to +n_lobes over the duration:
    # x = n_lobes * (2 * time / duration)
    # sinc_env = np.sinc(x)
    # This makes sinc_env = np.sinc(n_lobes) at time = duration/2.

    if n_lobes == 0: # Special case: rect function in freq domain, so infinite sinc in time. Not practical.
                     # Or, could mean just the main lobe of a sinc.
                     # Let's assume n_lobes refers to the argument scaling factor.
                     # If n_lobes is used as per `sinc(bandwidth * time / n_lobes)`, it must be non-zero.
        effective_n_lobes = 1 # Avoid division by zero, or define behavior for n_lobes=0
    else:
        effective_n_lobes = n_lobes

    # The interpretation of 'bandwidth' and 'n_lobes' for sinc is crucial.
    # Standard sinc for slice selection: B(t) = sinc(2*pi*W*t) where W is related to bandwidth.
    # np.sinc(x) means sin(pi*x)/(pi*x). So x = 2*W*t.
    # If 'bandwidth' is full width of main lobe (between first zero crossings), then BW = 2/T, where T is pulse duration for main lobe.
    # If duration is the full pulse length, and it contains `effective_n_lobes` on each side of center:
    # sinc_arg = time * (effective_n_lobes * 2 / duration) # makes argument go up to effective_n_lobes at t=duration/2
    # This is equivalent to `time / (duration / (2 * effective_n_lobes))`.
    # Let's use: `sinc_arg = time * effective_n_lobes * 2 / duration`
    # This way, `effective_n_lobes` directly controls the number of zero crossings in `[-duration/2, duration/2]`.
    # For example, if effective_n_lobes = 3, then sinc_arg goes from -3 to 3. np.sinc will have 3 zero crossings on each side of 0.

    sinc_arg = time * effective_n_lobes * 2 / duration
    sinc_env = np.sinc(sinc_arg)

    # Windowing is often applied to sinc pulses (e.g., Hamming) but not requested here.

    # Normalize to achieve desired flip angle
    # flip_angle_rad = gamma_rad_s_t * B1_peak * sum(sinc_env) * dt
    current_area = np.sum(sinc_env) * dt
    flip_angle_rad = np.deg2rad(flip_angle_deg)
    gyromagnetic_ratio_rad_s_t = gyromagnetic_ratio_hz_t * 2 * np.pi

    if abs(current_area * gyromagnetic_ratio_rad_s_t) < 1e-20: # Avoid division by zero if area is effectively zero
        # This can happen if sinc_env sums to zero (e.g. odd number of half-lobes and cancellation)
        # Or if gyromagnetic ratio is zero.
        if flip_angle_rad == 0:
            B1_amplitude_scalar = 0.0
        else:
            # A non-zero flip angle is requested, but the shape's integral is zero.
            # This implies an infinite B1_amplitude_scalar, which is an issue.
            # This can happen if the sinc pulse is perfectly symmetric around a zero-crossing for its integral.
            # For np.sinc, sum(sinc_env) is usually positive for typical lobe counts.
            raise ValueError(f"Cannot achieve non-zero flip angle. Pulse shape integral is near zero (Area: {current_area}). Check duration and n_lobes.")
    else:
        B1_amplitude_scalar = flip_angle_rad / (gyromagnetic_ratio_rad_s_t * current_area)

    rf_pulse = B1_amplitude_scalar * sinc_env # Resulting B1 in Tesla

    return rf_pulse, time
