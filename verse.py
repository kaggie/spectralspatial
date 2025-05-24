import torch
import numpy as np

def apply_basic_verse(
    rf_segment: torch.Tensor, 
    gradient_sub_lobe: torch.Tensor, 
    original_gradient_amplitude_of_rf_segment: float, 
    dt: float
) -> torch.Tensor:
    """
    Applies basic VERSE (Variable-Rate Selective Excitation) to an RF segment.
    This function time-stretches and amplitude-modulates an RF segment based on a 
    time-varying gradient sub-lobe, assuming the RF was designed for a constant 
    gradient amplitude.

    Args:
        rf_segment: 1D PyTorch Tensor (complex), the RF samples designed for a 
                    constant gradient amplitude.
        gradient_sub_lobe: 1D PyTorch Tensor (real), the actual time-varying 
                           gradient sub-lobe under which the RF will be played.
                           Length of this defines the output RF length.
        original_gradient_amplitude_of_rf_segment: Float, the constant gradient 
                                                   amplitude for which rf_segment 
                                                   was initially designed.
        dt: Float, time step in seconds.

    Returns:
        A 1D PyTorch Tensor (complex) for the VERSEd RF segment, same length as 
        gradient_sub_lobe.
    """
    device = rf_segment.device
    output_length = len(gradient_sub_lobe)

    if output_length == 0:
        return torch.tensor([], dtype=rf_segment.dtype, device=device)
        
    if abs(original_gradient_amplitude_of_rf_segment) < 1e-9 or torch.all(torch.abs(gradient_sub_lobe) < 1e-9):
        # If original gradient is zero, or target gradient is all zeros,
        # VERSE is ill-defined or results in zero RF.
        # If rf_segment is non-zero but gradients are zero, it implies infinite time scale or zero amplitude.
        # Return zeros matching gradient_sub_lobe length.
        return torch.zeros(output_length, dtype=rf_segment.dtype, device=device)

    if rf_segment.numel() == 0: # No input RF
        return torch.zeros(output_length, dtype=rf_segment.dtype, device=device)

    # Time scaling factor: G_orig / G_actual(t)
    # Handle potential division by zero in gradient_sub_lobe
    grad_actual_safe = gradient_sub_lobe.clone()
    # Smallest non-zero positive float for PyTorch tensor of same dtype
    epsilon = torch.finfo(grad_actual_safe.dtype).tiny 
    
    # Replace zeros with epsilon, maintaining sign if original_gradient_amplitude is also signed.
    # However, gradient_sub_lobe and original_gradient_amplitude are usually magnitudes for VERSE scaling.
    # Let's assume inputs represent magnitudes for scaling, or signs are handled upstream.
    # For safety, use abs for scaling calculation, but ensure signs are consistent if needed.
    # The problem implies G_orig / G_actual(t). If G_actual(t) can be negative, time_scale_factor can be negative.
    # Typically, VERSE uses magnitudes: |G_orig / G_actual(t)|.
    # Let's assume gradient_sub_lobe represents the amplitude profile that rf_segment will experience.
    # If gradient_sub_lobe can have negative values, time_scale_factor might become negative.
    # The physical meaning of negative time_scale_factor in cumulative sum of dt/time_scale_factor is problematic.
    # ss_verse.m uses gwave = abs(gwave) for calculating time_scale_factor (tau).
    
    grad_magnitude_actual_safe = torch.abs(gradient_sub_lobe)
    grad_magnitude_actual_safe[grad_magnitude_actual_safe < epsilon] = epsilon # Avoid division by zero

    original_grad_amp_abs = abs(original_gradient_amplitude_of_rf_segment)

    time_scale_factor = original_grad_amp_abs / grad_magnitude_actual_safe
    
    # Amplitude scaling factor: G_actual(t) / G_orig
    # This will be applied to the interpolated RF.
    # Note: if gradient_sub_lobe can be negative, this correctly flips RF phase.
    amplitude_scale_factor = gradient_sub_lobe / original_gradient_amplitude_of_rf_segment
    amplitude_scale_factor[torch.abs(torch.tensor(original_gradient_amplitude_of_rf_segment)) < epsilon] = 0.0 # Avoid div by zero if G_orig is zero


    # Original time vector for rf_segment (used for interpolation query points)
    # t_original_rf_pts corresponds to the *cumulative scaled time* from the original RF pulse
    # if it were played under the new gradient.
    # Each sample of the original RF pulse lasts dt. Under VERSE, it lasts dt * time_scale_factor_at_that_new_point.
    # This is complex. Let's use the standard VERSE integration:
    # Integral(G_orig dt_orig) = Integral(G_verse(t_verse) dt_verse)
    # dt_orig / dt_verse = G_verse(t_verse) / G_orig
    # So, the original RF's time axis is compressed/stretched.
    # New time points for sampling the original rf_segment:
    # t_map[k] = sum_{i=0 to k-1} (dt * G_orig / G_verse[i])
    # No, it's sum_{i=0 to k-1} (dt * G_verse[i] / G_orig)
    # From ss_verse.m: tau = dt ./ time_scale_factor = dt * abs(gradient_sub_lobe) / original_grad_amp_abs
    # t_stretched_cumulative = cumsum(tau) - tau[0] (to start from 0)
    
    # dt_prime is the duration each sample of the *new* (VERSED) pulse effectively maps to
    # in the *original* pulse's timeline.
    # dt_prime(t_new) = dt_new * (G_new(t_new) / G_orig)
    dt_prime = dt * (gradient_sub_lobe / original_gradient_amplitude_of_rf_segment)
    # Handle cases where original_gradient_amplitude_of_rf_segment is zero
    dt_prime[abs(torch.tensor(original_gradient_amplitude_of_rf_segment)) < epsilon] = 0.0

    # Cumulative time on the original RF pulse's timeline
    # These are the points in original RF time that correspond to each sample of the new VERSEd RF.
    t_map_to_original_rf = torch.cumsum(dt_prime, dim=0) - dt_prime[0] # Start sampling original RF from its t=0

    # Time vector for the original rf_segment (on which it is defined)
    t_original_rf_defined = torch.arange(rf_segment.numel(), device=device, dtype=dt_prime.dtype) * dt

    # Interpolate real and imaginary parts
    # np.interp needs x-coordinates (t_map_to_original_rf) to be monotonically increasing.
    # If gradient_sub_lobe changes sign, dt_prime can be negative, making t_map_to_original_rf non-monotonic.
    # This implies that VERSE typically assumes G_new(t_new)/G_orig is positive, or uses magnitudes.
    # Let's assume G_orig is positive and G_new(t) is also positive for calculating t_map.
    # The amplitude_scale_factor can still carry the sign.
    
    dt_prime_positive_grad = dt * torch.abs(gradient_sub_lobe) / original_grad_amp_abs
    dt_prime_positive_grad[abs(original_grad_amp_abs) < epsilon] = float('inf') # effectively makes RF zero if interpolated here

    t_map_to_original_rf_monotonic = torch.cumsum(dt_prime_positive_grad, dim=0) - dt_prime_positive_grad[0]

    # Ensure t_map_to_original_rf_monotonic does not exceed the defined range of t_original_rf_defined
    # Clamp values to the range of t_original_rf_defined to handle extrapolation behavior of interp
    max_original_time = t_original_rf_defined[-1] if t_original_rf_defined.numel() > 0 else 0.0
    t_map_to_original_rf_clamped = torch.clamp(t_map_to_original_rf_monotonic, 0.0, max_original_time)

    # Perform interpolation using numpy as PyTorch's interpolate is for image-like data
    # and complex interpolation isn't direct.
    rf_segment_np_real = rf_segment.real.cpu().numpy()
    rf_segment_np_imag = rf_segment.imag.cpu().numpy()
    t_original_rf_defined_np = t_original_rf_defined.cpu().numpy()
    t_map_to_original_rf_clamped_np = t_map_to_original_rf_clamped.cpu().numpy()

    # Handle empty t_original_rf_defined_np (e.g. if rf_segment has 1 point)
    if t_original_rf_defined_np.size == 0: # Should not happen if rf_segment.numel() > 0
        if rf_segment.numel() == 1: # Single point RF
            rf_interp_real_np = np.full_like(t_map_to_original_rf_clamped_np, rf_segment_np_real.item())
            rf_interp_imag_np = np.full_like(t_map_to_original_rf_clamped_np, rf_segment_np_imag.item())
        else: # Should be caught by rf_segment.numel() == 0 earlier
            rf_interp_real_np = np.zeros_like(t_map_to_original_rf_clamped_np)
            rf_interp_imag_np = np.zeros_like(t_map_to_original_rf_clamped_np)
    elif t_original_rf_defined_np.size == 1: # Original RF is a single point
        rf_interp_real_np = np.full_like(t_map_to_original_rf_clamped_np, rf_segment_np_real[0])
        rf_interp_imag_np = np.full_like(t_map_to_original_rf_clamped_np, rf_segment_np_imag[0])
    else:
        rf_interp_real_np = np.interp(t_map_to_original_rf_clamped_np, t_original_rf_defined_np, rf_segment_np_real)
        rf_interp_imag_np = np.interp(t_map_to_original_rf_clamped_np, t_original_rf_defined_np, rf_segment_np_imag)

    rf_interp_real = torch.tensor(rf_interp_real_np, dtype=rf_segment.real.dtype, device=device)
    rf_interp_imag = torch.tensor(rf_interp_imag_np, dtype=rf_segment.imag.dtype, device=device)
    rf_interpolated = torch.complex(rf_interp_real, rf_interp_imag)

    # Amplitude adjustment
    # rf_v(t_new) = rf_orig(t_mapped) * (G_new(t_new) / G_orig)
    rf_versed = rf_interpolated * amplitude_scale_factor
    
    # Ensure output length matches gradient_sub_lobe
    if rf_versed.numel() != output_length:
        # This might happen if interpolation somehow changes numel, though unlikely with np.interp
        # For safety, ensure it matches. This usually means an error in logic if sizes mismatch.
        # However, np.interp output size is same as t_map_to_original_rf_clamped_np which is output_length.
        pass

    return rf_versed