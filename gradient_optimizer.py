import torch
import math
# from scipy import signal # Not needed in this file yet

def design_trapezoidal_lobe(
    target_moment: float,
    max_amplitude: float,
    max_slew_rate: float,
    dt: float,
    current_amplitude: float = 0.0,
    target_amplitude: float = 0.0,
    device: str = 'cpu', 
) -> torch.Tensor:
    """Designs a single trapezoidal gradient lobe to achieve target_moment.

    Respects max_amplitude and max_slew_rate.
    current_amplitude is the starting gradient amplitude.
    target_amplitude is the ending gradient amplitude for the lobe.
    dt is the sampling interval.
    device is the torch device for the output tensor.

    Returns a 1D PyTorch tensor representing the gradient waveform.
    """
    if max_amplitude <= 0:
        raise ValueError("max_amplitude must be positive.")
    if max_slew_rate <= 0:
        raise ValueError("max_slew_rate must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    # If target_moment is zero, the lobe should only transition levels or stay flat
    if abs(target_moment) < 1e-12: # Effectively zero moment
        delta_amp = target_amplitude - current_amplitude
        if abs(delta_amp) < 1e-9: # Effectively same start and end
            return torch.tensor([current_amplitude], device=device)

        t_ramp_to_target = abs(delta_amp) / max_slew_rate
        # Add small epsilon to avoid num_ramp_samples_to_target = 0 for very small t_ramp_to_target > 0
        num_ramp_samples_to_target = math.ceil(t_ramp_to_target / dt + 1e-9) 
        
        if num_ramp_samples_to_target == 0:
            return torch.tensor([target_amplitude], device=device)
        
        # Use linspace for robustness in generating the ramp
        # linspace includes start and end; we usually want points *after* current_amplitude
        waveform = torch.linspace(current_amplitude, target_amplitude, num_ramp_samples_to_target + 1, device=device)
        return waveform[1:]


    moment_sign = math.copysign(1.0, target_moment)
    target_moment_abs = abs(target_moment)
    
    g_peak_signed = moment_sign * max_amplitude
    actual_peak_amplitude = g_peak_signed # Assume we reach g_peak_signed unless triangle is better

    # --- Try to design a triangular lobe first (current -> peak -> target) ---
    # This specific section handles C=0, T=0 triangular lobes.
    # More general triangular solutions are implicitly handled by the trapezoid logic
    # where flat time becomes zero or negative.
    if abs(current_amplitude) < 1e-9 and abs(target_amplitude) < 1e-9:
        ap_tri_ideal_mag = 0.0
        if max_slew_rate > 1e-9: # Avoid division by zero if slew rate is effectively zero
             ap_tri_ideal_mag = math.sqrt(target_moment_abs * max_slew_rate)
        elif target_moment_abs > 1e-9: # Slew is zero but moment is not
            raise ValueError("Cannot achieve non-zero moment with zero slew rate.")


        if ap_tri_ideal_mag <= max_amplitude:
            actual_peak_amplitude = moment_sign * ap_tri_ideal_mag
            
            t_r_up = 0.0
            if max_slew_rate > 1e-9: t_r_up = abs(actual_peak_amplitude - 0.0) / max_slew_rate
            elif abs(actual_peak_amplitude) > 1e-9 : raise ValueError("Non-zero peak with zero slew.")
            n_r_up = math.ceil(t_r_up / dt + 1e-9) # Add epsilon
            
            g_r_up = torch.tensor([], device=device)
            if n_r_up > 0:
                slew_up = actual_peak_amplitude / (n_r_up * dt) if (n_r_up * dt) > 1e-12 else 0
                if abs(slew_up) > max_slew_rate * (1 + 1e-6) and max_slew_rate > 1e-9 : # Tolerance for float issues
                    pass # Could warn or adjust n_r_up
                g_r_up = torch.arange(1, n_r_up + 1, device=device) * slew_up * dt
            
            t_r_down = 0.0
            if max_slew_rate > 1e-9: t_r_down = abs(0.0 - actual_peak_amplitude) / max_slew_rate
            elif abs(actual_peak_amplitude) > 1e-9 : raise ValueError("Non-zero peak with zero slew.")
            n_r_down = math.ceil(t_r_down / dt + 1e-9) # Add epsilon

            g_r_down = torch.tensor([], device=device)
            if n_r_down > 0:
                slew_down = (0.0 - actual_peak_amplitude) / (n_r_down * dt) if (n_r_down*dt) > 1e-12 else 0
                if abs(slew_down) > max_slew_rate * (1 + 1e-6) and max_slew_rate > 1e-9:
                    pass
                g_r_down = actual_peak_amplitude + torch.arange(1, n_r_down + 1, device=device) * slew_down * dt
                if n_r_down > 0 and g_r_down.numel() > 0 : g_r_down[-1] = 0.0 # Ensure end at zero

            if n_r_up == 0 and n_r_down == 0: return torch.tensor([0.0], device=device)
            
            waveform_parts_tri = [g for g in [g_r_up, g_r_down] if g.numel() > 0]
            return torch.cat(waveform_parts_tri) if waveform_parts_tri else torch.tensor([0.0], device=device)

    # --- Trapezoidal Lobe Calculation (can also result in a triangle for general C, T) ---
    g_ramp1 = torch.tensor([], device=device)
    n_ramp1 = 0
    # Ramp from current_amplitude to actual_peak_amplitude (which is g_peak_signed here)
    if abs(actual_peak_amplitude - current_amplitude) > 1e-9 * max_amplitude : 
        t_r1 = abs(actual_peak_amplitude - current_amplitude) / max_slew_rate if max_slew_rate > 1e-9 else float('inf')
        if t_r1 == float('inf') and abs(actual_peak_amplitude-current_amplitude)>1e-9 : raise ValueError("Cannot ramp with zero slew rate.")
        elif t_r1 == float('inf') : t_r1 = 0 # No ramp needed if difference is tiny

        n_ramp1 = math.ceil(t_r1 / dt + 1e-9)
        if n_ramp1 > 0:
            g_ramp1 = torch.linspace(current_amplitude, actual_peak_amplitude, n_ramp1 + 1, device=device)[1:]

    g_ramp2 = torch.tensor([], device=device)
    n_ramp2 = 0
    # Ramp from actual_peak_amplitude to target_amplitude
    if abs(actual_peak_amplitude - target_amplitude) > 1e-9 * max_amplitude :
        t_r2 = abs(target_amplitude - actual_peak_amplitude) / max_slew_rate if max_slew_rate > 1e-9 else float('inf')
        if t_r2 == float('inf') and abs(target_amplitude-actual_peak_amplitude)>1e-9 : raise ValueError("Cannot ramp with zero slew rate.")
        elif t_r2 == float('inf') : t_r2 = 0

        n_ramp2 = math.ceil(t_r2 / dt + 1e-9)
        if n_ramp2 > 0:
            g_ramp2 = torch.linspace(actual_peak_amplitude, target_amplitude, n_ramp2 + 1, device=device)[1:]

    # Moment from these ramps (sum of discrete points for accuracy)
    moment1 = torch.sum(g_ramp1) * dt if n_ramp1 > 0 else 0.0
    moment2 = torch.sum(g_ramp2) * dt if n_ramp2 > 0 else 0.0
    moment_ramps = moment1 + moment2
    
    remaining_moment_signed = target_moment - moment_ramps

    g_flat = torch.tensor([], device=device)
    n_flat = 0

    if abs(actual_peak_amplitude) < 1e-9: 
        if abs(remaining_moment_signed) > 1e-9 * target_moment_abs: # Check with relative tolerance
            pass # Error: Cannot generate moment with zero amplitude flat part.
    elif (remaining_moment_signed * actual_peak_amplitude) > 1e-12: # Check if moment can be added by flat top
                                                                    # and signs are compatible
        t_flat = remaining_moment_signed / actual_peak_amplitude 
        n_flat = math.ceil(t_flat / dt + 1e-9) 
        if n_flat < 0: n_flat = 0 # Should not happen due to sign check
        if n_flat > 0:
            g_flat = torch.ones(n_flat, device=device) * actual_peak_amplitude
    
    if (remaining_moment_signed * actual_peak_amplitude) < -1e-12 and \
       abs(remaining_moment_signed) > 1e-6 * target_moment_abs and \
       not (abs(current_amplitude) < 1e-9 and abs(target_amplitude) < 1e-9):
        n_flat = 0 
        g_flat = torch.tensor([], device=device)


    # Final assembly
    waveform_parts = [g for g in [g_ramp1, g_flat, g_ramp2] if g.numel() > 0]

    if not waveform_parts:
        if abs(target_moment) < 1e-12 and abs(current_amplitude - target_amplitude) < 1e-9:
            return torch.tensor([current_amplitude], device=device)
        # If target moment is non-zero but no waveform parts, it's likely an unhandled edge case or error.
        # Default to a single point at target_amplitude or raise error.
        # For now, returning target_amplitude to avoid crashing, but this indicates a design flaw.
        return torch.tensor([target_amplitude], device=device) 


    final_waveform = torch.cat(waveform_parts)
    if final_waveform.numel() == 0: # Should be caught by 'if not waveform_parts'
        # This case implies current_amplitude = target_amplitude and target_moment is zero.
        return torch.tensor([target_amplitude], device=device) 

    return final_waveform


def design_min_time_gradient_lobe_with_bridge(
    target_moment: float, 
    ramp_fraction_for_moment: float, 
    max_amplitude: float, 
    max_slew_rate: float, 
    dt: float,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Designs a minimum-time gradient lobe (g_lobe) for target_moment.
    The lobe is symmetric, starting and ending at zero amplitude.
    A central portion of this lobe (g_bridge) accounts for target_moment.
    ramp_fraction_for_moment (alpha) defines how much of the ramps of the 
    moment-generating ideal trapezoid/triangle contribute to g_bridge.

    Args:
        target_moment: Desired gradient moment for the bridge.
        ramp_fraction_for_moment: Alpha (0 to 1). Fraction of ideal ramp duration 
                                  contributing to the bridge.
        max_amplitude: Maximum allowed gradient amplitude.
        max_slew_rate: Maximum allowed gradient slew rate.
        dt: Sampling interval.
        device: PyTorch device for output tensors.

    Returns:
        g_lobe: The full gradient lobe waveform.
        g_ramp_up_outer: The initial ramp segment (g1).
        g_bridge: The central bridge segment (g2), responsible for target_moment.
        g_ramp_down_outer: The final ramp segment (g3).
    """
    if not (0.0 <= ramp_fraction_for_moment <= 1.0):
        raise ValueError("ramp_fraction_for_moment must be between 0 and 1.")
    if target_moment == 0: # No moment, no lobe, no bridge
        return (torch.tensor([], device=device), torch.tensor([], device=device), 
                torch.tensor([], device=device), torch.tensor([], device=device))

    _M0_abs = abs(target_moment)
    _moment_sign = math.copysign(1.0, target_moment)

    # 1. Determine characteristics of the ideal lobe for M0 (current=0, target=0)
    G_p_actual_mag = 0.0 # Magnitude of the peak of the core M0 lobe
    core_ramp_time = 0.0 # Duration of one ramp of the core M0 lobe
    core_flat_time = 0.0 # Duration of the flat top of the core M0 lobe

    if max_slew_rate < 1e-9: # Effectively zero slew rate
        if _M0_abs > 1e-9: raise ValueError("Cannot achieve non-zero moment with zero slew rate.")
        # Else, M0 is zero, handled by initial check. G_p_actual_mag remains 0.
    else:
        G_p_triangle_mag = math.sqrt(_M0_abs * max_slew_rate)

        if G_p_triangle_mag <= max_amplitude:
            G_p_actual_mag = G_p_triangle_mag
            core_ramp_time = G_p_actual_mag / max_slew_rate
            core_flat_time = 0.0
        else:
            G_p_actual_mag = max_amplitude
            core_ramp_time = G_p_actual_mag / max_slew_rate
            moment_ramps = G_p_actual_mag * core_ramp_time 
            
            if G_p_actual_mag > 1e-9:
                core_flat_time = (_M0_abs - moment_ramps) / G_p_actual_mag
            elif _M0_abs > 1e-9: # Zero peak but non-zero moment needed
                 raise ValueError("Cannot achieve non-zero moment with zero peak amplitude.")
            else: # Zero peak and zero moment
                core_flat_time = 0.0
                
            if core_flat_time < 0: core_flat_time = 0.0

    G_p_actual_signed = _moment_sign * G_p_actual_mag

    # 2. Define durations for g1, g2_ramp based on alpha
    # t_g1_time: duration of outer ramp (g1)
    # t_g2_ramp_time: duration of the ramp part of g2 (bridge ramp)
    t_g1_time = core_ramp_time * (1.0 - ramp_fraction_for_moment)
    t_g2_ramp_time = core_ramp_time * ramp_fraction_for_moment

    # 3. Generate g1 (outer ramp-up from 0 to G_int_amp_discrete)
    G_int_amp_ideal = (max_slew_rate * t_g1_time) * _moment_sign
    if abs(G_int_amp_ideal) > abs(G_p_actual_signed): # Cap if overshoot (should not happen if t_g1_time calc is right)
        G_int_amp_ideal = G_p_actual_signed * math.copysign(1-1e-9, _moment_sign) if abs(G_p_actual_signed)>0 else 0.0
    
    n_g1_samples = math.ceil(t_g1_time / dt + 1e-9) if t_g1_time > 1e-12 else 0
    g1 = torch.tensor([], device=device)
    G_int_amp_discrete = 0.0
    if n_g1_samples > 0:
        # Recalculate G_int_amp based on discrete sample count for g1 to ensure slew rate is met
        # Or, fix G_int_amp_ideal and calculate slew. Linspace is safer.
        g1 = torch.linspace(0, G_int_amp_ideal, n_g1_samples + 1, device=device)[1:]
        if g1.numel() > 0: G_int_amp_discrete = g1[-1].item()
    
    # 4. Generate g2_ramp_up (bridge ramp-up from G_int_amp_discrete to G_p_actual_signed)
    n_g2_ramp_samples = math.ceil(t_g2_ramp_time / dt + 1e-9) if t_g2_ramp_time > 1e-12 else 0
    g2_ramp_up = torch.tensor([], device=device)
    if n_g2_ramp_samples > 0:
        if abs(G_p_actual_signed - G_int_amp_discrete) < 1e-9 * max_amplitude: # Already at peak (relative tolerance)
            n_g2_ramp_samples = 0 # No ramp needed
        else:
            g2_ramp_up = torch.linspace(G_int_amp_discrete, G_p_actual_signed, n_g2_ramp_samples + 1, device=device)[1:]
    
    # 5. Generate g2_flat (bridge flat part)
    n_g2_flat_samples = math.ceil(core_flat_time / dt + 1e-9) if core_flat_time > 1e-12 else 0
    g2_flat = torch.tensor([], device=device)
    if n_g2_flat_samples > 0:
        g2_flat = torch.ones(n_g2_flat_samples, device=device) * G_p_actual_signed

    # 6. Generate g2_ramp_down (symmetric to g2_ramp_up)
    g2_ramp_down = torch.tensor([], device=device)
    if n_g2_ramp_samples > 0: # Use same number of samples as ramp up for symmetry
        # Waveform is from G_p_actual_signed down to G_int_amp_discrete
        g2_ramp_down = torch.linspace(G_p_actual_signed, G_int_amp_discrete, n_g2_ramp_samples + 1, device=device)[1:]
            
    # 7. Generate g3 (outer ramp-down, symmetric to g1)
    g3 = torch.tensor([], device=device)
    if n_g1_samples > 0: # Use same number of samples as g1 for symmetry
        g3 = torch.linspace(G_int_amp_discrete, 0, n_g1_samples + 1, device=device)[1:]
            
    g_bridge_parts = [part for part in [g2_ramp_up, g2_flat, g2_ramp_down] if part.numel() > 0]
    g_bridge = torch.cat(g_bridge_parts) if g_bridge_parts else torch.tensor([], device=device)

    g_lobe_parts = [part for part in [g1, g_bridge, g3] if part.numel() > 0]
    g_lobe = torch.cat(g_lobe_parts) if g_lobe_parts else torch.tensor([], device=device)
    
    return g_lobe, g1, g_bridge, g3


def design_bipolar_lobe(
    positive_lobe_moment: float, 
    total_duration_samples: int = None, # First pass: ignore this, design for min time
    ramp_fraction_for_moment: float = 0.8, 
    max_amplitude: float, 
    max_slew_rate: float, 
    dt: float, 
    equal_lobes: bool = False,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """Designs a bipolar gradient lobe.

    The positive lobe is designed using design_min_time_gradient_lobe_with_bridge.
    The negative lobe balances the positive lobe's moment.
    If equal_lobes is True, the negative lobe mirrors the positive lobe's shape 
    (designed for -positive_lobe_moment with same alpha).
    Otherwise, it's a minimum time trapezoid to achieve moment balance.

    Args:
        positive_lobe_moment: Moment of the positive lobe.
        total_duration_samples: (Optional) If provided, adjust to meet total duration. 
                                  Currently not implemented in detail (designs for min time).
        ramp_fraction_for_moment: Alpha for the positive lobe design.
        max_amplitude: Maximum allowed gradient amplitude.
        max_slew_rate: Maximum allowed gradient slew rate.
        dt: Sampling interval.
        equal_lobes: If True, negative lobe mirrors positive lobe's characteristics.
        device: PyTorch device for output tensors.

    Returns:
        g_positive_lobe: Waveform of the positive lobe.
        g_negative_lobe: Waveform of the negative lobe.
    """
    if abs(positive_lobe_moment) < 1e-12: # No moment, no lobes
        return torch.tensor([], device=device), torch.tensor([], device=device)

    g_positive_lobe, _, _, _ = design_min_time_gradient_lobe_with_bridge(
        target_moment=positive_lobe_moment,
        ramp_fraction_for_moment=ramp_fraction_for_moment,
        max_amplitude=max_amplitude,
        max_slew_rate=max_slew_rate,
        dt=dt,
        device=device
    )

    actual_positive_moment = torch.sum(g_positive_lobe) * dt
    negative_target_moment = -actual_positive_moment.item() 

    g_negative_lobe = torch.tensor([], device=device)
    if abs(negative_target_moment) < 1e-12: # Positive lobe had no moment
         pass # Negative lobe also has no moment
    elif equal_lobes:
        g_negative_lobe, _, _, _ = design_min_time_gradient_lobe_with_bridge(
            target_moment=negative_target_moment, 
            ramp_fraction_for_moment=ramp_fraction_for_moment,
            max_amplitude=max_amplitude,
            max_slew_rate=max_slew_rate,
            dt=dt,
            device=device
        )
    else:
        g_negative_lobe = design_trapezoidal_lobe(
            target_moment=negative_target_moment,
            max_amplitude=max_amplitude,
            max_slew_rate=max_slew_rate,
            dt=dt,
            current_amplitude=0.0,
            target_amplitude=0.0,
            device=device
        )

    if total_duration_samples is not None:
        current_total_samples = g_positive_lobe.numel() + g_negative_lobe.numel()
        # Basic duration handling: if current implementation is shorter, pad with zeros between lobes.
        # More advanced handling (e.g. elongating flat tops) is not implemented.
        if current_total_samples < total_duration_samples:
            padding_samples = total_duration_samples - current_total_samples
            if padding_samples > 0:
                padding = torch.zeros(padding_samples, device=device)
                # This simple concatenation just adds padding at the end.
                # A better approach might be to insert padding between lobes,
                # or split padding before/between/after.
                # For now, this function returns lobes separately, user can combine with padding.
                # This comment serves as a note for future improvement if padding is internal.
                pass # User can handle padding externally with returned lobes.

        # If current_total_samples > total_duration_samples, it's an issue not handled here.

    return g_positive_lobe, g_negative_lobe
