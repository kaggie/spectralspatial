import torch
import numpy as np
import math
from typing import List, Tuple

def calculate_aliased_specs(
    f_orig_hz: List[float], 
    a_orig: List[float], 
    d_orig: List[float], 
    fs_hz: float, 
    f_offset_hz: float = None, 
    symmetric_response: bool = False, 
    threshold_edge_factor: float = 0.03, # Currently unused in simplified f_offset calc
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Calculates aliased filter specifications based on original specifications,
    sampling frequency, and offset. Follows logic similar to ss_alias.m.

    Args:
        f_orig_hz: List of original frequency band edges in Hz 
                   (e.g., [f1_low, f1_high, f2_low, f2_high, ...]). 
                   Must be sorted and non-overlapping.
        a_orig: List of original amplitudes for each band (0 to 1). 
                Length must be len(f_orig_hz) / 2.
        d_orig: List of original ripples for each band. 
                Length must be len(f_orig_hz) / 2.
        fs_hz: Sampling frequency in Hz.
        f_offset_hz: Optional center frequency offset in Hz. If None, it's determined.
        symmetric_response: If True, mirror input specs around f_offset_hz 
                            before aliasing.
        threshold_edge_factor: Used if f_offset_hz is auto-calculated with advanced
                               logic (not in this simplified version), as a factor of 
                               fs_hz/2 for preferred minimum distance from Nyquist.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        f_aliased_norm: PyTorch Tensor of aliased band edges, normalized to [-1, 1].
        a_aliased: PyTorch Tensor of corresponding amplitudes.
        d_aliased: PyTorch Tensor of corresponding ripples.
        f_offset_hz_used: The float value of the frequency offset used.
    """
    # --- Input Validation ---
    if not f_orig_hz:
        # Return empty if no original bands provided
        return (torch.empty(0, device=device), torch.empty(0, device=device), 
                torch.empty(0, device=device), f_offset_hz if f_offset_hz is not None else 0.0)

    if len(f_orig_hz) % 2 != 0:
        raise ValueError("f_orig_hz must have an even number of elements (pairs of band edges).")
    if len(a_orig) != len(f_orig_hz) / 2 or len(d_orig) != len(f_orig_hz) / 2:
        raise ValueError("a_orig and d_orig must have half the length of f_orig_hz.")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")

    current_f_max = -float('inf')
    for i in range(0, len(f_orig_hz), 2):
        f_low = f_orig_hz[i]
        f_high = f_orig_hz[i+1]
        if f_low >= f_high:
            raise ValueError(f"Band edges must be monotonic: f_low < f_high. Found [{f_low}, {f_high}]")
        if f_low < current_f_max:
            raise ValueError("f_orig_hz bands must be sorted and non-overlapping.")
        current_f_max = f_high
        if (f_high - f_low) > fs_hz: # Bandwidth check
            # print(f"Warning: Bandwidth of [{f_low}, {f_high}] exceeds fs_hz. Returning empty specs.")
            return (torch.empty(0, device=device), torch.empty(0, device=device), 
                    torch.empty(0, device=device), f_offset_hz if f_offset_hz is not None else 0.0)

    f_bands_orig = np.array(f_orig_hz).reshape(-1, 2)
    a_bands_orig = np.array(a_orig)
    d_bands_orig = np.array(d_orig)

    # --- Determine f_offset_hz_used ---
    f_offset_hz_used = 0.0
    if f_offset_hz is not None:
        f_offset_hz_used = f_offset_hz
    elif symmetric_response:
        if not f_bands_orig.size > 0: # Should have been caught by "if not f_orig_hz"
             f_offset_hz_used = 0.0
        else:
             f_offset_hz_used = (f_bands_orig[0, 0] + f_bands_orig[0, 1]) / 2.0
    else: # f_offset_hz is None and not symmetric_response
        min_f_orig = f_bands_orig[0,0] if f_bands_orig.size > 0 else 0
        max_f_orig = f_bands_orig[-1,1] if f_bands_orig.size > 0 else 0
        # Simplified approach for now:
        f_offset_hz_used = (max_f_orig + min_f_orig) / 2.0
        # Note: ss_alias.m has a more complex search here to optimize band placement.
        # That logic considers band counts and edge distances, which is a future enhancement.

    # --- Symmetric Response ---
    if symmetric_response:
        # Mirror bands around f_offset_hz_used
        # f_mirrored = f_offset - (f_orig - f_offset) = 2*f_offset - f_orig
        f_sym_lower = 2 * f_offset_hz_used - f_bands_orig[:, 1]
        f_sym_higher = 2 * f_offset_hz_used - f_bands_orig[:, 0]
        
        # Combine original and mirrored (mirrored are flipped, so re-sort)
        f_bands_combined = np.concatenate((f_bands_orig, np.stack((f_sym_lower, f_sym_higher), axis=-1)), axis=0)
        a_bands_combined = np.concatenate((a_bands_orig, a_bands_orig), axis=0)
        d_bands_combined = np.concatenate((d_bands_orig, d_bands_orig), axis=0)
        
        # Sort by lower band edge
        sort_idx = np.argsort(f_bands_combined[:, 0])
        f_bands_current = f_bands_combined[sort_idx]
        a_bands_current = a_bands_combined[sort_idx]
        d_bands_current = d_bands_combined[sort_idx]
    else:
        f_bands_current = f_bands_orig
        a_bands_current = a_bands_orig
        d_bands_current = d_bands_orig

    # --- Normalization & Aliasing ---
    nyquist_freq = fs_hz / 2.0
    if nyquist_freq == 0: # Avoid division by zero
        return (torch.empty(0, device=device), torch.empty(0, device=device), 
                torch.empty(0, device=device), f_offset_hz_used)

    fn_bands = (f_bands_current - f_offset_hz_used) / nyquist_freq
    
    # Apply aliasing: fa_point = (fn + 1) % 2 - 1
    # Using fmod for behavior similar to MATLAB's mod(x,y) = x - y*floor(x/y)
    # Python's % is different for negative numbers.
    # (fn + 1) ensures positive input to fmod, then shift back.
    fa_bands_norm = np.fmod(fn_bands + 1.0, 2.0) - 1.0
    
    # Clip to handle potential floating point inaccuracies slightly outside [-1, 1]
    fa_bands_norm = np.clip(fa_bands_norm, -1.0, 1.0)

    # --- Band Splitting & Processing ---
    aliased_specs = [] # List of [f_low, f_high, amp, ripple]

    for i in range(fa_bands_norm.shape[0]):
        fa_low, fa_high = fa_bands_norm[i, 0], fa_bands_norm[i, 1]
        amp, ripple = a_bands_current[i], d_bands_current[i]

        if fa_low >= fa_high: # Band is zero-width or inverted after clipping
            if abs(fa_low - fa_high) < 1e-9: # Effectively zero width
                continue # Skip zero-width bands
            else: # Wrapped around: fa_high < fa_low
                # Split into two bands: [-1, fa_high_original] and [fa_low_original, 1]
                aliased_specs.append([-1.0, fa_high, amp, ripple])
                aliased_specs.append([fa_low, 1.0, amp, ripple])
        else:
            aliased_specs.append([fa_low, fa_high, amp, ripple])
    
    if not aliased_specs:
        return (torch.empty(0, device=device), torch.empty(0, device=device), 
                torch.empty(0, device=device), f_offset_hz_used)

    # --- Overlap Resolution ---
    # Sort by low edge first
    aliased_specs.sort(key=lambda x: x[0])
    
    final_bands_f = []
    final_bands_a = []
    final_bands_d = []
    
    if not aliased_specs: # Should be caught before but defensive
        return (torch.empty(0, device=device), torch.empty(0, device=device), 
                torch.empty(0, device=device), f_offset_hz_used)

    # More robust overlap: iterate and merge
    merged_specs = []
    current_spec = list(aliased_specs[0]) # Make a mutable copy

    for i in range(1, len(aliased_specs)):
        next_spec = aliased_specs[i]
        # Check for overlap: current_spec[1] > next_spec[0] (current_high > next_low)
        if current_spec[1] > next_spec[0] - 1e-9: # Overlap or touching (with tolerance)
            if abs(current_spec[2] - next_spec[2]) < 1e-9: # Amplitudes are same
                current_spec[1] = max(current_spec[1], next_spec[1]) # Merge: new high
                current_spec[3] = min(current_spec[3], next_spec[3]) # Merge: min ripple
            else: # Amplitudes different - incompatible overlap
                # print("Warning: Incompatible overlap detected. Returning empty specs.")
                return (torch.empty(0, device=device), torch.empty(0, device=device), 
                        torch.empty(0, device=device), f_offset_hz_used)
        else: # No overlap, finalize current_spec and start new one
            merged_specs.append(current_spec)
            current_spec = list(next_spec)
    merged_specs.append(current_spec) # Add the last processed spec

    aliased_specs = merged_specs

    # --- Symmetric Response Post-processing ---
    if symmetric_response:
        processed_sym_specs = []
        for f_low, f_high, amp, ripple in aliased_specs:
            if f_high > 1e-9: # Keep only bands where f_high_norm > 0 (small tolerance for 0)
                processed_sym_specs.append([max(0.0, f_low), f_high, amp, ripple])
        aliased_specs = processed_sym_specs
        # Re-sort again as max(0, f_low) might change order
        aliased_specs.sort(key=lambda x: x[0])


    if not aliased_specs:
        return (torch.empty(0, device=device), torch.empty(0, device=device), 
                torch.empty(0, device=device), f_offset_hz_used)

    # Separate into f, a, d lists
    for spec in aliased_specs:
        final_bands_f.extend([spec[0], spec[1]])
        final_bands_a.append(spec[2])
        final_bands_d.append(spec[3])
        
    f_aliased_tensor = torch.tensor(final_bands_f, dtype=torch.float32, device=device)
    a_aliased_tensor = torch.tensor(final_bands_a, dtype=torch.float32, device=device)
    d_aliased_tensor = torch.tensor(final_bands_d, dtype=torch.float32, device=device)

    return f_aliased_tensor, a_aliased_tensor, d_aliased_tensor, f_offset_hz_used