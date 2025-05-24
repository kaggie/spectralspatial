import torch
import math
from typing import List, Tuple

class SLRTransform:
    """
    Implements transformations related to Shinnar-Le Roux (SLR) RF pulse design.
    """

    def magnetization_to_b_poly_specs(
        self, 
        desired_mag_ripples: List[float], 
        desired_mag_amplitudes: List[float], 
        nominal_flip_angle_rad: float, 
        pulse_type: str, 
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Converts desired magnetization profile specifications (ripples and amplitudes)
        to B-polynomial specifications for RF pulse design. This method follows
        the logic similar to rf_ripple.m from John Pauly's SLR package.

        Args:
            desired_mag_ripples: List of ripples in the magnetization profile. 
                                 (e.g., [d1, d2] for passband and stopband).
            desired_mag_amplitudes: List of relative amplitudes (0 to 1) in the 
                                    magnetization profile corresponding to each band.
            nominal_flip_angle_rad: The target overall flip angle in radians.
            pulse_type: String, one of 'ex' (excitation), 'se' (spin-echo/refocusing),
                        'inv' (inversion), 'sat' (saturation).
            device: PyTorch device ('cpu' or 'cuda').

        Returns:
            b_poly_ripples: PyTorch Tensor of ripples for the B-polynomial design.
            b_poly_amplitudes: PyTorch Tensor of amplitudes for the B-polynomial design.
            adjusted_flip_angle_rad: Float, the potentially adjusted overall flip angle 
                                     due to amplitude constraints.
        """
        if len(desired_mag_ripples) != len(desired_mag_amplitudes):
            raise ValueError("desired_mag_ripples and desired_mag_amplitudes must have the same length.")
        if not desired_mag_ripples:
            raise ValueError("Input ripple/amplitude lists cannot be empty.")

        num_bands = len(desired_mag_amplitudes)
        b_poly_ripples_list = [0.0] * num_bands
        b_poly_amplitudes_list = [0.0] * num_bands
        
        # Convert to torch tensors for calculations
        mag_amp = torch.tensor(desired_mag_amplitudes, dtype=torch.float32, device=device)
        mag_rip = torch.tensor(desired_mag_ripples, dtype=torch.float32, device=device)
        
        # Nominal flip angle components
        alpha_2 = nominal_flip_angle_rad / 2.0
        sin_alpha_2 = torch.sin(torch.tensor(alpha_2, dtype=torch.float32, device=device))
        cos_alpha_2 = torch.cos(torch.tensor(alpha_2, dtype=torch.float32, device=device))

        if pulse_type == 'ex': # Excitation pulse
            # Mxy ~ sin(theta/2) * B(z)
            # beta_amp = Mxy_amp / sin(alpha/2)
            # beta_rip = Mxy_rip / sin(alpha/2)
            # Note: Mxy_amp is mag_amp, Mxy_rip is mag_rip
            if abs(sin_alpha_2.item()) < 1e-9: # Avoid division by zero if flip angle is ~0
                 b_poly_amplitudes_list = [0.0] * num_bands
                 b_poly_ripples_list = [r for r in desired_mag_ripples] # Ripples don't scale if amp is 0
            else:
                b_poly_amplitudes_list = (mag_amp / sin_alpha_2).tolist()
                b_poly_ripples_list = (mag_rip / sin_alpha_2).tolist()
            
            # For excitation, B_poly amplitudes can exceed 1 if Mxy/sin(alpha/2) > 1.
            # This is handled by the final normalization step.

        elif pulse_type in ['se', 'inv', 'sat']: # Refocusing, Inversion, Saturation
            # These pulses primarily affect Mz.
            # Mz = 1 - 2 * sin^2(theta/2) * |B(z)|^2
            # Let mzd = desired Mz profile points.
            # For a desired Mz amplitude 'm_amp' with ripple 'm_rip':
            # Mz_pass_center = m_amp
            # Mz_pass_edge_1 = m_amp + m_rip 
            # Mz_pass_edge_2 = m_amp - m_rip
            #
            # |B(z)|^2 = (1 - Mz) / (2 * sin^2(alpha/2))
            # So, |B(z)| = sqrt( (1-Mz) / (2*sin^2(alpha/2)) )
            
            if abs(sin_alpha_2.item()) < 1e-9 : # alpha is effectively 0
                # If flip is 0, Mz should be 1. (1-Mz) is 0. So |B| is 0.
                b_poly_amplitudes_list = [0.0] * num_bands
                # Ripples are tricky here. If B is 0, its ripple is also 0 or undefined.
                # Let's assume ripple on B is also 0 if B is 0.
                b_poly_ripples_list = [0.0] * num_bands
            else:
                for i in range(num_bands):
                    m_amp_band = mag_amp[i] # This is the desired Mz amplitude for the band
                    m_rip_band = mag_rip[i]
                    
                    # Determine target Mz values at band center, positive ripple, negative ripple
                    # Note: 'desired_mag_amplitudes' for these pulses are Mz values.
                    mz_mid = m_amp_band
                    mz_pos_rip_edge = m_amp_band + m_rip_band 
                    mz_neg_rip_edge = m_amp_band - m_rip_band

                    # Clip Mz values to be within physical limits [-1, 1]
                    mz_mid = torch.clamp(mz_mid, -1.0, 1.0)
                    mz_pos_rip_edge = torch.clamp(mz_pos_rip_edge, -1.0, 1.0)
                    mz_neg_rip_edge = torch.clamp(mz_neg_rip_edge, -1.0, 1.0)
                    
                    # Corresponding |B(z)| values squared
                    # B2_mid = (1 - mz_mid) / (2 * sin_alpha_2**2)
                    # B2_pos = (1 - mz_pos_rip_edge) / (2 * sin_alpha_2**2) # Corresponds to Mz+delta
                    # B2_neg = (1 - mz_neg_rip_edge) / (2 * sin_alpha_2**2) # Corresponds to Mz-delta
                    
                    # Avoid issues if 2*sin_alpha_2**2 is zero
                    denom = 2 * sin_alpha_2**2
                    if denom < 1e-12: # practically zero
                        # If Mz is not 1, then B would be infinite, which is unphysical.
                        # This case implies alpha is ~0, so Mz should be ~1.
                        # If mz_mid is not ~1, it's problematic.
                        # If mz_mid is ~1, then (1-mz_mid) is ~0, so B is ~0.
                        B2_mid = torch.tensor(0.0, device=device) if torch.abs(1.0 - mz_mid) < 1e-6 else torch.tensor(float('inf'), device=device)
                        B2_pos = torch.tensor(0.0, device=device) if torch.abs(1.0 - mz_pos_rip_edge) < 1e-6 else torch.tensor(float('inf'), device=device)
                        B2_neg = torch.tensor(0.0, device=device) if torch.abs(1.0 - mz_neg_rip_edge) < 1e-6 else torch.tensor(float('inf'), device=device)
                    else:
                        B2_mid = (1 - mz_mid) / denom
                        B2_pos = (1 - mz_pos_rip_edge) / denom
                        B2_neg = (1 - mz_neg_rip_edge) / denom

                    # Ensure B^2 terms are non-negative before sqrt
                    B2_mid = torch.clamp_min(B2_mid, 0.0)
                    B2_pos = torch.clamp_min(B2_pos, 0.0)
                    B2_neg = torch.clamp_min(B2_neg, 0.0)

                    B_mid_val = torch.sqrt(B2_mid)
                    B_pos_val = torch.sqrt(B2_pos) # |B| when Mz is at Mz_mid + Mz_ripple
                    B_neg_val = torch.sqrt(B2_neg) # |B| when Mz is at Mz_mid - Mz_ripple
                    
                    # B_poly amplitude is B_mid_val
                    # B_poly ripple is difference between B_mid_val and B_pos_val/B_neg_val
                    # Ripple on B is |B_pos_val - B_mid_val| or |B_neg_val - B_mid_val|
                    # Typically, for Mz stopbands (Mz ~ 1, B ~ 0), ripple on B is max(B_pos, B_neg)
                    # For Mz passbands (Mz ~ -1, B ~ 1/sin(a/2)), ripple on B is |B_mid - min(B_pos, B_neg)|
                    
                    # Pauly's rf_ripple.m logic:
                    # bAmp = B_mid_val
                    # bRip = max(abs(B_mid_val - B_pos_val), abs(B_mid_val - B_neg_val))
                    # This general form should work.
                    b_amp_band = B_mid_val
                    b_rip_band = torch.max(torch.abs(B_mid_val - B_pos_val), torch.abs(B_mid_val - B_neg_val))
                    
                    # Constraint: b_amp_band should not exceed 1 for physical B polynomial.
                    # (This is implicitly handled by the final normalization if nominal_flip_angle needs adjustment)
                    # However, rf_ripple.m sometimes has an internal check:
                    # if sin(alpha/2)*b_amp_band > 1 (approx), it means Mz would be < -1 or > 1 from the formula
                    # (1 - 2 * sin^2(a/2) * b_amp^2).
                    # This check is essentially `b_amp_band > 1/sin(a/2)`.
                    # If b_amp_band is calculated from Mz values already clamped to [-1,1],
                    # then b_amp_band itself should be physical.
                    # The constraint is more about `b_amp_band <= 1` if it's a *normalized* B poly.
                    # Here, b_amp_band is scaled by 1/sin(alpha/2) relative to a normalized B.
                    # The final normalization step scales all b_poly_amplitudes so that max is 1.
                    
                    b_poly_amplitudes_list[i] = b_amp_band.item()
                    b_poly_ripples_list[i] = b_rip_band.item()
        else:
            raise ValueError(f"Unknown pulse_type: {pulse_type}. Must be 'ex', 'se', 'inv', or 'sat'.")

        # --- Final Normalization and Flip Angle Adjustment ---
        # This step ensures that the maximum B-polynomial amplitude is 1.
        # If it's not, the flip angle is adjusted.
        
        b_poly_amp_tensor = torch.tensor(b_poly_amplitudes_list, dtype=torch.float32, device=device)
        b_poly_rip_tensor = torch.tensor(b_poly_ripples_list, dtype=torch.float32, device=device)

        adjusted_flip_angle_rad = nominal_flip_angle_rad
        
        if b_poly_amp_tensor.numel() > 0 : # Proceed if there are any bands
            max_b_amp = torch.max(b_poly_amp_tensor)
            if max_b_amp.item() < 1e-9 : # All B amplitudes are essentially zero
                # This can happen if flip angle is zero or Mxy/Mz specs lead to zero B.
                # No adjustment needed, ripples are relative to zero amplitude.
                pass
            elif abs(max_b_amp.item() - 1.0) > 1e-6 : # If max_b_amp is not already 1.0
                # Scale B amplitudes so the max is 1.0
                b_poly_amplitudes_final = b_poly_amp_tensor / max_b_amp
                # Scale B ripples by the same factor
                b_poly_ripples_final = b_poly_rip_tensor / max_b_amp
                
                # Adjust the overall flip angle:
                # Original transform: Mxy ~ sin(alpha/2)*B or |B|^2 ~ (1-Mz)/(2sin^2(alpha/2))
                # If B is scaled by 1/max_b_amp, then effectively sin(alpha/2) is scaled by max_b_amp.
                # new_sin_alpha_2 = sin_alpha_2 * max_b_amp
                new_sin_alpha_2_val = sin_alpha_2 * max_b_amp
                
                # Clamp new_sin_alpha_2_val to [-1, 1] before asin
                if new_sin_alpha_2_val.item() > 1.0: new_sin_alpha_2_val = torch.tensor(1.0, device=device)
                if new_sin_alpha_2_val.item() < -1.0: new_sin_alpha_2_val = torch.tensor(-1.0, device=device) # Should not be negative normally

                adjusted_alpha_2 = torch.asin(new_sin_alpha_2_val)
                adjusted_flip_angle_rad = (2 * adjusted_alpha_2).item()
                
                b_poly_amp_tensor = b_poly_amplitudes_final
                b_poly_rip_tensor = b_poly_ripples_final
            # Else, max_b_amp is already 1.0, no scaling needed.
        
        return b_poly_rip_tensor, b_poly_amp_tensor, adjusted_flip_angle_rad

    def b_poly_to_rf(self, b_poly_coeffs: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """
        Converts a B-polynomial (filter coefficients) to an RF pulse segment 
        (alpha polynomial coefficients). Equivalent to b2rf.c from Pauly's tools.
        NOTE: This is a placeholder and the core SLR math is not yet implemented.
        """
        # TODO: Implement the actual math for b2rf (e.g., inverse FFT, phase adjustments)
        # For now, as a placeholder, if this is for small tip angle approximation,
        # the b_poly_coeffs might be directly proportional to the RF pulse.
        # However, true SLR involves complex polynomials a and b.
        # rf(t) ~ (a(t)b*(-t) - a*(-t)b(t)) / (|a(t)|^2 + |b(t)|^2) (not quite)
        # More simply, for small tip, rf ~ B. For SLR, rf requires finding alpha from beta.
        # Beta(z) = FIR filter. Alpha(z)Alpha*(1/z*) + Beta(z)Beta*(1/z*) = 1
        # This involves spectral factorization of 1 - Beta(z)Beta*(1/z*) to get Alpha(z).
        # Then RF(t) is related to IFFT(Alpha(z)) and IFFT(Beta(z)).
        raise NotImplementedError("SLR B-polynomial to RF conversion (b2rf equivalent) is not yet implemented.")