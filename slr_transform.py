import torch
import math
from typing import List, Tuple
import torch.nn.functional as F # For conv1d
from mri_pulse_library.core.dsp_utils import spectral_factorization_cepstral
from fir_designer import FIRFilterDesigner # Assuming fir_designer.py is at root

class SLRTransform:
    """
    Implements transformations related to Shinnar-Le Roux (SLR) RF pulse design.
    """
    def __init__(self):
        """Initializes the SLRTransform class."""
        self._a_coeffs_last = None
        self._b_coeffs_last = None

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
        # raise NotImplementedError("SLR B-polynomial to RF conversion (b2rf equivalent) is not yet implemented.")
    # Replacing with the full implementation:
    def b_poly_to_rf(self,
                     b_poly_coeffs: torch.Tensor,
                     pulse_type: str = 'ex',
                     nfft_factor: int = 4, # Factor to determine NFFT based on input length
                     device: str = 'cpu'
                     ) -> torch.Tensor:
        """
        Converts B-polynomial coefficients (Beta(z)) to a complex RF pulse waveform.

        This implementation uses spectral factorization to find A(z) such that
        A(z)A*(1/z*) + B(z)B*(1/z*) = 1, and then calculates the RF pulse
        based on A(z) and B(z).

        Args:
            b_poly_coeffs (torch.Tensor): 1D tensor of real coefficients for the
                                          B-polynomial B(z). Length L_b.
            pulse_type (str, optional): Type of pulse. Currently supports 'ex' (excitation).
                                        Future support for 'se', 'inv', 'sat'. Defaults to 'ex'.
            nfft_factor (int, optional): Factor to multiply with polynomial length to get NFFT
                                         for spectral operations. Ensures sufficient resolution.
                                         Defaults to 4.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            torch.Tensor: 1D complex tensor representing the time-domain RF pulse
                          waveform (unitless shape), of the same length as b_poly_coeffs.
        """
        dev = torch.device(device)
        # Ensure b_coeffs is float32 for internal calculations, to match spectral_factorization_cepstral if it defaults to float32
        b_coeffs = b_poly_coeffs.to(device=dev, dtype=torch.float32)

        if b_coeffs.ndim != 1:
            raise ValueError("b_poly_coeffs must be a 1D tensor.")

        L_b = len(b_coeffs)
        if L_b == 0:
            return torch.tensor([], device=dev, dtype=torch.complex64)

        # --- 1. Compute P(z) = 1 - B(z)B*(z^-1) ---
        # B*(z^-1) for real b_coeffs is just b_coeffs reversed in time.
        b_coeffs_rev = torch.flip(b_coeffs, dims=[0])

        bb_star_coeffs = F.conv1d(b_coeffs.view(1, 1, -1),
                                  b_coeffs_rev.view(1, 1, -1),
                                  padding='full').squeeze() # Length 2*L_b - 1

        p_coeffs = torch.zeros_like(bb_star_coeffs, device=dev, dtype=torch.float32)
        center_idx_p = L_b - 1 # Center of p_coeffs (length 2*L_b - 1)
        p_coeffs[center_idx_p] = 1.0
        p_coeffs = p_coeffs - bb_star_coeffs

        # --- 2. Spectral Factorization of P(z) to get A(z) ---
        # A(z) should also have length L_b
        # Ensure nfft is at least the length of p_coeffs for spectral_factorization_cepstral
        min_nfft_spec_fact = len(p_coeffs)
        nfft_spec_fact = nfft_factor * min_nfft_spec_fact
        if nfft_spec_fact < min_nfft_spec_fact : nfft_spec_fact = min_nfft_spec_fact


        # Make sure spectral_factorization_cepstral is imported
        # from mri_pulse_library.core.dsp_utils import spectral_factorization_cepstral
        a_coeffs = spectral_factorization_cepstral(
            p_coeffs_real_symmetric=p_coeffs,
            target_a_length=L_b,
            nfft=nfft_spec_fact
        ).to(device=dev, dtype=torch.float32)

        # Store for potential debugging or advanced checks (optional)
        self._a_coeffs_last = a_coeffs
        self._b_coeffs_last = b_coeffs

        # --- 3. Calculate RF pulse from A(z) and B(z) ---
        min_nfft_rf = L_b
        nfft_rf = nfft_factor * min_nfft_rf
        if nfft_rf < min_nfft_rf: nfft_rf = min_nfft_rf


        if pulse_type == 'ex':
            # For excitation: RF(z) = B(z) / A_star_reflected(z)
            # A_star_reflected(z) means A*(1/z*). For real a_coeffs, this is time-reversed a_coeffs.
            a_coeffs_star_reflected = torch.flip(a_coeffs, dims=[0])

            B_omega = torch.fft.fft(b_coeffs, n=nfft_rf)
            A_star_reflected_omega = torch.fft.fft(a_coeffs_star_reflected, n=nfft_rf)

            RF_omega = B_omega / (A_star_reflected_omega + 1e-12) # Add epsilon for stability

            rf_pulse_full = torch.fft.ifft(RF_omega)
            rf_pulse = rf_pulse_full[:L_b].to(dtype=torch.complex64)

        # elif pulse_type in ['se', 'inv', 'sat']:
            # For refocusing (symmetric B(z)): RF(z) = B(z) / A(z)
            # A_omega = torch.fft.fft(a_coeffs, n=nfft_rf)
            # B_omega = torch.fft.fft(b_coeffs, n=nfft_rf)
            # RF_omega = B_omega / (A_omega + 1e-12) # Add epsilon
            # rf_pulse_full = torch.fft.ifft(RF_omega)
            # rf_pulse = rf_pulse_full[:L_b].to(dtype=torch.complex64)
            # raise NotImplementedError(f"SLR RF calculation for pulse_type '{pulse_type}' is not fully verified/implemented yet.")
        else:
            raise NotImplementedError(f"SLR RF calculation for pulse_type '{pulse_type}' is not yet implemented.")

        return rf_pulse

    def design_rf_pulse_from_mag_specs(
        self,
        desired_mag_ripples: List[float],
        desired_mag_amplitudes: List[float],
        nominal_flip_angle_rad: float,
        pulse_type: str,
        num_taps_b_poly: int,
        # Parameters for FIR design of B(z) - using Parks-McClellan for this example
        fir_bands_normalized: List[float], # e.g., [0, f_pass, f_stop, 1.0] (Nyquist=1.0)
        fir_desired_b_gains: List[float],  # e.g., [1, 0] for B(z) as lowpass
        fir_weights: List[float] = None,   # Weights for FIR design bands
        nfft_factor_b2rf: int = 4,         # NFFT factor for b_poly_to_rf
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, float]:
        """
        Designs an RF pulse from magnetization profile specifications using the SLR method.

        This method orchestrates the following steps:
        1. Converts magnetization specs to B-polynomial specifications using
           `magnetization_to_b_poly_specs`.
        2. Designs the B-polynomial FIR filter coefficients using Parks-McClellan
           (`FIRFilterDesigner.design_parks_mcclellan_real`).
        3. Normalizes the B-polynomial coefficients to ensure max|B(e^jω)| <= 1.
        4. Converts the B-polynomial coefficients to an RF pulse using `b_poly_to_rf`.

        Args:
            desired_mag_ripples (List[float]): Ripples in the target magnetization profile
                                               (e.g., [d_pass, d_stop]).
            desired_mag_amplitudes (List[float]): Amplitudes in the target magnetization
                                                  profile (e.g., [1, 0] for Mxy excitation).
            nominal_flip_angle_rad (float): Overall target flip angle in radians.
            pulse_type (str): Type of pulse ('ex', 'se', 'inv', 'sat').
            num_taps_b_poly (int): Number of taps for the B-polynomial FIR filter.
            fir_bands_normalized (List[float]): Frequency band edges for FIR design of B(z),
                                                normalized to Nyquist (0 to 1.0).
                                                E.g., [0, f_pass, f_stop, 1.0].
            fir_desired_b_gains (List[float]): Desired gains for B(z) in each band defined
                                               by fir_bands_normalized. Length must be
                                               len(fir_bands_normalized)/2. E.g., [1,0] for lowpass B.
            fir_weights (List[float], optional): Weights for each band in FIR design.
                                                 Defaults to equal weights if None.
            nfft_factor_b2rf (int, optional): NFFT factor for b_poly_to_rf step. Defaults to 4.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing:
                - rf_pulse (torch.Tensor): The designed complex RF pulse waveform (unitless shape).
                - adjusted_flip_angle_rad (float): The potentially adjusted overall flip angle.
        """
        dev = torch.device(device)

        # 1. Convert magnetization specs to B-polynomial specifications
        _, _, adjusted_flip_angle_rad = self.magnetization_to_b_poly_specs(
            desired_mag_ripples, desired_mag_amplitudes, nominal_flip_angle_rad, pulse_type, device
        )

        # 2. Design B-polynomial FIR filter coefficients
        if fir_weights is None:
            fir_weights = [1.0] * len(fir_desired_b_gains)

        b_poly_coeffs = FIRFilterDesigner.design_parks_mcclellan_real(
            num_taps=num_taps_b_poly,
            bands_normalized=fir_bands_normalized,
            desired_amplitudes=fir_desired_b_gains,
            weights=fir_weights,
            device=dev
        ).to(dtype=torch.float32)

        # 3. Normalize B-polynomial coefficients so that max|B(e^jω)| <= 1
        nfft_check = 2 * len(b_poly_coeffs)
        B_omega_check = torch.fft.fft(b_poly_coeffs, n=nfft_check)
        max_abs_B_omega = torch.max(torch.abs(B_omega_check))

        if max_abs_B_omega.item() > 1.0 + 1e-6:
            print(f"Warning: Max|B(e^jω)| = {max_abs_B_omega.item():.4f} > 1. Normalizing b_poly_coeffs.")
            b_poly_coeffs = b_poly_coeffs / max_abs_B_omega
        elif max_abs_B_omega.item() == 0.0:
             print("Warning: B-polynomial is all zeros. RF pulse will be zero.")

        # 4. Convert B-polynomial to RF pulse
        rf_pulse = self.b_poly_to_rf(
            b_poly_coeffs=b_poly_coeffs,
            pulse_type=pulse_type,
            nfft_factor=nfft_factor_b2rf,
            device=dev
        )

        return rf_pulse, adjusted_flip_angle_rad

if __name__ == '__main__':
    print("SLRTransform class defined.")
    slr_transformer = SLRTransform() # Instantiate the class

    # Example for magnetization_to_b_poly_specs (using its actual signature)
    print("\n--- magnetization_to_b_poly_specs Example ---")
    try:
        ripples = [0.01, 0.01]
        amplitudes = [1.0, 0.0]
        flip_angle = math.pi / 2.0
        pulse_type_ex = 'ex'
        device_ex = 'cpu'
        b_rips, b_amps, adj_flip = slr_transformer.magnetization_to_b_poly_specs(
            ripples, amplitudes, flip_angle, pulse_type_ex, device_ex
        )
        print(f"For Mxy_specs (rips={ripples}, amps={amplitudes}, flip={flip_angle:.2f}rad, type='{pulse_type_ex}'):")
        print(f"  B-poly ripples: {b_rips.tolist()}")
        print(f"  B-poly amplitudes: {b_amps.tolist()}")
        print(f"  Adjusted flip angle: {adj_flip:.2f}rad")

    except Exception as e:
        print(f"Error in magnetization_to_b_poly_specs example: {e}")
        import traceback
        traceback.print_exc()

    # ... (The rest of the __main__ block for b_poly_to_rf example, copied from
    #      mri_pulse_library/slr_transform.py, can largely remain as is,
    #      but ensure it calls slr_transformer.b_poly_to_rf(...) correctly) ...
    # ... (The FIRFilterDesigner import and usage should be fine if fir_designer.py is at root)
    # ... (The manual b_coeffs fallback is also fine)
    # ... (The check for A*A + B*B should also be fine)
    print("\n--- b_poly_to_rf Example ---")

    try:
        # from mri_pulse_library.simulators.fir_designer import FIRFilterDesigner # Path if it was in simulators
        # Assuming fir_designer.py is at the root, alongside slr_transform.py
        # from fir_designer import FIRFilterDesigner # This should now be at the top of the file
        fir_designer_available = True # Assume it's imported at top
    except NameError: # If FIRFilterDesigner was not imported due to path issues at top
        print("FIRFilterDesigner class not available (check import at top of file).")
        fir_designer_available = False


    if fir_designer_available:
        try:
            num_taps_b = 31
            bands_fir = [0, 0.2, 0.3, 1.0]
            desired_fir_gains = [1, 0]
            weights_fir = [1, 1]

            b_coeffs_example = FIRFilterDesigner.design_parks_mcclellan_real(
                num_taps=num_taps_b,
                bands_normalized=bands_fir,
                desired_amplitudes=desired_fir_gains,
                weights=weights_fir,
                device='cpu'
            )
            print(f"Example b_poly_coeffs (L={len(b_coeffs_example)}): {b_coeffs_example[:5].numpy()}...{b_coeffs_example[-5:].numpy()}")

            B_omega_val = torch.fft.fft(b_coeffs_example, n=1024)
            max_B_mag = torch.max(torch.abs(B_omega_val))
            if max_B_mag > 1.0:
                print(f"Normalizing b_coeffs_example by factor: {max_B_mag.item()}")
                b_coeffs_example /= max_B_mag

            # Use the already instantiated slr_transformer
            rf_pulse_example = slr_transformer.b_poly_to_rf(b_coeffs_example, pulse_type='ex', device='cpu')

            print(f"Generated RF pulse shape (first 5 points): {rf_pulse_example[:5].real.numpy()} + 1j*{rf_pulse_example[:5].imag.numpy()}")
            print(f"RF pulse length: {len(rf_pulse_example)}")

            if hasattr(slr_transformer, '_a_coeffs_last') and slr_transformer._a_coeffs_last is not None:
                a_coeffs_check = slr_transformer._a_coeffs_last
                A_omega_check = torch.fft.fft(a_coeffs_check, n=1024)
                B_omega_check = torch.fft.fft(b_coeffs_example, n=1024)
                sum_sq_mag_freq = torch.abs(A_omega_check)**2 + torch.abs(B_omega_check)**2
                print(f"Check: Mean of |A(omega)|^2 + |B(omega)|^2: {torch.mean(sum_sq_mag_freq).item()} (should be close to 1.0)")
                if not (0.9 < torch.mean(sum_sq_mag_freq).item() < 1.1):
                     print("Warning: |A|^2 + |B|^2 is not close to 1. Check filter design or factorization.")
            else:
                print("Could not perform A*A+B*B check as _a_coeffs_last was not available.")

        except ImportError:
            print("FIRFilterDesigner import failed inside try block, using very simple manual b_coeffs for b_poly_to_rf example.")
            b_coeffs_manual = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1], dtype=torch.float32)
            B_omega_manual = torch.fft.fft(b_coeffs_manual, n=1024)
            if torch.max(torch.abs(B_omega_manual)) > 1.0:
                 b_coeffs_manual /= torch.max(torch.abs(B_omega_manual))
            rf_pulse_manual = slr_transformer.b_poly_to_rf(b_coeffs_manual, pulse_type='ex', device='cpu')
            print(f"Generated RF pulse with manual b_coeffs (first 5 points): {rf_pulse_manual[:5].real.numpy()} + 1j*{rf_pulse_manual[:5].imag.numpy()}")
            print(f"RF pulse length: {len(rf_pulse_manual)}")
        except NotImplementedError as e:
            print(f"Caught expected error for other pulse types: {e}")
        except Exception as e_main:
            import traceback
            traceback.print_exc()
            print(f"Error in b_poly_to_rf example with FIRFilterDesigner: {e_main}")
    else: # If fir_designer_available is False from the start
        print("FIRFilterDesigner not available, using very simple manual b_coeffs for b_poly_to_rf example.")
        b_coeffs_manual = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1], dtype=torch.float32)
        B_omega_manual = torch.fft.fft(b_coeffs_manual, n=1024)
        if torch.max(torch.abs(B_omega_manual)) > 1.0:
            b_coeffs_manual /= torch.max(torch.abs(B_omega_manual))
        rf_pulse_manual = slr_transformer.b_poly_to_rf(b_coeffs_manual, pulse_type='ex', device='cpu')
        print(f"Generated RF pulse with manual b_coeffs (first 5 points): {rf_pulse_manual[:5].real.numpy()} + 1j*{rf_pulse_manual[:5].imag.numpy()}")
        print(f"RF pulse length: {len(rf_pulse_manual)}")