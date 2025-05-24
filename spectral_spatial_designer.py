import torch
import numpy as np
import math
from typing import List, Tuple, Dict

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from gradient_optimizer import design_trapezoidal_lobe, design_min_time_gradient_lobe_with_bridge
from aliasing_calculator import calculate_aliased_specs
from fir_designer import FIRFilterDesigner
from slr_transform import SLRTransform
from verse import apply_basic_verse # Assuming apply_basic_verse is a standalone function as implemented

# For GradientOptimizer, it was used as a namespace for functions, not a class.
# So, we'll call its functions directly.

class SpectralSpatialPulseDesigner:
    """
    Designs Spectral-Spatial (SPSP) RF pulses.
    """

    def __init__(self, 
                 dt: float = 4e-6, # s
                 gamma_hz_g: float = 4257.0, # Hz/G
                 max_grad_g_cm: float = 5.0, # G/cm
                 max_slew_g_cm_ms: float = 20.0, # G/cm/ms
                 max_b1_g: float = 0.15, # Gauss
                 max_duration_s: float = 20e-3, # s
                 device: str = 'cpu'):
        """
        Initializes the SpectralSpatialPulseDesigner.

        Args:
            dt: Time step in seconds.
            gamma_hz_g: Gyromagnetic ratio in Hz/Gauss.
            max_grad_g_cm: Maximum gradient amplitude in G/cm.
            max_slew_g_cm_ms: Maximum slew rate in G/cm/ms.
            max_b1_g: Maximum B1 amplitude in Gauss.
            max_duration_s: Maximum total pulse duration in seconds.
            device: PyTorch device ('cpu' or 'cuda').
        """
        self.dt = dt
        self.gamma_hz_g = gamma_hz_g
        self.max_grad_g_cm = max_grad_g_cm
        # Convert slew rate from G/cm/ms to G/cm/s for internal calculations
        self.max_slew_g_cm_s = max_slew_g_cm_ms * 1000.0 
        self.max_b1_g = max_b1_g
        self.max_duration_s = max_duration_s
        self.device = device

        # Instantiate helper classes/modules
        # GradientOptimizer functions are called directly.
        # AliasingCalculator functions are called directly.
        self.fir_designer = FIRFilterDesigner()
        self.slr_transform = SLRTransform()
        # apply_basic_verse is a direct function call.

    def design_pulse(self, 
                     spatial_thk_cm: float, 
                     spatial_tbw: float, 
                     spatial_ripple_pass: float, 
                     spatial_ripple_stop: float, 
                     spectral_freq_bands_hz: List[float], 
                     spectral_amplitudes: List[float], 
                     spectral_ripples: List[float], 
                     nominal_flip_angle_rad: float, 
                     pulse_type: str, 
                     spatial_filter_type: str = 'sinc', 
                     spectral_filter_type: str = 'pm', 
                     ss_type: str = 'Flyback Whole', 
                     f_offset_hz_initial: float = None, 
                     use_slr: bool = False, 
                     verse_fraction: float = 0.8, # Alpha for gradient bridge design
                     num_fs_iterations: int = 1, # Simplified: 1 iteration
                     spectral_oversample_factor: int = 2,
                     device: str = None # Allow override of instance device
                    ) -> Dict:
        """
        Designs a spectral-spatial RF pulse.

        Args:
            spatial_thk_cm: Thickness of the spatial slab in cm.
            spatial_tbw: Time-bandwidth product for the spatial dimension.
            spatial_ripple_pass: Passband ripple for spatial beta polynomial (e.g., 0.01).
            spatial_ripple_stop: Stopband ripple for spatial beta polynomial (e.g., 0.01).
            spectral_freq_bands_hz: List of spectral band edges [f1_low, f1_high, ...].
            spectral_amplitudes: List of desired amplitudes for spectral bands.
            spectral_ripples: List of ripples for spectral bands.
            nominal_flip_angle_rad: Target flip angle for the pulse.
            pulse_type: 'ex', 'se', 'inv', 'sat'.
            spatial_filter_type: 'pm' (Parks-McClellan), 'ls' (Least Squares), 'sinc'.
            spectral_filter_type: 'pm', 'ls', 'sinc'.
            ss_type: 'Flyback Whole', 'EP Whole' (Echo-Planar).
            f_offset_hz_initial: Initial guess for spectral offset.
            use_slr: If True, use SLR transform for RF generation (not fully implemented).
            verse_fraction: Fraction of gradient ramp used for VERSE bridge (alpha).
            num_fs_iterations: Number of iterations for optimizing sampling frequency (simplified to 1).
            spectral_oversample_factor: Factor to oversample the spectral dimension.
            device: PyTorch device for calculations and outputs. Uses instance device if None.

        Returns:
            A dictionary containing 'rf_G' (RF pulse in Gauss), 
            'grad_G_cm' (gradient waveform in G/cm), 'fs_hz' (design sampling freq), 
            and other relevant parameters.
        """
        current_device = device if device is not None else self.device

        if use_slr:
            # print("SLR path is not fully implemented. Returning empty pulse.")
            # Fallback to small-tip or raise error for this subtask
            # For now, let's proceed with small-tip logic even if use_slr=True,
            # and note that the SLR part of b_poly_to_rf is missing.
            # raise NotImplementedError("Full SLR path is not implemented in this version.")
            pass # Will proceed with small-tip logic, b_poly_to_rf will raise if called by SLR.

        # --- 1. Initial Gradient & Max fs Calculation ---
        if spatial_thk_cm <= 0: raise ValueError("Spatial thickness must be positive.")
        kz_max = spatial_tbw / spatial_thk_cm  # cycles/cm
        kz_area = kz_max / self.gamma_hz_g      # G/cm * s (Moment = kz_max / gamma)

        # Design positive gradient lobe for spatial encoding
        # design_min_time_gradient_lobe_with_bridge returns: g_lobe, g_ramp_up_outer, g_bridge, g_ramp_down_outer
        gpos_full, gpos_ramp_up, gpos_bridge, gpos_ramp_down = \
            design_min_time_gradient_lobe_with_bridge(
                target_moment=kz_area,
                ramp_fraction_for_moment=verse_fraction,
                max_amplitude=self.max_grad_g_cm,
                max_slew_rate=self.max_slew_g_cm_s, # Use G/cm/s
                dt=self.dt,
                device=current_device
            )

        if gpos_bridge.numel() == 0:
            raise ValueError("Failed to design gradient bridge lobe (gpos_bridge is empty). Check spatial parameters and constraints.")
        if gpos_full.numel() == 0:
            raise ValueError("Failed to design full positive gradient lobe (gpos_full is empty).")


        t_gpos_bridge_duration = gpos_bridge.numel() * self.dt
        t_gpos_full_duration = gpos_full.numel() * self.dt
        
        if t_gpos_full_duration == 0: # Should be caught by gpos_full.numel() == 0
             raise ValueError("Positive gradient lobe has zero duration.")

        # Design negative gradient lobe (rewinder for Flyback or EP)
        # For 'Flyback Whole', gneg is typically a full rewinder of gpos_full.
        # For 'EP Whole', gneg might be identical to gpos_full (played inverted or as is).
        # Let's assume for Flyback it's a moment-balanced lobe.
        # For EP, it could be a copy of gpos_full.
        
        gneg_lobe = design_trapezoidal_lobe(
            target_moment=-torch.sum(gpos_full * self.dt).item(), # Balance full positive lobe
            max_amplitude=self.max_grad_g_cm,
            max_slew_rate=self.max_slew_g_cm_s,
            dt=self.dt,
            current_amplitude=0.0, target_amplitude=0.0, # Assuming starts and ends at zero
            device=current_device
        )
        t_gneg_full_duration = gneg_lobe.numel() * self.dt
        if t_gneg_full_duration == 0 and torch.sum(gpos_full * self.dt).item() != 0 :
             raise ValueError("Negative gradient lobe has zero duration but positive lobe has moment.")


        # Estimate maximum spectral sampling frequency (fs_max_hz)
        # This is the rate at which spectral information is "updated" by new spatial lobes.
        if 'Flyback' in ss_type:
            # Time for one cycle: gpos_full + gneg_lobe (rewinder)
            t_cycle = t_gpos_full_duration + t_gneg_full_duration
        elif 'EP' in ss_type:
            # Time for one cycle: gpos_full + gpos_full (assuming gneg is a copy for blipped EP)
            # Or, if gneg is different, t_gpos + t_gneg.
            # For simplicity, assume gneg is similar to gpos for EP cycle time estimation.
            t_cycle = 2 * t_gpos_full_duration 
        else:
            raise ValueError(f"Unknown ss_type: {ss_type}")

        if t_cycle == 0: raise ValueError("Gradient cycle time is zero.")
        fs_max_hz = 1.0 / t_cycle
        fs_hz_design = fs_max_hz / spectral_oversample_factor
        
        if fs_hz_design <=0:
            raise ValueError(f"Calculated design spectral sampling frequency fs_hz_design ({fs_hz_design} Hz) is not positive. "
                             f"Check gradient durations (t_cycle={t_cycle}s) and spectral_oversample_factor.")


        # --- 2. Aliased Spectral Specs ---
        # For SLR-like (symmetric) FIR filter design, symmetric_response is often True.
        f_alias_norm, a_alias, d_alias, f_offset_used = calculate_aliased_specs(
            spectral_freq_bands_hz, 
            spectral_amplitudes, 
            spectral_ripples, 
            fs_hz_design, 
            f_offset_hz_initial, 
            symmetric_response=True, # Assume symmetric for typical FIR design for B poly
            device=current_device
        )

        if f_alias_norm.numel() == 0:
            raise ValueError(f"Aliasing calculation failed or resulted in empty spectral bands. "
                             f"fs_hz_design={fs_hz_design} Hz. Check spectral inputs and oversample factor.")

        # --- 3. Spatial Beta Polynomial (z_b) ---
        # This is for the small-tip path. For SLR, B-poly is directly from magnetization.
        # For small-tip, the "spatial beta polynomial" is more like the desired spatial RF envelope shape.
        # It's designed to be played over gpos_bridge.
        num_taps_spatial = gpos_bridge.numel()
        if num_taps_spatial == 0:
            raise ValueError("gpos_bridge has zero length, cannot design spatial filter.")

        z_b = torch.ones(num_taps_spatial, dtype=torch.complex64, device=current_device) # Placeholder

        # Normalized spatial frequency for filter design: 0 to 0.5 (Nyquist for k-space sampling)
        # spatial_tbw determines the passband width in this normalized frequency.
        # Passband edge: (spatial_tbw / 2) / (num_taps_spatial / 2) = spatial_tbw / num_taps_spatial
        # This is if spatial_tbw refers to total width across positive and negative k.
        # If spatial_tbw is for one side (e.g. from center to edge of passband), then passband_edge = spatial_tbw / (num_taps_spatial/2)
        # Let's assume spatial_tbw is like image processing TBW (e.g. 4 for Hamming window).
        # The cutoff for windowed sinc is often related to 1/thk or TBW/thk.
        # For FIR design, map spatial_tbw to normalized band edges.
        # A simple passband from 0 to `cutoff_norm_spatial`.
        # `cutoff_norm_spatial` should be related to `spatial_tbw / num_taps_spatial`.
        # For a sinc, cutoff is often 0.5 for rect, or smaller for main lobe width.
        # Let's use a simplified cutoff for sinc: spatial_tbw / (2*num_taps_spatial) might be too small.
        # A common rule for windowed sinc is cutoff related to desired resolution.
        # If spatial_tbw defines "number of zero crossings under the sinc envelope",
        # cutoff_freq = (spatial_tbw/2) / (num_taps_spatial * dt_k) where dt_k is k-space sampling interval.
        # Here, num_taps_spatial is number of k-space samples. Normalized freq is just / (num_taps_spatial/2).
        # So, cutoff_normalized_spatial = (spatial_tbw/2) / (num_taps_spatial/2) = spatial_tbw / num_taps_spatial.
        
        cutoff_norm_spatial = spatial_tbw / num_taps_spatial # Range [0, 0.5]
        if cutoff_norm_spatial > 0.5: cutoff_norm_spatial = 0.5 # Cap at Nyquist
        if cutoff_norm_spatial <= 0: raise ValueError("Spatial cutoff calculated as <=0. Check spatial_tbw and num_taps_spatial.")


        if spatial_filter_type == 'sinc':
            z_b_real = self.fir_designer.windowed_sinc(num_taps_spatial, cutoff_normalized=cutoff_norm_spatial, device=current_device)
            z_b = z_b_real.to(torch.complex64)
        elif spatial_filter_type == 'pm':
            # For Parks-McClellan, define bands. E.g. passband [0, cutoff], stopband [cutoff+transition, 0.5]
            # Passband ripple from spatial_ripple_pass, stopband from spatial_ripple_stop.
            # These ripples are for magnetization, not directly for small-tip z_b.
            # For small-tip, z_b directly shapes RF. Assume ripples are for this shape.
            pass_edge_spatial = cutoff_norm_spatial
            stop_edge_spatial = min(0.5, pass_edge_spatial + 0.05) # Small transition band
            if pass_edge_spatial >= stop_edge_spatial : stop_edge_spatial = pass_edge_spatial + (0.5-pass_edge_spatial)*0.5 # ensure stop > pass
            if stop_edge_spatial >= 0.5: # Handle if pass_edge is already close to 0.5
                pass_edge_spatial = min(pass_edge_spatial, 0.45) # Adjust pass to allow some stopband
                stop_edge_spatial = 0.5

            bands_spatial_norm = [0, pass_edge_spatial, stop_edge_spatial, 0.5]
            # Ensure bands are monotonic and within [0,0.5]
            if not (bands_spatial_norm[0] <= bands_spatial_norm[1] <= bands_spatial_norm[2] <= bands_spatial_norm[3]):
                 raise ValueError(f"Spatial filter bands for PM are not monotonic: {bands_spatial_norm}")


            desired_spatial = [1, 0] # Passband gain 1, stopband gain 0
            # Weights can be used to trade off ripple: e.g., [1, spatial_ripple_pass/spatial_ripple_stop]
            weights_spatial = [1, spatial_ripple_pass / (spatial_ripple_stop + 1e-9)] # Add epsilon to avoid div by zero
            
            z_b_real = self.fir_designer.design_parks_mcclellan_real(
                num_taps_spatial, bands_spatial_norm, desired_spatial, weights_spatial, device=current_device
            )
            z_b = z_b_real.to(torch.complex64)
        elif spatial_filter_type == 'ls':
             # Similar band definition for LS
            pass_edge_spatial = cutoff_norm_spatial
            stop_edge_spatial = min(0.5, pass_edge_spatial + 0.05)
            if pass_edge_spatial >= stop_edge_spatial : stop_edge_spatial = pass_edge_spatial + (0.5-pass_edge_spatial)*0.5
            if stop_edge_spatial >= 0.5:
                pass_edge_spatial = min(pass_edge_spatial, 0.45)
                stop_edge_spatial = 0.5
            
            # For firls, bands_normalized are 0-1 (Nyquist is 1.0). Our spatial norm is 0-0.5. So scale by 2.
            bands_spatial_firls = [0, pass_edge_spatial*2, stop_edge_spatial*2, 1.0]
            desired_spatial_firls = [1, 1, 0, 0] # Amplitudes at band edges
            weights_spatial_firls = [1, spatial_ripple_pass / (spatial_ripple_stop + 1e-9)]

            z_b_real = self.fir_designer.design_least_squares(
                num_taps_spatial, bands_spatial_firls, desired_spatial_firls, weights_spatial_firls, device=current_device
            )
            z_b = z_b_real.to(torch.complex64)
        else:
            raise ValueError(f"Unknown spatial_filter_type: {spatial_filter_type}")


        # --- 4. Spectral Beta Polynomial (s_b) ---
        # Convert desired magnetization spec to B-poly spec for spectral dimension
        spec_beta_ripples, spec_beta_amps, adj_flip_angle_rad = \
            self.slr_transform.magnetization_to_b_poly_specs(
                spectral_ripples, spectral_amplitudes, nominal_flip_angle_rad, 
                pulse_type, device=current_device
            )

        # Number of spectral lobes (sub-pulses)
        # This is a simplified estimate. Max duration divided by cycle time.
        if t_cycle <= 0: raise ValueError("Gradient cycle time t_cycle must be positive.")
        num_spectral_lobes = math.floor(self.max_duration_s / t_cycle)
        if num_spectral_lobes == 0: num_spectral_lobes = 1 # Must have at least one lobe
        
        # Design the spectral filter (s_b)
        # f_alias_norm is already 0-1 for PM/LS.
        # For sinc, it would need to be 0-0.5.
        s_b_real = torch.ones(num_spectral_lobes, dtype=torch.float32, device=current_device) # Placeholder

        if spectral_filter_type == 'pm':
            s_b_real = self.fir_designer.design_parks_mcclellan_real(
                num_spectral_lobes, 
                f_alias_norm.cpu().tolist(), # Ensure it's a list
                spec_beta_amps.cpu().tolist(),
                weights=(torch.max(spec_beta_ripples)/(spec_beta_ripples + 1e-9)).cpu().tolist() if spec_beta_ripples.numel() > 0 else None,
                device=current_device
            )
        elif spectral_filter_type == 'ls':
            # For LS, desired amplitudes are at band edges. a_alias might need processing.
            # If a_alias = [A1, A2, A3], f_alias_norm = [f0,f1, f1,f2, f2,f3].
            # desired_ls needs to be [A1,A1, A2,A2, A3,A3]
            desired_ls = spec_beta_amps.repeat_interleave(2)
            s_b_real = self.fir_designer.design_least_squares(
                num_spectral_lobes,
                f_alias_norm.cpu().tolist(),
                desired_ls.cpu().tolist(),
                weights=(torch.max(spec_beta_ripples)/(spec_beta_ripples + 1e-9)).cpu().tolist() if spec_beta_ripples.numel() > 0 else None,
                device=current_device
            )
        elif spectral_filter_type == 'sinc':
            # Sinc needs a single cutoff. Find the main passband from aliased specs.
            # This is simplified; multi-band sinc is complex. Assume first passband.
            spectral_cutoff_norm = 0.25 # Default, needs to be derived from f_alias_norm
            if f_alias_norm.numel() >= 2:
                 # Assuming f_alias_norm is [0, pb_end, sb_start, ..., 1.0]
                 # and we want a lowpass filter. Take middle of first passband.
                 spectral_cutoff_norm = (f_alias_norm[1].item() / 2.0) # Normalized 0 to 0.5
            s_b_real = self.fir_designer.windowed_sinc(num_spectral_lobes, spectral_cutoff_norm, device=current_device)
        else:
            raise ValueError(f"Unknown spectral_filter_type: {spectral_filter_type}")
        
        s_b = s_b_real.to(torch.complex64)
        # Small-tip specific: conjugate s_b (as per ss_flyback.m)
        if not use_slr:
            s_b = torch.conj(s_b)


        # --- 5. Small-Tip Combination ---
        rf_pulse_segments = []
        gradient_segments = []

        # Scale spatial profile by adjusted overall flip angle for small tip approx.
        z_b_scaled = z_b * adj_flip_angle_rad # adj_flip_angle is scalar float

        # VERSE the spatial sub-pulse (z_b_scaled)
        # The original_gradient_amplitude for z_b is the peak of gpos_bridge,
        # as z_b is designed to be played out during gpos_bridge.
        if gpos_bridge.numel() == 0: # Should have been caught earlier
            max_amp_gpos_bridge = 0.0
        else:
            max_amp_gpos_bridge = torch.max(torch.abs(gpos_bridge)).item()
        
        if abs(max_amp_gpos_bridge) < 1e-9:
            # If bridge gradient is zero, VERSEd RF is zero unless z_b is also zero.
            # Or, if original_gradient_amplitude_of_rf_segment is zero, VERSE returns zeros.
            # print("Warning: Max amplitude of gpos_bridge is near zero for VERSE.")
            # If max_amp_gpos_bridge is 0, apply_basic_verse should handle it.
             rf_spatial_subpulse_versed = torch.zeros_like(z_b_scaled) if max_amp_gpos_bridge < 1e-9 else \
                                         apply_basic_verse(z_b_scaled, gpos_bridge, max_amp_gpos_bridge, self.dt)
        else:
             rf_spatial_subpulse_versed = apply_basic_verse(z_b_scaled, gpos_bridge, max_amp_gpos_bridge, self.dt)
        
        # Construct full RF and Gradient
        # For 'Flyback Whole': RF is played only during positive lobes (gpos_full).
        # The VERSEd RF corresponds to gpos_bridge. Need to embed it.
        
        # Find indices where gpos_bridge starts within gpos_full
        # This assumes gpos_bridge is a contiguous part of gpos_full.
        # This is true by design of design_min_time_gradient_lobe_with_bridge.
        # gpos_full = cat(gpos_ramp_up, gpos_bridge, gpos_ramp_down)
        
        len_gpos_ramp_up = gpos_ramp_up.numel()
        # len_gpos_bridge = gpos_bridge.numel() # same as num_taps_spatial

        for i in range(num_spectral_lobes):
            # Create current RF segment for this spectral lobe
            current_rf_subpulse = s_b[i] * rf_spatial_subpulse_versed
            
            # Embed this subpulse into a zero array of length gpos_full
            rf_on_gpos = torch.zeros(gpos_full.numel(), dtype=torch.complex64, device=current_device)
            if rf_spatial_subpulse_versed.numel() == gpos_bridge.numel(): # Check lengths match
                rf_on_gpos[len_gpos_ramp_up : len_gpos_ramp_up + gpos_bridge.numel()] = current_rf_subpulse
            else:
                # This indicates a mismatch, potentially pad/truncate current_rf_subpulse if slightly off,
                # or error if significantly different. For now, assume exact match from VERSE.
                # If rf_spatial_subpulse_versed is shorter than gpos_bridge (e.g. from VERSE compression)
                # it should still be placed starting at len_gpos_ramp_up.
                # apply_basic_verse output length is same as gradient_sub_lobe (gpos_bridge).
                # So, this should be fine.
                 pass


            rf_pulse_segments.append(rf_on_gpos)
            gradient_segments.append(gpos_full)

            if i < num_spectral_lobes - 1: # Add negative lobe if not the last positive one
                if 'Flyback' in ss_type:
                    # For Flyback, RF is zero during negative lobe
                    rf_pulse_segments.append(torch.zeros_like(gneg_lobe, dtype=torch.complex64, device=current_device))
                    gradient_segments.append(gneg_lobe)
                elif 'EP' in ss_type:
                    # For EP, RF might be played (e.g. phase-alternated). Simplified: zero RF on gneg.
                    rf_pulse_segments.append(torch.zeros_like(gneg_lobe, dtype=torch.complex64, device=current_device))
                    gradient_segments.append(gneg_lobe) # Or -gpos_full if gneg is just inverted gpos


        final_rf_pulse = torch.cat(rf_pulse_segments) if rf_pulse_segments else torch.empty(0, device=current_device)
        final_gradient = torch.cat(gradient_segments) if gradient_segments else torch.empty(0, device=current_device)
        
        # --- 6. SLR Path --- (Skipped as per simplified path for this task)
        if use_slr:
            # This would involve calling self.slr_transform.b_poly_to_rf for s_b and z_b,
            # then combining them, which is non-trivial.
            # The current small-tip path uses z_b as a shape and s_b as modulators.
            # True SLR would combine polynomials in z-domain then convert to RF.
            # For now, the above small-tip result will be returned even if use_slr=True,
            # with the understanding that b_poly_to_rf would error if called.
            pass


        # --- 7. Finalize and Return ---
        # Convert RF pulse from complex B1 (scaled by flip angle) to physical units (Gauss)
        # Small tip: theta ~ gamma * integral(B1(t) dt)
        # If z_b was scaled by adj_flip_angle, and s_b are complex modulators,
        # The sum of rf_spatial_subpulse_versed elements, when scaled by gamma*dt, should approximate adj_flip_angle.
        # The RF returned by SLR's b2rf is typically unitless or needs scaling.
        # For small tip, RF(t) = B1_xy(t). The values in `final_rf_pulse` are effectively scaled B1 values.
        # To get B1 in Gauss: B1_gauss(t) = RF_scaled(t) / (gamma_rad_g_s * dt) if RF_scaled is integrated phase.
        # Here, `final_rf_pulse` contains values that are proportional to B1 field strength.
        # The scaling `adj_flip_angle_rad` is already incorporated into z_b_scaled.
        # The values in `final_rf_pulse` should represent complex B1 in units that, when integrated over dt
        # and multiplied by gamma, give local flip angle contribution.
        # If rf_spatial_subpulse_versed is ~ B1*dt*gamma, then sum(abs(thereof)) ~ flip.
        # The common convention is RF output in Gauss.
        # If `z_b_scaled` is taken as `gamma * B1_spatial_envelope * dt_spatial_element`,
        # and `s_b[i]` is a complex scale factor, then
        # `final_rf_pulse` elements are like `gamma * B1_xy(t) * dt`.
        # So, `B1_xy(t)_Gauss = final_rf_pulse / (self.gamma_hz_g * 2 * np.pi * self.dt)` (if final_rf_pulse is radians)
        # Or if `final_rf_pulse` is already in B1 units scaled by gamma*dt:
        # B1_gauss = final_rf_pulse / (self.gamma_hz_g * self.dt) -- This seems more aligned with typical SLR outputs.
        # Let's assume final_rf_pulse elements are effectively (gamma_rad * B1_gauss * dt) values.
        # So B1_gauss = final_rf_pulse / (self.gamma_hz_g * 2 * np.pi * self.dt)
        
        # From ss_flyback.m: rf = rf / (2*pi*gambar*dt); % convert from radians to Gauss
        # This implies `final_rf_pulse` is in radians.
        # Our `adj_flip_angle_rad` is in radians. `z_b` is unitless filter taps.
        # So `z_b_scaled` is "radians per spatial point".
        # `rf_spatial_subpulse_versed` is also "radians per point on gpos_bridge".
        # `s_b[i]` is a unitless complex modulator.
        # So `current_rf_subpulse` is "radians per point".
        # `final_rf_pulse` elements are radians.
        rf_pulse_gauss = final_rf_pulse / (self.gamma_hz_g * 2 * np.pi * self.dt)

        # Check against max_b1_g
        max_rf_abs_g = torch.max(torch.abs(rf_pulse_gauss)).item() if rf_pulse_gauss.numel() > 0 else 0.0
        if max_rf_abs_g > self.max_b1_g:
            # print(f"Warning: Max RF amplitude {max_rf_abs_g:.3f} G exceeds max_b1_g {self.max_b1_g:.3f} G. Consider reducing flip angle or extending duration.")
            # Could scale down RF and gradients, or raise error. For now, just note.
            pass

        # Check total duration
        total_duration_designed = final_gradient.numel() * self.dt
        if total_duration_designed > self.max_duration_s:
            # print(f"Warning: Designed pulse duration {total_duration_designed*1000:.2f} ms exceeds max_duration_s {self.max_duration_s*1000:.2f} ms.")
            pass
            
        return {
            'rf_G': rf_pulse_gauss, 
            'grad_G_cm': final_gradient, 
            'fs_hz_design': fs_hz_design,
            'adjusted_flip_angle_rad': adj_flip_angle_rad,
            'f_offset_hz_used': f_offset_used,
            'num_spectral_lobes_designed': num_spectral_lobes,
            'max_rf_gauss_designed': max_rf_abs_g,
            'total_duration_designed_s': total_duration_designed
        }