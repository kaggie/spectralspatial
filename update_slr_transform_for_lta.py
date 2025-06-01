import torch
import numpy as np # For math.pi if needed, and other utilities
# from mri_pulse_library.core.bloch_sim import bloch_simulate # Assuming this is the single-channel version
# Not importing bloch_simulate at the top of the script to avoid circular dependencies if this script modifies a file that bloch_sim might depend on.
# The method code itself will need it, so the target file slr_transform.py should have it.
from typing import Tuple # For type hinting

def add_lta_method_to_slr_transform(file_path="slr_transform.py"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Cannot add LTA method.")
        return

    class_end_line_idx = -1
    main_block_markers = ["if __name__ == '__main__':", "# Main block for examples"]

    for i in range(len(lines) -1, -1, -1):
        for marker in main_block_markers:
            if marker in lines[i]:
                class_end_line_idx = i
                break
        if class_end_line_idx != -1:
            break

    if class_end_line_idx == -1:
        # Fallback: find end of class SLRTransform definition by searching for start and then unindented lines
        class_start_idx = -1
        for i, line in enumerate(lines):
            if "class SLRTransform:" in line or "class SLRTransform(object):" in line :
                class_start_idx = i
                break

        if class_start_idx != -1:
            # Find the end of the class by looking for the next line that is not indented
            # This assumes standard Python indentation (4 spaces for methods)
            class_indentation = -1
            if class_start_idx + 1 < len(lines):
                line_after_class_def = lines[class_start_idx+1]
                class_indentation = len(line_after_class_def) - len(line_after_class_def.lstrip())

            if class_indentation > 0:
                for i in range(class_start_idx + 1, len(lines)):
                    current_indentation = len(lines[i]) - len(lines[i].lstrip())
                    if lines[i].strip() == "" or lines[i].strip().startswith("#"): # Allow empty lines or comments
                        if i + 1 < len(lines): # Check next line if current is skippable
                             next_line_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                             if next_line_indent < class_indentation and lines[i+1].strip() != "":
                                 class_end_line_idx = i # Insert before this less-indented line
                                 break
                        else: # Reached end of file
                            class_end_line_idx = i + 1
                            break
                    elif current_indentation < class_indentation :
                        class_end_line_idx = i # Found line that ends the class
                        break
                if class_end_line_idx == -1 : # If loop finished, class ends at end of file
                    class_end_line_idx = len(lines)
            else: # No indented lines after class def, highly unlikely for a real class
                 print(f"Warning: Could not determine indentation for SLRTransform class in {file_path}.")


    lta_method_code = """
    def iterative_lta_rf_scale(
        self,
        initial_rf_gauss: torch.Tensor,
        target_flip_angle_rad: float,
        gradient_waveform_gcms: torch.Tensor,
        dt_s: float,
        gyromagnetic_ratio_hz_g: float = 4257.0,
        num_iterations: int = 10,
        b0_offset_hz: float = 0.0,
        target_tolerance_rad: float = 1e-3,
        max_b1_amplitude_gauss: float = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, float]:
        \"\"\"
        Iteratively scales an initial RF pulse to achieve a target on-resonance
        flip angle, accounting for Large-Tip Angle (LTA) effects.

        Args:
            initial_rf_gauss (torch.Tensor): Complex initial RF waveform (Gauss).
            target_flip_angle_rad (float): Desired on-resonance flip angle (radians).
            gradient_waveform_gcms (torch.Tensor): Slice-select gradient (G/cm)
                                                 active during the RF pulse. Shape (N_timepoints,).
            dt_s (float): Time step of RF and gradient waveforms (seconds).
            gyromagnetic_ratio_hz_g (float, optional): Gyromagnetic ratio in Hz/Gauss.
            num_iterations (int, optional): Maximum number of scaling iterations. Defaults to 10.
            b0_offset_hz (float, optional): B0 off-resonance for simulation. Defaults to 0.0.
            target_tolerance_rad (float, optional): Stop if achieved flip angle is within
                                                    this tolerance of the target. Defaults to 1e-3.
            max_b1_amplitude_gauss (float, optional): If provided, the peak amplitude of the
                                                      scaled RF pulse (in Gauss) will be clipped.
                                                      Defaults to None.
            device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            Tuple[torch.Tensor, float]:
                - scaled_rf_gauss (torch.Tensor): The LTA-corrected RF pulse (Gauss).
                - final_achieved_flip_angle_rad (float): Flip angle after final iteration.
        \"\"\"
        # This method requires bloch_simulate. Ensure it's imported in the target file.
        from mri_pulse_library.core.bloch_sim import bloch_simulate
        import numpy # For np.rad2deg, should be available if torch is

        dev = torch.device(device)
        if not isinstance(initial_rf_gauss, torch.Tensor):
            raise ValueError("initial_rf_gauss must be a PyTorch Tensor.")
        if not initial_rf_gauss.is_complex():
            if self.verbose:
                self._log("Warning: initial_rf_gauss for LTA scaling was real. Casting to complex.")
            current_rf_gauss = initial_rf_gauss.to(dtype=torch.complex64, device=dev)
        else:
            current_rf_gauss = initial_rf_gauss.to(device=dev)

        if not isinstance(gradient_waveform_gcms, torch.Tensor):
            raise ValueError("gradient_waveform_gcms must be a PyTorch Tensor.")

        sim_grad = gradient_waveform_gcms.to(device=dev, dtype=torch.float32)
        if sim_grad.ndim > 1 and sim_grad.shape[-1] == 1: # Ensure it's effectively 1D for Gz
            sim_grad = sim_grad.squeeze(-1)
        if sim_grad.ndim != 1:
            raise ValueError("gradient_waveform_gcms should be 1D (Gz) of shape (N_timepoints,).")


        if len(current_rf_gauss) != len(sim_grad):
            raise ValueError("RF and Gradient waveforms must have the same number of timepoints.")

        achieved_flip_angle_rad = 0.0

        for iteration in range(num_iterations):
            sim_rf = current_rf_gauss.to(dtype=torch.complex64)

            M_final_xyz_np = bloch_simulate(
                rf=sim_rf.cpu(),
                grad=sim_grad.cpu(),
                dt=dt_s,
                gamma=gyromagnetic_ratio_hz_g,
                b0=b0_offset_hz,
                spatial_positions=torch.tensor([0.0]),
                T1=1e6, T2=1e6,
                return_all=False
            )

            M_final = torch.from_numpy(M_final_xyz_np).to(dev).squeeze()

            m_xy = torch.sqrt(M_final[0]**2 + M_final[1]**2)
            m_z = M_final[2]
            achieved_flip_angle_rad = torch.atan2(m_xy, m_z).item()

            if self.verbose:
                self._log(f"LTA Iteration {iteration+1}/{num_iterations}: Achieved FA = {numpy.rad2deg(achieved_flip_angle_rad):.2f} deg (Target: {numpy.rad2deg(target_flip_angle_rad):.2f} deg)")

            if abs(achieved_flip_angle_rad - target_flip_angle_rad) <= target_tolerance_rad:
                if self.verbose:
                    self._log("Target flip angle achieved within tolerance.")
                break

            if abs(achieved_flip_angle_rad) < 1e-9: # Avoid division by zero
                if self.verbose:
                    self._log("Warning: Achieved flip angle is near zero. Cannot compute scaling factor. Stopping.")
                break

            scale_factor = target_flip_angle_rad / achieved_flip_angle_rad

            if self.verbose and abs(scale_factor-1.0) > 0.5 :
                 self._log(f"Warning: Large scaling factor {scale_factor:.2f} applied. Result might be unstable.")

            current_rf_gauss = current_rf_gauss * scale_factor

            if max_b1_amplitude_gauss is not None:
                peak_b1_current = torch.max(torch.abs(current_rf_gauss))
                if peak_b1_current > max_b1_amplitude_gauss:
                    if max_b1_amplitude_gauss == 0:
                        clipping_scale = 0.0
                    elif peak_b1_current > 1e-12:
                        clipping_scale = max_b1_amplitude_gauss / peak_b1_current
                    else:
                        clipping_scale = 1.0

                    current_rf_gauss = current_rf_gauss * clipping_scale
                    if self.verbose:
                        self._log(f"  RF pulse peak B1 ({peak_b1_current*1e3:.2f} mG) exceeded limit. Clipped to {max_b1_amplitude_gauss*1e3:.2f} mG.")
        else:
            if self.verbose and num_iterations > 0 :
                self._log(f"LTA iterations finished. Final achieved FA = {numpy.rad2deg(achieved_flip_angle_rad):.2f} deg")

        return current_rf_gauss.to(initial_rf_gauss.device if isinstance(initial_rf_gauss, torch.Tensor) else dev), achieved_flip_angle_rad
"""
    if class_end_line_idx != -1:
        indented_lta_code = "\n" + "\n".join(["    " + line if line.strip() else line for line in lta_method_code.splitlines()]) + "\n"

        # Check if the method already exists to avoid duplication (simple check)
        method_def_line = "def iterative_lta_rf_scale("
        already_exists = any(method_def_line in line for line in lines)

        if not already_exists:
            lines.insert(class_end_line_idx, indented_lta_code)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Successfully added 'iterative_lta_rf_scale' method to {file_path}")
            except Exception as e:
                print(f"Error writing updated content to {file_path}: {e}")
                print("LTA method code (intended for manual integration):")
                print(lta_method_code)
        else:
            print(f"Method 'iterative_lta_rf_scale' already found in {file_path}. No changes made.")

    else:
        print(f"Error: Could not find suitable insertion point in {file_path}.")
        print("LTA method code (intended for manual integration):")
        print(lta_method_code)


if __name__ == '__main__':
    # This script is meant to modify slr_transform.py, so provide its path
    slr_file_path = os.path.join("mri_pulse_library", "slr_transform.py")
    # Check if the file exists in the expected relative path for typical execution
    if not os.path.exists(slr_file_path):
        # Fallback to just "slr_transform.py" if not found in specific lib path
        # This might happen if script is run from within the mri_pulse_library dir
        slr_file_path = "slr_transform.py"
        if not os.path.exists(slr_file_path):
             print(f"Script {__file__} could not find {slr_file_path} or mri_pulse_library/slr_transform.py")
             slr_file_path = input("Please enter the correct path to slr_transform.py: ")


    add_lta_method_to_slr_transform(slr_file_path)
