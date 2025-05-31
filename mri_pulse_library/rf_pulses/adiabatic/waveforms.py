"""Generation of common adiabatic RF pulse waveforms."""
import torch
import numpy as np # For np.zeros_like in examples
from mri_pulse_library.core.bloch_sim import bloch_simulate
from mri_pulse_library.core.constants import GAMMA_HZ_PER_G_PROTON, GAMMA_HZ_PER_T_PROTON
# import numpy as np # Not strictly needed if using torch.pi, torch.tanh etc.

def generate_hs_pulse_waveforms(
    pulse_duration_s: float,
    num_timepoints: int,
    peak_b1_tesla: float,
    bandwidth_hz: float,
    beta: float,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates amplitude and frequency modulation waveforms for a Hyperbolic Secant (HS)
    adiabatic pulse, based on the Silver-Hoult type formulation.

    The pulse is defined over a symmetric time interval [-pulse_duration_s/2, pulse_duration_s/2].

    Amplitude Modulation A(t): A_max * sech(beta * tau)
    Frequency Modulation f(t): (bandwidth_hz / 2) * tanh(beta * tau)
    where tau = 2 * t / pulse_duration_s.

    Args:
        pulse_duration_s (float): Total duration of the pulse in seconds.
        num_timepoints (int): Number of time points to sample the waveform.
        peak_b1_tesla (float): Peak B1 amplitude (A_max) of the pulse in Tesla.
        bandwidth_hz (float): Full bandwidth of the frequency sweep in Hz.
        beta (float): Parameter controlling the shape and truncation of the
                      hyperbolic secant and tangent functions. A common value is ~5-10.
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - time_s (torch.Tensor): Time vector in seconds, shape (num_timepoints,).
            - amplitude_tesla (torch.Tensor): RF amplitude waveform in Tesla,
                                             shape (num_timepoints,).
            - freq_offset_hz (torch.Tensor): RF frequency offset waveform in Hz,
                                            shape (num_timepoints,).
    """
    if pulse_duration_s <= 0:
        raise ValueError("Pulse duration must be positive.")
    if num_timepoints <= 1:
        raise ValueError("Number of timepoints must be greater than 1.")
    if peak_b1_tesla < 0:
        raise ValueError("Peak B1 amplitude cannot be negative.")
    if bandwidth_hz < 0:
        raise ValueError("Bandwidth cannot be negative.")
    if beta <= 0:
        raise ValueError("Beta parameter must be positive.")

    dev = torch.device(device)

    time_s = torch.linspace(-pulse_duration_s / 2.0, pulse_duration_s / 2.0, num_timepoints, device=dev, dtype=torch.float32)
    tau = 2.0 * time_s / pulse_duration_s

    amplitude_tesla = peak_b1_tesla / torch.cosh(beta * tau)
    freq_offset_hz = (bandwidth_hz / 2.0) * torch.tanh(beta * tau)

    return time_s, amplitude_tesla, freq_offset_hz

if __name__ == '__main__':
    print("Adiabatic HS pulse waveform generator defined.")

    duration = 10e-3
    n_points = 1024
    b1_max = 15e-6
    bw_hz = 4000
    beta_val = 5.0

    try:
        time_vec, amp_vec, freq_vec = generate_hs_pulse_waveforms(
            pulse_duration_s=duration,
            num_timepoints=n_points,
            peak_b1_tesla=b1_max,
            bandwidth_hz=bw_hz,
            beta=beta_val,
            device='cpu'
        )

        print(f"Generated waveforms with {len(time_vec)} points.")
        print(f"Time vector: min={time_vec.min().item():.4f} s, max={time_vec.max().item():.4f} s")
        print(f"Amplitude vector (Tesla): min={amp_vec.min().item():.2e}, max={amp_vec.max().item():.2e}")
        print(f"Frequency offset vector (Hz): min={freq_vec.min().item():.2f}, max={freq_vec.max().item():.2f}")

        # To visualize (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 1, sharex=True)
        # axs[0].plot(time_vec.numpy(), amp_vec.numpy())
        # axs[0].set_title('Amplitude Modulation (Tesla)')
        # axs[1].plot(time_vec.numpy(), freq_vec.numpy())
        # axs[1].set_title('Frequency Modulation (Hz)')
        # axs[1].set_xlabel('Time (s)')
        # plt.show()

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_bir4_waveforms(
    segment_duration_s: float,
    num_timepoints_segment: int,
    peak_b1_tesla_segment: float,
    bandwidth_hz_segment: float,
    beta_segment: float,
    segment_phases_rad: list = None, # List or tuple of 4 phases in radians
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates amplitude and frequency modulation waveforms for a BIR-4
    (B1-Insensitive Rotation) adiabatic pulse. This implementation constructs
    the BIR-4 pulse from four identical Hyperbolic Secant (HS) full-passage
    segments with specified phase shifts applied to each segment.

    Args:
        segment_duration_s (float): Duration of EACH of the 4 HS segments in seconds.
        num_timepoints_segment (int): Number of time points PER HS segment.
        peak_b1_tesla_segment (float): Peak B1 amplitude of each HS segment in Tesla.
        bandwidth_hz_segment (float): Full bandwidth of the frequency sweep for each
                                      HS segment in Hz.
        beta_segment (float): Beta parameter controlling the shape of each HS segment.
        segment_phases_rad (list or tuple, optional): A list or tuple of 4 phase values
            (in radians) to be applied to the four HS segments respectively.
            Defaults to [0, torch.pi, 0, torch.pi].
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - time_s (torch.Tensor): Concatenated time vector for the full BIR-4 pulse,
                                     shape (4 * num_timepoints_segment,). Units: seconds.
            - amplitude_tesla_complex (torch.Tensor): Complex RF amplitude waveform,
                incorporating phase shifts. Shape: (4 * num_timepoints_segment,). Units: Tesla.
            - freq_offset_hz (torch.Tensor): Concatenated RF frequency offset waveform,
                                             shape (4 * num_timepoints_segment,). Units: Hz.
    """
    if segment_duration_s <= 0:
        raise ValueError("Segment duration must be positive.")
    if num_timepoints_segment <= 1:
        raise ValueError("Number of timepoints per segment must be greater than 1.")

    if segment_phases_rad is None:
        segment_phases_rad = [0, torch.pi, 0, torch.pi] # Default BIR-4 phases
    if not isinstance(segment_phases_rad, (list, tuple)) or len(segment_phases_rad) != 4:
        raise ValueError("segment_phases_rad must be a list or tuple of 4 phase values.")

    dev = torch.device(device)
    dtype_complex = torch.complex64

    all_time_segments = []
    all_amp_segments_complex = []
    all_freq_segments = []

    current_time_offset_s = 0.0

    for i in range(4):
        # Generate waveforms for a single HS segment
        # generate_hs_pulse_waveforms returns time centered around 0 for the segment
        time_seg_centered, amp_seg, freq_seg = generate_hs_pulse_waveforms(
            pulse_duration_s=segment_duration_s,
            num_timepoints=num_timepoints_segment,
            peak_b1_tesla=peak_b1_tesla_segment,
            bandwidth_hz=bandwidth_hz_segment,
            beta=beta_segment,
            device=device # Pass through the device
        )

        # Shift time for this segment
        # The first point of time_seg_centered is -segment_duration_s / 2.0
        # We want the first point of the shifted segment to be current_time_offset_s
        time_seg_shifted = time_seg_centered - time_seg_centered[0] + current_time_offset_s
        all_time_segments.append(time_seg_shifted)

        # Apply phase to the amplitude segment
        phase_val = torch.tensor(segment_phases_rad[i], device=dev, dtype=torch.float32) # Ensure phase is float for exp
        phase_factor = torch.exp(1j * phase_val).to(dtype_complex)
        amp_seg_phased = amp_seg.to(dtype_complex) * phase_factor # amp_seg is real, make it complex with phase
        all_amp_segments_complex.append(amp_seg_phased)

        all_freq_segments.append(freq_seg)

        current_time_offset_s += segment_duration_s # Prepare offset for the next segment

    # Concatenate all segments
    final_time_s = torch.cat(all_time_segments)
    final_amplitude_tesla_complex = torch.cat(all_amp_segments_complex)
    final_freq_offset_hz = torch.cat(all_freq_segments)

    return final_time_s, final_amplitude_tesla_complex, final_freq_offset_hz

if __name__ == '__main__':
    print("Adiabatic HS pulse waveform generator defined.")

    duration = 10e-3
    n_points = 1024
    b1_max = 15e-6
    bw_hz = 4000
    beta_val = 5.0

    try:
        time_vec, amp_vec, freq_vec = generate_hs_pulse_waveforms(
            pulse_duration_s=duration,
            num_timepoints=n_points,
            peak_b1_tesla=b1_max,
            bandwidth_hz=bw_hz,
            beta=beta_val,
            device='cpu'
        )

        print(f"Generated waveforms with {len(time_vec)} points.")
        print(f"Time vector: min={time_vec.min().item():.4f} s, max={time_vec.max().item():.4f} s")
        print(f"Amplitude vector (Tesla): min={amp_vec.min().item():.2e}, max={amp_vec.max().item():.2e}")
        print(f"Frequency offset vector (Hz): min={freq_vec.min().item():.2f}, max={freq_vec.max().item():.2f}")

        # To visualize (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 1, sharex=True)
        # axs[0].plot(time_vec.numpy(), amp_vec.numpy())
        # axs[0].set_title('Amplitude Modulation (Tesla)')
        # axs[1].plot(time_vec.numpy(), freq_vec.numpy())
        # axs[1].set_title('Frequency Modulation (Hz)')
        # axs[1].set_xlabel('Time (s)')
        # plt.show()

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- BIR-4 Pulse Waveform Generation Example ---")
    bir4_seg_dur = 2.5e-3  # 2.5 ms per segment
    bir4_n_points_seg = 256
    bir4_b1_max_seg = 15e-6
    bir4_bw_seg = 4000
    bir4_beta_seg = 5.0
    # Default phases [0, pi, 0, pi] will be used

    try:
        bir4_time, bir4_amp_cplx, bir4_freq = generate_bir4_waveforms(
            segment_duration_s=bir4_seg_dur,
            num_timepoints_segment=bir4_n_points_seg,
            peak_b1_tesla_segment=bir4_b1_max_seg,
            bandwidth_hz_segment=bir4_bw_seg,
            beta_segment=bir4_beta_seg,
            device='cpu'
        )
        total_bir4_points = 4 * bir4_n_points_seg
        print(f"Generated BIR-4 waveforms with {bir4_time.shape[0]} total points (expected {total_bir4_points}).")
        print(f"BIR-4 Time vector: min={bir4_time.min().item():.4f} s, max={bir4_time.max().item():.4f} s, total_dur={bir4_time[-1] - bir4_time[0] + bir4_seg_dur/bir4_n_points_seg :.4f}s")
        print(f"BIR-4 Amplitude vector (complex Tesla): peak_abs={torch.max(torch.abs(bir4_amp_cplx)).item():.2e}")
        # Check phase by looking at a point in each segment (e.g., midpoint)
        mid_point_seg0 = torch.angle(bir4_amp_cplx[bir4_n_points_seg//2]).item() / torch.pi
        mid_point_seg1 = torch.angle(bir4_amp_cplx[bir4_n_points_seg + bir4_n_points_seg//2]).item() / torch.pi
        print(f"Phase check (as multiples of pi): Seg0 ~{mid_point_seg0:.2f}pi, Seg1 ~{mid_point_seg1:.2f}pi")
        print(f"BIR-4 Frequency offset vector (Hz): min={bir4_freq.min().item():.2f}, max={bir4_freq.max().item():.2f}")

        # Example of using default phases explicitly
        custom_phases = [0, torch.pi, torch.pi, 0] # Example variant
        bir4_time_custom, _, _ = generate_bir4_waveforms(
            segment_duration_s=bir4_seg_dur,
            num_timepoints_segment=bir4_n_points_seg,
            peak_b1_tesla_segment=bir4_b1_max_seg,
            bandwidth_hz_segment=bir4_bw_seg,
            beta_segment=bir4_beta_seg,
            segment_phases_rad=custom_phases,
            device='cpu'
        )
        print(f"Generated BIR-4 with custom phases {custom_phases} successfully.")


        # To visualize BIR-4 (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig_bir4, axs_bir4 = plt.subplots(3, 1, sharex=True)
        # axs_bir4[0].plot(bir4_time.numpy(), torch.abs(bir4_amp_cplx).numpy())
        # axs_bir4[0].set_title('BIR-4 Amplitude |B1(t)| (Tesla)')
        # axs_bir4[1].plot(bir4_time.numpy(), torch.angle(bir4_amp_cplx).numpy() / torch.pi)
        # axs_bir4[1].set_title('BIR-4 Phase arg(B1(t)) (multiples of pi)')
        # axs_bir4[2].plot(bir4_time.numpy(), bir4_freq.numpy())
        # axs_bir4[2].set_title('BIR-4 Frequency Modulation (Hz)')
        # axs_bir4[2].set_xlabel('Time (s)')
        # plt.tight_layout()
        # plt.show()

    except ValueError as e:
        print(f"Error during BIR-4 example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during BIR-4 example: {e}")

    print("\n--- BIR-4 Simulation Example (using bloch_simulate) ---")
    bir4_dt_s = bir4_time[1].item() - bir4_time[0].item()
    T_TO_G = 10000.0
    bir4_amp_cplx_gauss = bir4_amp_cplx * T_TO_G

    b1_scales_sim = [0.7, 1.0, 1.3]
    print(f"Simulating BIR-4 inversion for B1 scales: {b1_scales_sim}")
    for scale in b1_scales_sim:
        bir4_fm_phase_rad = torch.cumsum(bir4_freq * bir4_dt_s, dim=0) * 2 * torch.pi
        final_bir4_rf_for_sim_gauss = bir4_amp_cplx_gauss * torch.exp(1j * bir4_fm_phase_rad.to(bir4_amp_cplx_gauss.device))
        scaled_rf_gauss = final_bir4_rf_for_sim_gauss * scale

        M_final_bir4 = bloch_simulate(
            rf=scaled_rf_gauss.cpu().numpy(),
            grad=np.zeros_like(scaled_rf_gauss.cpu().numpy().real),
            dt=bir4_dt_s,
            gamma=GAMMA_HZ_PER_G_PROTON,
            b0=0.0,
            mx0=0.0, my0=0.0, mz0=1.0,
            return_all=False
        )
        final_mz_bir4 = M_final_bir4[0,0,2]
        print(f"  B1 scale {scale:.1f}: Final Mz = {final_mz_bir4:.4f}")


def generate_wurst_waveforms(
    pulse_duration_s: float,
    num_timepoints: int,
    peak_b1_tesla: float,
    bandwidth_hz: float,
    power_n: int = 20,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates amplitude and frequency modulation waveforms for a WURST
    (Wideband, Uniform Rate, Smooth Truncation) adiabatic pulse.

    The pulse is defined over a symmetric time interval tau = [-1, 1],
    which corresponds to t = [-pulse_duration_s/2, pulse_duration_s/2].

    Amplitude Modulation A(t): A_max * (1 - |tau|^n)
    Frequency Modulation f(t): (bandwidth_hz / 2) * tau
                               (sweep from -bandwidth/2 to +bandwidth/2)

    Args:
        pulse_duration_s (float): Total duration of the pulse (T) in seconds.
        num_timepoints (int): Number of time points to sample the waveform.
        peak_b1_tesla (float): Peak B1 amplitude (A_max) of the pulse in Tesla.
                               This is the amplitude at the center of the pulse (tau=0).
        bandwidth_hz (float): Full bandwidth of the linear frequency sweep in Hz.
        power_n (int, optional): Exponent 'n' controlling the smoothness of the
                                 amplitude modulation. Defaults to 20.
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - time_s (torch.Tensor): Time vector in seconds, shape (num_timepoints,).
            - amplitude_tesla (torch.Tensor): RF amplitude waveform in Tesla (real-valued),
                                             shape (num_timepoints,).
            - freq_offset_hz (torch.Tensor): RF frequency offset waveform in Hz,
                                            shape (num_timepoints,).
    """
    if pulse_duration_s <= 0:
        raise ValueError("Pulse duration must be positive.")
    if num_timepoints <= 1:
        raise ValueError("Number of timepoints must be greater than 1.")
    if peak_b1_tesla < 0:
        raise ValueError("Peak B1 amplitude cannot be negative.")
    if bandwidth_hz < 0: # Bandwidth can be zero for a non-swept pulse, but typically positive
        raise ValueError("Bandwidth cannot be negative for a sweep.")
    if power_n <= 0:
        raise ValueError("Power n must be a positive integer.")

    dev = torch.device(device)
    dtype_real = torch.float32 # Standard for real-valued waveforms

    # Time vector from -pulse_duration_s/2 to +pulse_duration_s/2
    time_s = torch.linspace(-pulse_duration_s / 2.0, pulse_duration_s / 2.0, num_timepoints, device=dev, dtype=dtype_real)

    # Normalized time tau from -1 to 1
    tau = 2.0 * time_s / pulse_duration_s
    # Ensure tau is exactly within [-1, 1] to avoid issues with power_n for abs(tau)>1 if N is large
    tau = torch.clamp(tau, -1.0, 1.0)


    # Amplitude Modulation: A(t) = A_max * (1 - |tau|^n)
    amplitude_tesla = peak_b1_tesla * (1.0 - torch.abs(tau)**power_n)

    # Frequency Modulation: f(t) = (bandwidth_hz / 2) * tau
    freq_offset_hz = (bandwidth_hz / 2.0) * tau

    return time_s, amplitude_tesla, freq_offset_hz

if __name__ == '__main__':
    print("Adiabatic HS pulse waveform generator defined.")

    duration = 10e-3
    n_points = 1024
    b1_max = 15e-6
    bw_hz = 4000
    beta_val = 5.0

    try:
        time_vec, amp_vec, freq_vec = generate_hs_pulse_waveforms(
            pulse_duration_s=duration,
            num_timepoints=n_points,
            peak_b1_tesla=b1_max,
            bandwidth_hz=bw_hz,
            beta=beta_val,
            device='cpu'
        )

        print(f"Generated waveforms with {len(time_vec)} points.")
        print(f"Time vector: min={time_vec.min().item():.4f} s, max={time_vec.max().item():.4f} s")
        print(f"Amplitude vector (Tesla): min={amp_vec.min().item():.2e}, max={amp_vec.max().item():.2e}")
        print(f"Frequency offset vector (Hz): min={freq_vec.min().item():.2f}, max={freq_vec.max().item():.2f}")

        # To visualize (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 1, sharex=True)
        # axs[0].plot(time_vec.numpy(), amp_vec.numpy())
        # axs[0].set_title('Amplitude Modulation (Tesla)')
        # axs[1].plot(time_vec.numpy(), freq_vec.numpy())
        # axs[1].set_title('Frequency Modulation (Hz)')
        # axs[1].set_xlabel('Time (s)')
        # plt.show()

    except ValueError as e:
        print(f"Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- BIR-4 Pulse Waveform Generation Example ---")
    bir4_seg_dur = 2.5e-3  # 2.5 ms per segment
    bir4_n_points_seg = 256
    bir4_b1_max_seg = 15e-6
    bir4_bw_seg = 4000
    bir4_beta_seg = 5.0
    # Default phases [0, pi, 0, pi] will be used

    try:
        bir4_time, bir4_amp_cplx, bir4_freq = generate_bir4_waveforms(
            segment_duration_s=bir4_seg_dur,
            num_timepoints_segment=bir4_n_points_seg,
            peak_b1_tesla_segment=bir4_b1_max_seg,
            bandwidth_hz_segment=bir4_bw_seg,
            beta_segment=bir4_beta_seg,
            device='cpu'
        )
        total_bir4_points = 4 * bir4_n_points_seg
        print(f"Generated BIR-4 waveforms with {bir4_time.shape[0]} total points (expected {total_bir4_points}).")
        print(f"BIR-4 Time vector: min={bir4_time.min().item():.4f} s, max={bir4_time.max().item():.4f} s, total_dur={bir4_time[-1] - bir4_time[0] + bir4_seg_dur/bir4_n_points_seg :.4f}s")
        print(f"BIR-4 Amplitude vector (complex Tesla): peak_abs={torch.max(torch.abs(bir4_amp_cplx)).item():.2e}")
        # Check phase by looking at a point in each segment (e.g., midpoint)
        mid_point_seg0 = torch.angle(bir4_amp_cplx[bir4_n_points_seg//2]).item() / torch.pi
        mid_point_seg1 = torch.angle(bir4_amp_cplx[bir4_n_points_seg + bir4_n_points_seg//2]).item() / torch.pi
        print(f"Phase check (as multiples of pi): Seg0 ~{mid_point_seg0:.2f}pi, Seg1 ~{mid_point_seg1:.2f}pi")
        print(f"BIR-4 Frequency offset vector (Hz): min={bir4_freq.min().item():.2f}, max={bir4_freq.max().item():.2f}")

        # Example of using default phases explicitly
        custom_phases = [0, torch.pi, torch.pi, 0] # Example variant
        bir4_time_custom, _, _ = generate_bir4_waveforms(
            segment_duration_s=bir4_seg_dur,
            num_timepoints_segment=bir4_n_points_seg,
            peak_b1_tesla_segment=bir4_b1_max_seg,
            bandwidth_hz_segment=bir4_bw_seg,
            beta_segment=bir4_beta_seg,
            segment_phases_rad=custom_phases,
            device='cpu'
        )
        print(f"Generated BIR-4 with custom phases {custom_phases} successfully.")


        # To visualize BIR-4 (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig_bir4, axs_bir4 = plt.subplots(3, 1, sharex=True)
        # axs_bir4[0].plot(bir4_time.numpy(), torch.abs(bir4_amp_cplx).numpy())
        # axs_bir4[0].set_title('BIR-4 Amplitude |B1(t)| (Tesla)')
        # axs_bir4[1].plot(bir4_time.numpy(), torch.angle(bir4_amp_cplx).numpy() / torch.pi)
        # axs_bir4[1].set_title('BIR-4 Phase arg(B1(t)) (multiples of pi)')
        # axs_bir4[2].plot(bir4_time.numpy(), bir4_freq.numpy())
        # axs_bir4[2].set_title('BIR-4 Frequency Modulation (Hz)')
        # axs_bir4[2].set_xlabel('Time (s)')
        # plt.tight_layout()
        # plt.show()

    except ValueError as e:
        print(f"Error during BIR-4 example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during BIR-4 example: {e}")

    print("\n--- WURST Pulse Waveform Generation Example ---")
    wurst_dur = 8e-3  # 8 ms
    wurst_n_points = 512
    wurst_b1_max = 12e-6 # 12 uT
    wurst_bw = 5000      # 5 kHz
    wurst_n_power = 40

    try:
        wurst_time, wurst_amp, wurst_freq = generate_wurst_waveforms(
            pulse_duration_s=wurst_dur,
            num_timepoints=wurst_n_points,
            peak_b1_tesla=wurst_b1_max,
            bandwidth_hz=wurst_bw,
            power_n=wurst_n_power,
            device='cpu'
        )

        print(f"Generated WURST waveforms with {wurst_time.shape[0]} points.")
        print(f"WURST Time vector: min={wurst_time.min().item():.4f} s, max={wurst_time.max().item():.4f} s")
        print(f"WURST Amplitude vector (Tesla): min={wurst_amp.min().item():.2e} (expected ~0), max={wurst_amp.max().item():.2e} (expected ~{wurst_b1_max:.2e})")
        print(f"WURST Frequency offset vector (Hz): min={wurst_freq.min().item():.2f} (expected ~{-wurst_bw/2:.2f}), max={wurst_freq.max().item():.2f} (expected ~{wurst_bw/2:.2f})")

        # To visualize WURST (requires matplotlib):
        # import matplotlib.pyplot as plt
        # fig_wurst, axs_wurst = plt.subplots(2, 1, sharex=True)
        # axs_wurst[0].plot(wurst_time.numpy(), wurst_amp.numpy())
        # axs_wurst[0].set_title('WURST Amplitude |B1(t)| (Tesla)')
        # axs_wurst[1].plot(wurst_time.numpy(), wurst_freq.numpy())
        # axs_wurst[1].set_title('WURST Frequency Modulation (Hz)')
        # axs_wurst[1].set_xlabel('Time (s)')
        # plt.tight_layout()
        # plt.show()

    except ValueError as e:
        print(f"Error during WURST example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during WURST example: {e}")

    print("\n--- WURST Simulation Example (using bloch_simulate) ---")
    wurst_dt_s = wurst_time[1].item() - wurst_time[0].item()
    wurst_amp_gauss = wurst_amp * T_TO_G

    wurst_phase_rad = torch.cumsum(wurst_freq * wurst_dt_s, dim=0) * 2 * torch.pi
    wurst_rf_complex_gauss = wurst_amp_gauss * torch.exp(1j * wurst_phase_rad.to(wurst_amp_gauss.device))

    print(f"Simulating WURST inversion for B1 scales: {b1_scales_sim}")
    for scale in b1_scales_sim:
        scaled_rf_wurst_gauss = wurst_rf_complex_gauss * scale
        M_final_wurst = bloch_simulate(
            rf=scaled_rf_wurst_gauss.cpu().numpy(),
            grad=np.zeros_like(scaled_rf_wurst_gauss.cpu().numpy().real),
            dt=wurst_dt_s,
            gamma=GAMMA_HZ_PER_G_PROTON,
            b0=0.0,
            mx0=0.0, my0=0.0, mz0=1.0,
            return_all=False
        )
        final_mz_wurst = M_final_wurst[0,0,2]
        print(f"  B1 scale {scale:.1f}: Final Mz = {final_mz_wurst:.4f}")
