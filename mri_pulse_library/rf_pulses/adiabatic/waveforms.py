"""Generation of common adiabatic RF pulse waveforms."""
import torch
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
