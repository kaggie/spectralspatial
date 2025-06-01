# mri_pulse_library/core/dsp_utils.py
import torch
import math

def spectral_factorization_cepstral(
    p_coeffs_real_symmetric: torch.Tensor,
    target_a_length: int,
    nfft: int = None
) -> torch.Tensor:
    """
    Performs spectral factorization of a real, symmetric polynomial P(z)
    to find a minimum-phase polynomial A(z) such that P(z) = A(z)A*(1/z*).
    This implementation uses the cepstral method.

    P(z) is assumed to have a non-negative frequency response P(e^jω) >= 0.
    The input p_coeffs_real_symmetric are the coefficients of P(z).

    Args:
        p_coeffs_real_symmetric (torch.Tensor): 1D tensor of real, symmetric
            coefficients of the polynomial P(z) = sum(p_k * z^-k).
            Example: For P(z) = 1 - B(z)B*(1/z*), these are [1, 0, ...] - bb_star_coeffs.
            Length of p_coeffs_real_symmetric is typically 2*L_b - 1 if B(z) has L_b coeffs.
        target_a_length (int): The desired length for the output A(z) coefficients.
                               Typically, if B(z) has L_b coeffs, A(z) also has L_b coeffs.
        nfft (int, optional): N-point FFT/IFFT to use. If None, defaults to
                              the next power of 2 >= len(p_coeffs_real_symmetric).

    Returns:
        torch.Tensor: 1D tensor of real coefficients for the minimum-phase
                      polynomial A(z), of length target_a_length.
    """
    device = p_coeffs_real_symmetric.device
    dtype = p_coeffs_real_symmetric.dtype

    if p_coeffs_real_symmetric.ndim != 1:
        raise ValueError("p_coeffs_real_symmetric must be a 1D tensor.")
    if not torch.allclose(p_coeffs_real_symmetric, p_coeffs_real_symmetric.flip(dims=[0])) and len(p_coeffs_real_symmetric) > 1:
        # This check is for symmetry of P(z)'s coefficients, which implies P(e^j_omega) is real.
        # For P(z) = 1 - B(z)B*(z^-1), P(z) is symmetric if B is real.
        # print("Warning: p_coeffs_real_symmetric may not be perfectly symmetric. P(e^j_omega) should be real.")
        pass


    if nfft is None:
        nfft = 1 << (len(p_coeffs_real_symmetric) - 1).bit_length() # Next power of 2
        if nfft < len(p_coeffs_real_symmetric): # Ensure nfft is at least len(p_coeffs)
             nfft = 1 << len(p_coeffs_real_symmetric).bit_length()


    # 1. Compute frequency response of P(z)
    # P(e^jω) must be non-negative.
    P_omega = torch.fft.fft(p_coeffs_real_symmetric, n=nfft)

    # P(e^jω) should be real due to p_coeffs symmetry. Clamp to avoid log of zero/small negative.
    P_omega_real_clamped = torch.clamp_min(P_omega.real, 1e-12)
    # If P_omega has small imaginary parts due to numerical noise, taking .real is okay.
    # Or, one could verify that imag part is indeed negligible.
    if torch.norm(P_omega.imag) / torch.norm(P_omega.real) > 1e-5 and len(p_coeffs_real_symmetric) > 1 :
         print(f"Warning: P(e^j_omega) has a significant imaginary component ({torch.norm(P_omega.imag).item()}). P_coeffs might not be perfectly symmetric.")


    # 2. Log magnitude and Cepstrum for minimum-phase factor A(z)
    # log |A(e^jω)| = 0.5 * log P(e^jω)
    log_abs_A_omega = 0.5 * torch.log(P_omega_real_clamped)

    # Cepstrum c_hat[n] = IFFT( log |A(e^jω)| ). This is real and even.
    cepstrum_log_abs_A = torch.fft.ifft(log_abs_A_omega).real

    # 3. Construct cepstrum c_min[n] for the minimum-phase factor A(z)
    # c_min[n] = c_hat[0] for n=0
    # c_min[n] = 2*c_hat[n] for 1 <= n <= N/2 -1 (positive time)
    # c_min[n] = c_hat[N/2] for n=N/2 (Nyquist, if N is even)
    # c_min[n] = 0 for n > N/2 (causal part for DFT)
    # For ifft to yield real a_coeffs, the spectrum of c_min (log A_min(e^jω)) must be conjugate symmetric.
    # This means c_min itself must be real and symmetric after this construction for DFT.
    # The cepstrum c_min[n] should be constructed to be causal for the DFT, which means it is zero for n > N/2
    # and c_min[n] for n < 0 is related to c_min[n] for n > 0 by symmetry if a_coeffs are real.
    # The standard windowing approach:
    c_min_phase_cepstrum = torch.zeros_like(cepstrum_log_abs_A)
    c_min_phase_cepstrum[0] = cepstrum_log_abs_A[0] # DC component

    if nfft % 2 == 0: # Even NFFT
        c_min_phase_cepstrum[1 : nfft//2] = 2 * cepstrum_log_abs_A[1 : nfft//2]
        c_min_phase_cepstrum[nfft//2] = cepstrum_log_abs_A[nfft//2] # Nyquist component
        # For real output from ifft(fft(c_min_phase_cepstrum_for_fft)), c_min_phase_cepstrum_for_fft must be real&even
        # The c_min_phase_cepstrum here is the true cepstrum of the min-phase sequence,
        # which is causal (not necessarily symmetric for FFT input unless sequence itself is symmetric).
        # To get log A_min(e^jω) which is conjugate symmetric, its IFFT (the cepstrum) should be real.
        # The c_min_phase_cepstrum as defined here is for the DFT of a causal sequence.
        # For FFT, we need to ensure the input results in conjugate symmetric spectrum if we want real time-domain.
        # The cepstrum of a real sequence is real.
        # The current c_min_phase_cepstrum is correct for FFT to get log(A_min).
    else: # Odd NFFT
        c_min_phase_cepstrum[1 : (nfft+1)//2] = 2 * cepstrum_log_abs_A[1 : (nfft+1)//2]
    # Components for n > N/2 (negative times for DFT) are zero.
    # If we need to make c_min_phase_cepstrum symmetric for FFT input to get a real log A_omega,
    # then we must mirror: c_min_phase_cepstrum[nfft//2+1:] = c_min_phase_cepstrum[1:nfft//2].flip(dims=[0])
    # However, the standard formulation is that c_min as constructed is the input to FFT.

    # 4. Recover A(z)
    # log A_min(e^jω) = FFT(c_min[n])
    log_A_omega_min_phase = torch.fft.fft(c_min_phase_cepstrum)

    # A_min(e^jω) = exp(log A_min(e^jω))
    A_omega_min_phase = torch.exp(log_A_omega_min_phase)

    # a_min[n] = IFFT(A_min(e^jω))
    a_coeffs_full = torch.fft.ifft(A_omega_min_phase).real # Result should be real

    # Truncate to target_a_length
    a_coeffs = a_coeffs_full[:target_a_length]

    # Normalize a_coeffs[0] to be positive if needed (convention for uniqueness)
    if a_coeffs[0] < 0:
        a_coeffs = -a_coeffs

    return a_coeffs.to(dtype=dtype) # Ensure original dtype

if __name__ == '__main__':
    print("DSP Utilities defined.")
    # Example for spectral_factorization_cepstral
    # Needs scipy for some examples, so we'll keep it minimal or use torch-only methods if possible
    try:
        # Example from a known min-phase A(z) -> P(z) -> factorize back to A(z)
        a_known = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float32) # Min-phase A(z)
        L_a_known = len(a_known)

        a_known_rev = a_known.flip(dims=[0])
        p_coeffs_from_a = torch.nn.functional.conv1d(a_known.view(1,1,-1), a_known_rev.view(1,1,-1), padding='full').squeeze()

        print(f"Known A(z) coeffs: {a_known}")
        print(f"P(z) coeffs from A(z)A*(1/z*): {p_coeffs_from_a}")

        a_factorized = spectral_factorization_cepstral(p_coeffs_from_a, target_a_length=L_a_known, nfft=1024)
        print(f"Factorized A(z) coeffs: {a_factorized}")
        print(f"Sum of abs diff: {torch.sum(torch.abs(a_known - a_factorized)).item()}")

        # Test with P(z) = 1 - B(z)B*(1/z*)
        # Create a simple B(z)
        # b_test = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32) # L_b = 3
        # Using a filter that might not be max-phase for B to make P potentially interesting
        temp_b_coeffs = torch.tensor([0.2, 0.5, -0.3, 0.1], dtype=torch.float32)

        # Ensure max |B(e^jomega)| <= 1 for P(e^jomega) >=0 for P(z) = 1 - |B(z)|^2
        B_omega_test = torch.fft.fft(temp_b_coeffs, n=1024)
        max_abs_B_omega = torch.max(torch.abs(B_omega_test))

        b_test = temp_b_coeffs
        if max_abs_B_omega > 1.0: # Normalize only if max magnitude is > 1
            b_test = temp_b_coeffs / (max_abs_B_omega + 1e-6) # Add epsilon for safety
            print(f"Normalized b_test coeffs: {b_test} (max |B(omega)| was {max_abs_B_omega.item()})")
        else:
            print(f"Original b_test coeffs: {b_test} (max |B(omega)| is {max_abs_B_omega.item()})")


        L_b_test = len(b_test)
        b_test_rev = b_test.flip(dims=[0])
        bb_star_coeffs = torch.nn.functional.conv1d(b_test.view(1,1,-1), b_test_rev.view(1,1,-1), padding='full').squeeze()

        # P(z) = delta(n) - BB*(z)
        # Length of bb_star_coeffs is 2*L_b_test - 1. Center is at index L_b_test - 1.
        p_coeffs_1_minus_bb = torch.zeros_like(bb_star_coeffs)
        p_coeffs_1_minus_bb[L_b_test - 1] = 1.0 # delta function at center for P(z)=1
        p_coeffs_1_minus_bb -= bb_star_coeffs

        print(f"\nTest with P(z) = 1 - B(z)B*(1/z*) for B(z) = {b_test}")
        print(f"BB*(z) coeffs: {bb_star_coeffs}")
        print(f"P(z) = delta - BB*(z) coeffs: {p_coeffs_1_minus_bb}")

        # Ensure P(omega) is non-negative before factorization
        P_omega_check = torch.fft.fft(p_coeffs_1_minus_bb, n=1024).real
        if torch.min(P_omega_check) < -1e-6: # Allow small numerical noise
            print(f"Warning: P(e^j_omega) for 1-|B|^2 is negative ({torch.min(P_omega_check).item()}), spectral factorization may fail or be inaccurate.")
            # This happens if |B(e^jomega)| > 1 at any frequency.

        a_factorized_from_b = spectral_factorization_cepstral(p_coeffs_1_minus_bb, target_a_length=L_b_test, nfft=1024)
        print(f"Factorized A(z) from P(1-|B|^2): {a_factorized_from_b}")

        # Verification: A(z)A*(1/z*) should approx equal P(z)
        a_fact_rev = a_factorized_from_b.flip(dims=[0])
        p_reconstructed = torch.nn.functional.conv1d(a_factorized_from_b.view(1,1,-1), a_fact_rev.view(1,1,-1), padding='full').squeeze()
        print(f"Reconstructed P(z) from factorized A(z): {p_reconstructed}")
        # Compare up to common length
        len_compare = min(len(p_coeffs_1_minus_bb), len(p_reconstructed))
        diff_p = torch.sum(torch.abs(p_coeffs_1_minus_bb[:len_compare] - p_reconstructed[:len_compare])).item()
        print(f"Sum of abs diff for P(z) (first {len_compare} coeffs): {diff_p}")

    except ImportError:
        print("Skipping some advanced __main__ examples in dsp_utils.py as scipy.signal is not available.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred during spectral factorization example: {e}")
