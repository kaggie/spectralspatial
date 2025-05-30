# Safety Guidelines for MRI Pulse Design

Designing RF and gradient pulses for MRI requires careful consideration of safety limits
to protect subjects and equipment. This document outlines key safety parameters and
general guidelines. **Always consult your institution's IRB, safety officers, and scanner
manufacturer's recommendations.**

## Key Safety Parameters

1.  **Specific Absorption Rate (SAR)**:
    *   **Definition**: Measure of RF power absorbed by tissue, typically in Watts per kilogram (W/kg).
    *   **Limits**: Governed by regulatory bodies (e.g., FDA, IEC) and vary by body region (whole body, head, local). Limits often depend on averaging time.
    *   **Factors**: RF pulse amplitude, duration, duty cycle, subject size, and coil type.
    *   **Calculation**: Requires estimation of B1+rms field and tissue models. This library may provide tools for B1+rms calculation, but full SAR estimation is complex and often scanner-dependent.

2.  **Peripheral Nerve Stimulation (PNS)**:
    *   **Definition**: Sensation (tingling, twitching) caused by rapidly switching magnetic gradients inducing electric fields in the body.
    *   **Limits**: Based on `dB/dt` (rate of change of magnetic field) or electric field strength. Limits vary by gradient axis and scanner model.
    *   **Factors**: Gradient amplitude, slew rate, duration, and waveform shape. Sequences with rapid gradient switching (e.g., EPI) are of particular concern.

3.  **Acoustic Noise**:
    *   **Definition**: Loud noise produced by gradient coils vibrating due to Lorentz forces during rapid current switching.
    *   **Limits**: Occupational safety limits for noise exposure. Hearing protection is standard.
    *   **Factors**: Gradient amplitude, slew rate, and pulse sequence design.

4.  **B1+rms Limits**:
    *   **Definition**: Root mean square of the B1+ field over time, often used as a surrogate for local SAR or to stay within hardware limits of the RF amplifier and transmit coil.
    *   **Limits**: Scanner-specific, often defined per TR or over a specific time window.

## General Guidelines for Pulse Design

*   **Minimize RF Power**:
    *   Use the lowest possible flip angles that achieve the desired contrast.
    *   Optimize pulse shapes for efficiency (e.g., reduce peak B1 for same flip angle where possible).
    *   Consider SAR implications of pulse repetition (TR) and number of pulses in a sequence.
*   **Manage Gradient Activity**:
    *   Use the lowest necessary gradient amplitudes and slew rates.
    *   Optimize gradient waveforms to minimize `dB/dt` for a given imaging goal (e.g., gradient moment shaping).
    *   Be mindful of cumulative effects in long sequences.
*   **Use Vendor Tools**:
    *   Most MRI scanners have built-in safety monitoring and prediction tools. Utilize these tools when implementing sequences on actual hardware.
    *   Understand how your scanner calculates and limits SAR and PNS.
*   **Testing and Validation**:
    *   Simulate SAR and PNS characteristics of new pulses/sequences if tools are available.
    *   Phantom testing is crucial before human scans.
*   **Stay Informed**:
    *   Keep up-to-date with current safety guidelines and research (e.g., ISMRM safety committee resources).

## Disclaimer

This library provides tools for designing RF and gradient pulses. It is the **user's responsibility** to ensure that any pulses or sequences designed using this library are safe and compliant with all applicable regulations and institutional policies when used in practice. The developers of this library assume no liability for misuse or unsafe operation.File 'mri_pulse_library/docs/safety_guidelines.md' created successfully.
