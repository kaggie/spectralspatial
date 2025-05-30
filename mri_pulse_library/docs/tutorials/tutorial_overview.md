# MRI Pulse Library Tutorials

Welcome to the `mri_pulse_library` tutorials! These tutorials aim to guide you through
the various functionalities of the library, from basic pulse design to more advanced
sequence simulation.

## Planned Tutorials

1.  **Tutorial 1: Designing Basic RF Pulses**
    *   Generating Hard, Sinc, and Gaussian pulses.
    *   Understanding key parameters: duration, bandwidth, flip angle.
    *   Visualizing pulse waveforms (amplitude and phase).

2.  **Tutorial 2: Simulating RF Pulse Performance**
    *   Simulating slice profiles for selective pulses.
    *   Simulating frequency response for hard pulses.
    *   Understanding B0 and B1 effects (briefly).
    *   Using the `PulseValidationMetrics` to analyze profiles (FWHM, ripple).

3.  **Tutorial 3: Adiabatic Pulses**
    *   Generating an HS1 adiabatic pulse.
    *   Understanding adiabaticity and its parameters (mu, beta).
    *   Simulating B1 insensitivity of adiabatic pulses.

4.  **Tutorial 4: Spectral-Spatial (SPSP) Pulses**
    *   Designing a basic SPSP pulse.
    *   Simulating its 2D (spatial-spectral) profile.
    *   Introduction to multi-band SPSP pulse design.
    *   Simulating a 3D multi-band SPSP pulse.

5.  **Tutorial 5: Introduction to Gradient Pulse Design (Future)**
    *   Designing a simple trapezoidal slice-selection gradient.
    *   Calculating gradient moments.
    *   Matching gradients to RF pulses.

6.  **Tutorial 6: Building a Simple Sequence (Future)**
    *   Combining RF and gradient pulses.
    *   Managing timing (TR, TE).
    *   Simulating a basic GRE or SE sequence block.

7.  **Tutorial 7: Exporting Pulses (Future)**
    *   Overview of vendor adapter concepts.
    *   Example of preparing pulse data for export (conceptual).

## Getting Started

Before diving into the tutorials, ensure you have the library installed and a suitable
Python environment (e.g., with NumPy, SciPy, Matplotlib for visualization, and PyTorch for the Bloch simulator).

Each tutorial will be provided as a Jupyter notebook or a Python script with detailed explanations.

---

*We welcome contributions and suggestions for new tutorials!*
