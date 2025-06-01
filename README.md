# MRI Pulse Design Library (spectralspatial Core)

Welcome to the **MRI Pulse Design Library**, a versatile Python-based toolkit engineered for the design, simulation, and analysis of a wide array of Radiofrequency (RF) pulses used in Magnetic Resonance Imaging (MRI). While featuring a powerful core for **spectral-spatial (SPSP) RF pulse design**, this library is expanding to provide a comprehensive suite for various MRI applications, from research to clinical deployment.

Built on PyTorch, this library leverages GPU acceleration for computationally intensive tasks and offers seamless interoperability with NumPy and MATLAB.

---

## Key Capabilities

This library offers a growing set of tools for MRI pulse design:

-   **Advanced Spectral-Spatial (SPSP) Pulse Design**:
    -   Sophisticated tools for creating excitation, refocusing, inversion, and saturation SPSP pulses.
    -   Fine-grained control over slice thickness, time-bandwidth products, flip angles, passband/stopband ripples, spectral band selection, and more.
    -   Supports multiple filter design methods (Parks-McClellan, Least Squares, Sinc, SLR) and gradient trajectories (Flyback, Echo-Planar).
-   **General RF Pulse Generation Modules**:
    -   Functions for generating various fundamental and advanced RF pulses.
    -   Modules for simple pulses (Hard, Sinc, Gaussian), detailed Adiabatic pulses, and flexible Composite pulses.
-   **VERSE Optimization**: Integrated Variable Rate Selective Excitation (VERSE) tools for SAR reduction and pulse shortening in SPSP designs.
-   **Simulation & Analysis**:
    -   Bloch equation solvers for simulating magnetization dynamics.
    -   Tools for analyzing frequency/spatial responses, k-space trajectories, and estimating SAR/RF power.
-   **Export & Interoperability**:
    -   Save designed pulses in common scanner formats (Siemens `.pta`, GE `.mod`, Philips `.rf`, Pulseq `.txt`).
    -   Load/save `.mat` files and convert between PyTorch tensors and NumPy arrays.

---

## RF Pulse Toolkit Overview

The library provides functions to generate a diverse range of RF pulses:

*   **Simple Pulses**:
    *   `generate_hard_pulse()`: Basic rectangular pulses.
    *   `generate_sinc_pulse()`: Sinc-modulated pulses for slice selection.
    *   `generate_gaussian_pulse()`: Gaussian-shaped pulses.
*   **Spectral-Spatial (SPSP) Pulses**: The core `SpectralSpatialPulseDesigner` class enables tailored design of pulses with simultaneous spatial and spectral selectivity.
*   **Adiabatic Pulses**: Robust pulses designed to be insensitive to B1 RF field inhomogeneities. Detailed implementations include:
    *   `generate_bir4_pulse()`: B1-Insensitive Rotation pulse (BIR-4).
    *   `generate_hs_pulse()`: Hyperbolic Secant pulse (HS1 and variants).
    *   `generate_wurst_pulse()`: Wideband, Uniform Rate, Smooth Truncation pulse.
    *   `generate_goia_wurst_pulse()`: Gradient Offset Independent Adiabaticity WURST pulse, including synchronized gradient waveform generation.
*   **Composite Pulses**: Sequences of sub-pulses designed for improved flip angle accuracy and phase response.
    *   `generate_composite_pulse_sequence()`: A generic function to create composite pulses from 'hard', 'sinc', or 'gaussian' sub-pulses, with support for inter-pulse delays.
    *   `generate_refocusing_90x_180y_90x()`: A specific implementation of the 90x-180y-90x refocusing pulse.
*   **Multiband Pulses**:
    *   `MultibandPulseDesigner`: A dedicated tool in `mri_pulse_library.rf_pulses.multiband.designer` to generate simultaneous multiband pulses by phase/frequency modulation of a user-provided baseband pulse. Supports peak B1 control for the combined waveform.
    *   Scripted combination of SPSP designs also remains possible for custom/advanced approaches.
*   **Parallel Transmit (pTx) Pulses**: Tools for leveraging multiple transmit coils to improve B1+ field homogeneity and enable advanced spatial excitation.
    *   `calculate_static_shims()`: Calculates static complex weights per channel for B1+ homogenization in a target ROI (see `mri_pulse_library.ptx.shimming`).
    *   `STAPTxDesigner`: Designs multi-channel RF pulses for specified kT-points using the Small Tip Angle (STA) approximation (see `mri_pulse_library.ptx.sta_designer`).
    *   Advanced Dynamic Designers: The library also includes `SpokesPulseDesigner` and `UniversalPulseDesigner` (in `mri_pulse_library.simulators`) which are optimization-based tools for designing dynamic pTx pulses using full Bloch simulations.

*   **Future Directions**: Development plans include dedicated tools for advanced small-tip angle pulse sequences.

---

## Installation

To get started, ensure you have PyTorch, NumPy, Matplotlib, and SciPy installed:
```bash
pip install torch numpy matplotlib scipy
```
Then, clone the repository:
```bash
git clone https://github.com/kaggie/spectralspatial.git
cd spectralspatial
```
(Note: The pip install command is for dependencies; this library itself is used by running scripts from the cloned repository.)

---

## Quick Start: Spectral-Spatial Pulse Design

Design a tailored SPSP pulse using the `SpectralSpatialPulseDesigner`:

```python
from spectral_spatial_designer import SpectralSpatialPulseDesigner

# Initialize the designer with system and pulse constraints
designer = SpectralSpatialPulseDesigner(
    dt=4e-6,                  # Time step (s)
    gamma_hz_g=4257.0,        # Gyromagnetic ratio (Hz/G)
    max_grad_g_cm=5.0,        # Max gradient (G/cm)
    max_slew_g_cm_ms=20.0,    # Max slew rate (G/cm/ms)
    max_b1_g=0.15,            # Max B1 amplitude (Gauss)
    max_duration_s=20e-3,     # Max pulse duration (s)
    device='cpu'              # Use 'cuda' for GPU acceleration
)

# Design an example excitation pulse
spsp_pulse_outputs = designer.design_pulse(
    spatial_thk_cm=0.5,               
    spatial_tbw=4,                    
    spatial_ripple_pass=0.01,         
    spatial_ripple_stop=0.01,         
    spectral_freq_bands_hz=[-500, 0, 0, 500], # Example: Passband at 0 Hz, stopbands elsewhere
    spectral_amplitudes=[1, 0],       
    spectral_ripples=[0.01, 0.01],    
    nominal_flip_angle_rad=1.57, # 90 degrees
    pulse_type='ex', # Excitation
    spatial_filter_type='pm', # Parks-McClellan for spatial
    spectral_filter_type='pm', # Parks-McClellan for spectral
    ss_type='Flyback Whole', # Flyback gradient trajectory
    use_slr=False                     
)

# Access designed waveforms (rf_G, grad_G_cm, etc.) from spsp_pulse_outputs
print(f"SPSP Pulse designed with duration: {spsp_pulse_outputs['total_duration_designed_s'] * 1000:.2f} ms")
```

For examples of generating adiabatic or composite pulses, please refer to the documentation within the respective modules in `mri_pulse_library.rf_pulses`.

---
## Advanced Simulation Capabilities (Overview)

The `spectralspatial` project aims to provide a comprehensive environment for RF pulse design and analysis. While the SPSP designer is a core component, the broader vision includes:

### Core Simulation & Design Tools
-   **Bloch Equation Solvers:** For 1D, 2D, and 3D magnetization dynamics. (Multi-pool support planned).
-   **Advanced Pulse Design Algorithms:** SLR (partially supported), Optimal Control (future), dedicated tools for Adiabatic/Composite/Multiband/pTx pulses (partially implemented, ongoing development).
-   **VERSE Optimization:** For SAR reduction in SPSP pulses (supported).

### Realistic Environment Modeling
-   **Field Inhomogeneities:** B0 (supported via offsets) and B1 (planned for advanced sims).
-   **Tissue Models:** User-defined properties (T1, T2, etc.) for simulation (supported).

### Performance Analysis & Visualization
-   **Profile Analysis:** Slice profiles, inversion efficiency, flip angle mapping (supported via simulation).
-   **SAR Calculation:** Basic estimation (supported), advanced methods (planned).
-   **Off-Resonance & Robustness Analysis:** Supported and planned for expansion.
-   **Visualization:** Waveforms, k-space, spatial profiles (supported).

### Integration & Usability
-   **Scanner Format Export:** Siemens, GE, Philips, Pulseq (supported).
-   **Scripting API & GPU Acceleration:** Core features.

---

## Project Structure (Illustrative)

```
spectralspatial/
|-- mri_pulse_library/        # Core library for pulse generation
|   |-- core/                 # Basic utilities, constants, Bloch sim
|   |-- rf_pulses/            # RF pulse generation modules
|   |   |-- simple/           # Hard, Sinc, Gaussian pulses
|   |   |-- adiabatic/        # BIR-4, HS, WURST, GOIA-WURST
|   |   |-- composite/        # Generic and specific composite pulses
|   |   |-- spectral_spatial/ # (Legacy or specific SPSP components if any)
|   |-- gradient_pulses/
|   |-- sequences/            # (Future: sequence building blocks)
|   |-- simulators/           # (Future: advanced simulators)
|   |-- tests/                # Unit tests
|-- spectral_spatial_designer.py # Main SPSP designer script/class
|-- fir_designer.py
|-- slr_transform.py
|-- verse.py
|-- examples/                 # Example scripts and notebooks
|-- docs/                     # Documentation
|   |-- wiki/                 # Detailed wiki pages
|-- README.md
...
```

---

This library is under active development. Contributions and feedback are welcome!
