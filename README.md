# spectralspatial

A **PyTorch-based toolbox for designing spectral-spatial (SPSP) RF pulses** for MRI, supporting a wide range of excitation, refocusing, and spectral-selective applications.  
This package enables rapid and customizable design, simulation, and export of SPSP pulses for both research and deployment on clinical and preclinical MRI systems.

---

## Features

- **Flexible Spectral-Spatial Pulse Design**:  
  - Excitation, refocusing, inversion, and saturation pulses  
  - Control over slice thickness, time-bandwidth, flip angle, pass/stop ripples, spectral bands, and more
- **Multiple Filter Types**:  
  Parks-McClellan (PM), Least Squares (LS), Sinc, SLR, and custom
- **Gradient Trajectory Support**:  
  Flyback, Echo-Planar (EP), and more
- **Multi-band and Multi-slice Support**
- **VERSE Optimization**:  
  Variable Rate Selective Excitation tools for SAR reduction and pulse shortening
- **Simulation & Analysis**:  
  Bloch simulation, frequency/spatial response, k-space visualization, SAR and RF power estimation
- **Export**:  
  Save designed pulses in Siemens `.pta`, GE `.mod`, Philips `.rf`, Pulseq `.txt` formats for scanner deployment
- **MATLAB/NumPy Interoperability**:  
  Load and save `.mat` files, work seamlessly with PyTorch tensors and NumPy arrays

---

## Supported RF Pulse Types

This library aims to support a comprehensive suite of RF pulses for MRI. Current support and future development will include:

*   **Hard Pulses:** Simple rectangular pulses for basic excitation.
*   **Slice-Selective Pulses:** Shaped pulses (e.g., sinc, Gaussian) for exciting specific anatomical slices. *Currently supported via SPSP designer for shaped pulses.*
*   **Spectral-Spatial Pulses:** For simultaneous spatial and spectral selection. *Core feature of this library.*
*   **Adiabatic Pulses:** Pulses robust to B1 inhomogeneities (e.g., BIR-4, Hyperbolic Secant, WURST, GOIA-WURST). *Placeholder implementations available (see 'Additional Pulse Types Implemented' section below).*
*   **Composite Pulses:** Series of sub-pulses for robust flip angles and phase response. *Placeholder implementation available (see 'Additional Pulse Types Implemented' section below).*
*   **Multiband Pulses:** Excite multiple slices simultaneously. *Supported via scripted combination and planned for dedicated design tools.*
*   **Parallel Transmit (pTx) Pulses:** For B1 shimming and advanced excitation using multiple transmit coils. *Planned for future implementation.*
*   **Small-Tip Angle Pulses:** For fast gradient echo sequences. *Supported via SPSP designer and Bloch simulation.*

---

## Additional Pulse Types Implemented

This library now includes placeholder implementations for several advanced RF pulse types:

### Adiabatic Pulses
The following adiabatic pulses, known for their robustness to B1 inhomogeneities, have been added with basic placeholder structures:
- **BIR-4 (B1-Insensitive Rotation):** Implemented in `mri_pulse_library.rf_pulses.adiabatic.bir4_pulse.generate_bir4_pulse()`.
- **WURST (Wideband, Uniform Rate, Smooth Truncation):** Implemented in `mri_pulse_library.rf_pulses.adiabatic.wurst_pulse.generate_wurst_pulse()`.
- **GOIA-WURST (Gradient Offset Independent Adiabaticity):** Implemented in `mri_pulse_library.rf_pulses.adiabatic.goia_wurst_pulse.generate_goia_wurst_pulse()`. This pulse type also returns a placeholder gradient waveform.

### Composite Pulses
A generic function for generating composite pulses from a sequence of sub-pulses (currently modeled as hard pulses) has been implemented:
- **Generic Composite Pulse Sequence:** Implemented in `mri_pulse_library.rf_pulses.composite.composite_pulse.generate_composite_pulse_sequence()`. This allows for the construction of various composite pulses by defining the properties of their constituent sub-pulses.

These implementations serve as a foundation for future development and more detailed characterization.

---

## Installation

```bash
pip install torch numpy matplotlib scipy
git clone https://github.com/kaggie/spectralspatial.git
cd spectralspatial
```

---

## Quick Start

### 1. Design a Spectral-Spatial Pulse

```python
from spectral_spatial_designer import SpectralSpatialPulseDesigner

designer = SpectralSpatialPulseDesigner(
    dt=4e-6,                  # Time step (s)
    gamma_hz_g=4257.0,        # Gyromagnetic ratio (Hz/G)
    max_grad_g_cm=5.0,        # Max gradient (G/cm)
    max_slew_g_cm_ms=20.0,    # Max slew rate (G/cm/ms)
    max_b1_g=0.15,            # Max B1 amplitude (Gauss)
    max_duration_s=20e-3,     # Max pulse duration (s)
    device='cpu'              # or 'cuda'
)

pulse = designer.design_pulse(
    spatial_thk_cm=0.5,               
    spatial_tbw=4,                    
    spatial_ripple_pass=0.01,         
    spatial_ripple_stop=0.01,         
    spectral_freq_bands_hz=[-500, 0, 0, 500], 
    spectral_amplitudes=[1, 0],       
    spectral_ripples=[0.01, 0.01],    
    nominal_flip_angle_rad=1.57,      
    pulse_type='ex',                  
    spatial_filter_type='pm',         
    spectral_filter_type='pm',        
    ss_type='Flyback Whole',          
    use_slr=False                     
)
```

### 2. Example Use Cases

#### Water-Fat Separation

```python
pulse = designer.design_pulse(
    spatial_thk_cm=0.5,
    spatial_tbw=4,
    spatial_ripple_pass=0.01,
    spatial_ripple_stop=0.01,
    spectral_freq_bands_hz=[-500, -420, -60, 60],  # Fat stop, water pass
    spectral_amplitudes=[0, 1],
    spectral_ripples=[0.01, 0.01],
    nominal_flip_angle_rad=1.57,
    pulse_type='ex',
    spatial_filter_type='pm',
    spectral_filter_type='pm',
    ss_type='Flyback Whole',
    use_slr=False
)
```

#### Hyperpolarized 13C Spectral-Spatial Pulse

```python
pulse = designer.design_pulse(
    spatial_thk_cm=1.0,
    spatial_tbw=6,
    spatial_ripple_pass=0.02,
    spatial_ripple_stop=0.02,
    spectral_freq_bands_hz=[-300, -200, 200, 300],
    spectral_amplitudes=[1, 0],
    spectral_ripples=[0.02, 0.02],
    nominal_flip_angle_rad=0.35,
    pulse_type='ex',
    spatial_filter_type='pm',
    spectral_filter_type='pm',
    ss_type='EP Whole',
    use_slr=False
)
```

---

## Output

- `rf_G`: Complex RF waveform (Gauss)
- `grad_G_cm`: Gradient waveform (G/cm)
- `fs_hz_design`: Design spectral sampling frequency (Hz)
- `adjusted_flip_angle_rad`: Actual flip angle used
- `total_duration_designed_s`: Pulse duration (s)
- ... and more

---

## Advanced Simulator Capabilities

The `spectralspatial` toolbox aims to provide a comprehensive environment for designing, analyzing, and optimizing RF pulses. Key capabilities include:

### Core Simulation & Design
- **Bloch Equation Solvers:**
    - 1D, 2D, and 3D Bloch simulations for magnetization evolution.
    - Support for Bloch-McConnell equations for multi-pool models (e.g., fat-water, CEST, MT) is planned.
    - Numerical solvers (e.g., Runge-Kutta) for complex pulse shapes.
- **Pulse Design Algorithms:**
    - Shinnar-LeRoux (SLR) Algorithm. *Partially supported.*
    - Optimal Control Theory (OCT) for constrained pulse design. *Future development.*
    - Adiabatic pulse design tools. *Future development.*
    - Composite pulse design algorithms. *Future development.*
    - Multiband pulse design tools. *Supported via scripted combination and planned for dedicated design tools.*
    - Parallel Transmit (pTx) pulse design. *Future development.*
    - Small-tip angle approximation for fast simulations. *Supported.*
- **Flexible Spectral-Spatial Pulse Design**:
    - Excitation, refocusing, inversion, and saturation pulses.
    - Control over slice thickness, time-bandwidth, flip angle, pass/stop ripples, spectral bands.
- **Multiple Filter Types**: Parks-McClellan (PM), Least Squares (LS), Sinc, SLR, and custom.
- **Gradient Trajectory Support**: Flyback, Echo-Planar (EP), and more.
- **VERSE Optimization**: Variable Rate Selective Excitation tools for SAR reduction and pulse shortening. *Supported.*

### Realistic Environment Modeling
- **Field Inhomogeneity Modeling:**
    - B0 inhomogeneity (static field imperfections). *Simulation supports B0 offsets.*
    - B1 inhomogeneity (transmit RF field variations). *Planned for advanced simulations.*
- **Gradient Field Modeling:** Representation of gradient fields, including imperfections (future).
- **RF Coil Modeling:** Incorporating B1+ transmit and B1- receive sensitivities (future).
- **Phantom and Tissue Models:**
    - User-defined phantoms with tissue properties (T1, T2, T2*, PD, chemical shift). *Supported via simulation parameters.*
    - Multi-compartment tissue models (future).
    - Motion modeling (future).

### Performance Analysis & Visualization
- **Slice Profile Analysis:** Plotting and quantifying excitation profiles (thickness, ripple, transition width, side lobes). *Supported.*
- **Inversion Profile Analysis:** Quantifying inversion efficiency. *Supported via simulation.*
- **Flip Angle Mapping:** Generating 2D/3D maps of achieved flip angles. *Supported via simulation.*
- **Phase Profile Analysis:** Examining phase distribution. *Supported via simulation.*
- **Specific Absorption Rate (SAR) Calculation:** Quantifying power deposition. *Basic estimation supported, advanced SAR planned.*
- **Off-Resonance Effects:** Analyzing pulse performance under B0 offsets. *Supported.*
- **Robustness Analysis:** Evaluating performance across B0/B1 variations (partially supported, planned for expansion).
- **Visualization Tools:**
    - Magnetization vector trajectories (future).
    - Spatial profiles (1D, 2D, 3D). *Supported.*
    - Time-domain RF/gradient waveforms. *Supported.*
    - k-space trajectories. *Supported.*
    - Heat maps/contour plots for field distributions (future).

### Integration & Usability
- **Pulse Sequence Integration:** Simulating pulses within a complete sequence context (future).
- **Parameter Optimization Tools:** For automatic pulse parameter tuning (future).
- **Input/Output:** Import/export in various formats (Siemens, GE, Philips, Pulseq). *Supported.*
- **Graphical User Interface (GUI):** For intuitive interaction (future).
- **Scripting API:** For automation and integration (Python-based). *Core feature.*
- **GPU Acceleration:** For faster simulations. *Supported via PyTorch device selection.*
- **MATLAB/NumPy Interoperability:** Load/save `.mat` files, tensor/array conversion. *Supported.*

---

## Directory Structure

```
spectralspatial/
  spectral_spatial_designer.py
  fir_designer.py
  slr_transform.py
  aliasing_calculator.py
  gradient_optimizer.py
  verse.py
  examples/
  ...
```

---
