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

## Advanced Features

- **Simulation:**  
  Bloch simulator and small-tip approximation for validating pulse profiles and spectral selectivity
- **Analysis and Visualization:**  
  Frequency/spatial response, k-space trajectory plots, passband/stopband metrics
- **VERSE:**  
  Apply variable-rate selective excitation for SAR or hardware limits
- **Multi-band/Multi-slice:**  
  Scripted combination of bands or slices for parallel or simultaneous excitation
- **Exporters:**  
  Save pulses for Siemens, GE, Philips, Pulseq, and research toolchains

---

## Interoperability

- **MATLAB/NumPy:**  
  Load and save `.mat` files, convert between PyTorch tensors and NumPy arrays

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
