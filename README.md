
## Quick Start

### 1. Design a Spectral-Spatial Pulse (Python API)

```python
from spectral_spatial_designer import SpectralSpatialPulseDesigner

designer = SpectralSpatialPulseDesigner(
    dt=4e-6,                  # Time step (s)
    gamma_hz_g=4257.0,        # Gyromagnetic ratio (Hz/G)
    max_grad_g_cm=5.0,        # Max gradient (G/cm)
    max_slew_g_cm_ms=20.0,    # Max slew rate (G/cm/ms)
    max_b1_g=0.15,            # Max B1 amplitude (Gauss)
    max_duration_s=20e-3,     # Max pulse duration (s)
    device='cpu'              # 'cuda' for GPU
)

pulse = designer.design_pulse(
    spatial_thk_cm=0.5,               # Slice thickness (cm)
    spatial_tbw=4,                    # Spatial time-bandwidth product
    spatial_ripple_pass=0.01,         # Spatial passband ripple
    spatial_ripple_stop=0.01,         # Spatial stopband ripple
    spectral_freq_bands_hz=[-500, 0, 0, 500], # Spectral bands (Hz)
    spectral_amplitudes=[1, 0],       # Desired amplitudes (e.g. water pass, fat stop)
    spectral_ripples=[0.01, 0.01],    # Spectral ripples
    nominal_flip_angle_rad=1.57,      # Flip angle (rad)
    pulse_type='ex',                  # 'ex', 'se', 'inv', or 'sat'
    spatial_filter_type='pm',         # 'pm', 'ls', or 'sinc'
    spectral_filter_type='pm',        # 'pm', 'ls', or 'sinc'
    ss_type='Flyback Whole',          # 'Flyback Whole', 'EP Whole', etc.
    use_slr=False                     # Use SLR transform or not
)

# The resulting dictionary contains:
#   pulse['rf_G']          - RF pulse (Gauss, complex PyTorch tensor)
#   pulse['grad_G_cm']     - Gradient waveform (G/cm)
#   pulse['fs_hz_design']  - Design spectral sampling frequency
#   ... and more.
```

### 2. Example: Water-Fat Separation

Design a dual-band pulse that excites water (0 Hz) and suppresses fat (-440 Hz):

```python
pulse = designer.design_pulse(
    spatial_thk_cm=0.5,
    spatial_tbw=4,
    spatial_ripple_pass=0.01,
    spatial_ripple_stop=0.01,
    spectral_freq_bands_hz=[-500, -420, -60, 60], # Fat stop, water pass
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

### 3. Example: Hyperpolarized 13C Spectral-Spatial Pulse

```python
pulse = designer.design_pulse(
    spatial_thk_cm=1.0,
    spatial_tbw=6,
    spatial_ripple_pass=0.02,
    spatial_ripple_stop=0.02,
    spectral_freq_bands_hz=[-300, -200, 200, 300], # Custom bands for 13C
    spectral_amplitudes=[1, 0],
    spectral_ripples=[0.02, 0.02],
    nominal_flip_angle_rad=0.35,  # Smaller flip for substrate preservation
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
