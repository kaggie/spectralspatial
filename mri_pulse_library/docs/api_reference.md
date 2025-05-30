# API Reference

This document provides a reference for the Application Programming Interface (API)
of the `mri_pulse_library`.

## Core Modules (`mri_pulse_library.core`)

### `core.constants`
Contains physical constants, primarily gyromagnetic ratios.
- `GAMMA_HZ_PER_T_PROTON`
- `GAMMA_MHZ_PER_T_PROTON`
- `GAMMA_RAD_PER_S_PER_T_PROTON`
- `GAMMA_HZ_PER_G_PROTON`

### `core.bloch_sim`
Low-level Bloch equation simulator.
- `bloch_simulate(...)`
- `rotate_magnetization(...)`
- `apply_relaxation(...)`

<!-- Add other core modules like fourier_utils, sar_calc as they are filled -->

## RF Pulses (`mri_pulse_library.rf_pulses`)

### `rf_pulses.simple`
- `generate_hard_pulse(...)`
- `generate_sinc_pulse(...)`
- `generate_gaussian_pulse(...)`

### `rf_pulses.spectral_spatial`
- `generate_spsp_pulse(...)`
- `generate_3d_multiband_spsp(...)`

### `rf_pulses.adiabatic`
- `generate_hs1_pulse(...)`

<!-- Add composite when functions are defined -->
<!-- ### `rf_pulses.composite` -->

## Simulators (`mri_pulse_library.simulators`)

### `simulators.bloch_simulator`
User-facing wrappers for Bloch simulations.
- `simulate_hard_pulse_profile(...)`
- `simulate_slice_selective_profile(...)`
- `simulate_hard_pulse(...)`
- `simulate_sinc_pulse(...)`
- `simulate_gaussian_pulse(...)`
- `simulate_spsp_pulse(...)`
- `simulate_hs1_pulse(...)`
- `simulate_3d_multiband_spsp(...)`

### `simulators.pulse_validator`
Tools for analyzing pulse performance.
- `validate_pulse_performance(...)`
- `analyze_slice_profile(...)`
- `PulseValidationMetrics` class

<!-- Add gradient_pulses, sequences, vendor_adapters as they are implemented -->

<!-- This document should be auto-generated or meticulously maintained. -->
