# mri_pulse_library/rf_pulses/adiabatic/__init__.py

# Attempt to import existing HSnPulse, fail gracefully if not found or causes error
try:
    from .hs_pulse import HSnPulse
    existing_hs_pulse_imported = True
except ImportError:
    HSnPulse = None # Define as None if not found
    existing_hs_pulse_imported = False

from .waveforms import generate_hs_pulse_waveforms

__all__ = [
    'generate_hs_pulse_waveforms'
]
if existing_hs_pulse_imported and HSnPulse is not None:
    __all__.append('HSnPulse')
elif HSnPulse is None and existing_hs_pulse_imported: # If .hs_pulse exists but HSnPulse isn't in it
    pass # HSnPulse remains None, not added to __all__
