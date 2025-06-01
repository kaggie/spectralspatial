# mri_pulse_library/rf_pulses/__init__.py
from . import simple
from . import spectral_spatial
from . import adiabatic
from . import composite # Add new import
from .multiband import MultibandPulseDesigner
# To be added later:
# e.g. if there are other categories like VERSE pulses, etc.

__all__ = [
    'simple',
    'spectral_spatial',
    'adiabatic',
    'composite',
    'MultibandPulseDesigner'
]
# Note: 'multiband' as a module is not explicitly added to __all__,
# but MultibandPulseDesigner from it is. This is a common pattern.
# If direct access to the 'multiband' module (e.g. rf_pulses.multiband.some_other_func)
# is desired, then 'multiband' should be added to __all__ and imported as 'from . import multiband'.
# The current import "from .multiband import MultibandPulseDesigner" is fine for exposing the class.
