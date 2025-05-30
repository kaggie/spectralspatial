# mri_pulse_library/__init__.py
# Main init for the library
from . import core
from . import rf_pulses
from . import simulators
# Add other main modules as they become functional
# from . import gradient_pulses
# from . import sequences
# from . import vendor_adapters
# from . import examples

# You might want to expose key functions or classes at the top level, e.g.:
# from .core.constants import GAMMA_HZ_PER_T_PROTON
# from .rf_pulses.simple import generate_hard_pulse
# from .simulators import simulate_hard_pulse
