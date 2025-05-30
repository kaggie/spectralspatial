# mri_pulse_library/core/constants.py
# Gyromagnetic ratio for Hydrogen (1H) in Hz/T
GAMMA_HZ_PER_T_PROTON = 42.57747892e6
# Gyromagnetic ratio for Hydrogen (1H) in MHz/T
GAMMA_MHZ_PER_T_PROTON = 42.57747892
# Gyromagnetic ratio for Hydrogen (1H) in rad/s/T
GAMMA_RAD_PER_S_PER_T_PROTON = GAMMA_HZ_PER_T_PROTON * 2 * 3.14159265359

# For convenience, a common gamma in Hz/G (as used in some existing files)
# 1 T = 10000 G
GAMMA_HZ_PER_G_PROTON = GAMMA_HZ_PER_T_PROTON / 10000.0
