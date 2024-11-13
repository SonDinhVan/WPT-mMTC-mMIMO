"""
This file contains the constants and parameters of the system.
"""
import numpy as np

# Constants
# Speed of light
LIGHT_SPEED = 3.0 * 10**8
K_B = 1.381 * 10**-23
T0 = 290
NOISE_FIGURE = 7.2

# Channel constants
# Main frequency in GHz
FREQUENCY = 2.5
# Bandwidth MHz
BANDWIDTH = 10
# Noise power in watt
NOISE_POWER = BANDWIDTH * 10**6 * K_B * T0 * 10**(NOISE_FIGURE/10)

# Symbol interval
SYMBOL_INTERVAL = 1 / 3.0 / 10**6
# Coherence interval (#symbol)
TAU_C = 100

# Energy harvesting model
E_MAX = 24.0 * 10**-3
a_PARAMETER = 150.0
b_PARAMETER = 0.0014
c_PARAMETER = 1.0 / (1 + np.exp(a_PARAMETER * b_PARAMETER))
E_TH = 0.0

# System constants
# Deployment area
RADIUS = 25.0
# Transmit power of UE by default in Watts
P_MAX_UE = 0.5
POWER_AP = 10.0
