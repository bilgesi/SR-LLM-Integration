# constants.py

"""
Physical and simulation constants used across multiple simulation classes.
Keeping them in a single file ensures easy maintenance and global consistency.
"""

# Physical constants
GRAVITY = 9.81         # Gravitational acceleration (m/s^2)
AIR_DENSITY = 1.225    # Air density at sea level (kg/m^3)

# Default simulation settings (can be overridden)
DEFAULT_DT = 0.01      # Default time step size (seconds)


FITNESS_FAIL_SCORE = 9999.0
EPSILON = 1e-8
EARLY_STOP_TOLERANCE = 0.001
STALE_GEN_LIMIT = 3
MUTATION_STD_DEV = 0.02
