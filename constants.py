#!/usr/bin/env python
# coding: utf-8

# # Constants
# 
# Project-wide fixed constants. Shared by `sim/`, `data/`, `models/`, `eval/`. Imported as `from constants import ...`, so run everything from the `PHYSWM` root.
# 
# Geometry and timing (`DT`, `WORLD_BOUNDS`) are frozen: changing them invalidates already-collected data. The `V_*`/`OMEGA_*` values are the default action-sampling policy for data collection.

# In[ ]:


DT = 0.1                                 # simulation timestep
L = 0.12                                 # triangle length (altitude) in world units
W = 0.06                                 # triangle base width in world units
WORLD_BOUNDS = ((0.0, 1.0), (0.0, 1.0))  # ((x_min, x_max), (y_min, y_max))

# Action sampling (random-walk policy for data collection)
V_MEAN = 0.18      # mean linear velocity
V_STD = 0.05       # linear-velocity spread
OMEGA_MEAN = 0.0   # mean angular velocity (no turn bias)
OMEGA_STD = 0.6    # angular-velocity spread

