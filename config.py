"""Project config: paths, the active run, and frozen physics constants.

Single source of truth for the whole project (sim, data, models, eval).
Imported as `from config import ...`. Run everything from the PHYSWM root.

Plain hand-edited module (no notebook source) - edit it directly.
"""

from pathlib import Path

# Project root: the folder holding config.py. Found by walking up from cwd so
# imports resolve no matter where a notebook kernel was started.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())

# --- Active run + paths ----------------------------------------------------
# Change RUN to repoint every dataset and checkpoint below at once.
RUN = "run03"

DATASETS_DIR    = ROOT / "data" / "datasets"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"

DATA_PATH = DATASETS_DIR / f"{RUN}.h5"
CKPT_PATH = CHECKPOINTS_DIR / f"{RUN}_jepa.pt"

# Per-run report folder: probe metrics table + figures land here.
RESULTS_DIR = ROOT / "results"
REPORT_DIR  = RESULTS_DIR / RUN

# Shared RNG seed. train and probe must use the SAME seed so the probe scores
# on the exact held-out episodes the encoder never trained on.
SEED = 0

# --- Frozen physics --------------------------------------------------------
# Geometry and timing are FROZEN: every collected frame was rendered with these
# values, so changing them invalidates existing datasets. Treat a change as
# needing a fresh RUN and re-collection, not an in-place tweak.
DT = 0.1                                 # simulation timestep
L = 0.12                                 # triangle length (altitude) in world units
W = 0.06                                 # triangle base width in world units
WORLD_BOUNDS = ((0.0, 1.0), (0.0, 1.0))  # ((x_min, x_max), (y_min, y_max))

# --- Action-sampling policy ------------------------------------------------
# Defaults for the random-walk data collection. Not frozen; collect_dataset can
# override these per run (the values used are stored in each dataset's attrs).
V_MEAN = 0.18      # mean linear velocity
V_STD = 0.05       # linear-velocity spread
OMEGA_MEAN = 0.0   # mean angular velocity (no turn bias)
OMEGA_STD = 0.6    # angular-velocity spread
