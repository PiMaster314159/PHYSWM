"""Project config: paths, the active run, physics constants, and hyperparameters.

Single source of truth for the whole project (sim, data, models, eval).
Imported as `from config import ...`. Run everything from the PHYSWM root.

Plain hand-edited module (no notebook source) - edit it directly. Module
defaults across the codebase (encoder dims, training knobs, collection params)
pull from the values here, so changing a value once changes it everywhere.
"""

from pathlib import Path

# Project root: the folder holding config.py. Found by walking up from cwd so
# imports resolve no matter where a notebook kernel was started.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())

# --- Active run + paths ----------------------------------------------------
# Change RUN to repoint every dataset, checkpoint, and report folder at once.
RUN = "run03_128x128"

DATASETS_DIR    = ROOT / "data" / "datasets"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"

DATA_PATH = DATASETS_DIR / f"{RUN}.h5"
CKPT_PATH = CHECKPOINTS_DIR / f"{RUN}_jepa_long.pt"

# Per-run report folder: probe metrics table + figures land here.
RESULTS_DIR = ROOT / "results"
REPORT_DIR  = RESULTS_DIR / f"{RUN}_long"

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

# --- Data collection -------------------------------------------------------
N_EPISODES = 5000
MAX_STEPS  = 100
HOLD_K     = 4
GRID_SIZE  = 128        # render resolution; must match the dataset's stored grid_size

# Action-sampling policy (random-walk). Not frozen; collect_dataset can override.
V_MEAN = 0.18      # mean linear velocity
V_STD = 0.05       # linear-velocity spread
OMEGA_MEAN = 0.0   # mean angular velocity (no turn bias)
OMEGA_STD = 0.6    # angular-velocity spread

# --- Model architecture ----------------------------------------------------
IN_CHANNELS      = 1
LATENT_DIM       = 128
ENCODER_CHANNELS = (32, 64, 128)   # output channels per stride-2 conv stage
PREDICTOR_HIDDEN = 256
ACTION_DIM       = 2               # (v, omega)

# --- Training --------------------------------------------------------------
LR         = 1e-3
LAM        = 0.005   # SIGReg weight
EPOCHS     = 10
BATCH_SIZE = 256
