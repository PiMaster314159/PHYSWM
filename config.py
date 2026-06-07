"""Project config: paths, the active run, physics constants, and hyperparameters.

Single source of truth for the whole project (sim, data, models, eval).
Imported as `from config import ...`. Run everything from the PHYSWM root.

Plain hand-edited module (no notebook source) - edit it directly. Module
defaults across the codebase (encoder dims, training knobs, collection params)
pull from the values here, so changing a value once changes it everywhere.

Two identities, kept separate:
  RUN        - the DATASET (.h5). One dataset feeds many model experiments.
  EXPERIMENT - the MODEL run (checkpoint + results). Auto-built at the bottom
               from the dataset + the knobs that define a run, so the folder
               name always says exactly what produced it.
"""

from pathlib import Path

# Project root: the folder holding config.py. Found by walking up from cwd so
# imports resolve no matter where a notebook kernel was started.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())

DATASETS_DIR    = ROOT / "data" / "datasets"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
RESULTS_DIR     = ROOT / "results"

# --- Dataset --------------------------------------------------------------
# RUN names the .h5 file only. It is shared across model experiments, so it
# does NOT name the checkpoint or results (those are per-experiment, below).
RUN       = "run03_128x128"
DATA_PATH = DATASETS_DIR / f"{RUN}.h5"

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
PRED_STEP  = 1          # transition horizon in steps (1 = single step). Must divide HOLD_K so each transition spans one action.

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
PREDICTOR_MODE   = "residual"      # "mlp" | "residual" | "physics"
PHYSICS_LOCK_POSE = False          # physics mode: if True, the MLP cannot correct dims 0,1,2 (pose is pure kinematics)

# --- Training --------------------------------------------------------------
LR         = 1e-3
LAM        = 0.005   # SIGReg weight
LAM_PHYS   = 0.0     # physics-consistency loss weight (physics mode only; pulls encoded next pose toward the kinematic prediction)
EPOCHS     = 10
BATCH_SIZE = 256

# --- Experiment identity ---------------------------------------------------
# Names the checkpoint and the results folder. Auto-built from the dataset plus
# the knobs that define a model run, so you never hand-rename folders and the
# name is self-describing. NOTE is an optional manual label for one-offs (e.g.
# a learning-rate sweep): set NOTE = "lr2e3" and it gets appended.
TAG = f"{PREDICTOR_MODE}_s{PRED_STEP}_e{EPOCHS}"          # e.g. "physics_s4_e10"
if PREDICTOR_MODE == "physics":
    if LAM_PHYS > 0:       TAG += f"_lp{LAM_PHYS:g}"      # e.g. "..._lp1" for the loss-term sweep
    if PHYSICS_LOCK_POSE:  TAG += "_lock"                 # the hard architectural variant
NOTE       = ""
EXPERIMENT = f"{RUN}_{TAG}" + (f"_{NOTE}" if NOTE else "")

CKPT_PATH  = CHECKPOINTS_DIR / f"{EXPERIMENT}.pt"
REPORT_DIR = RESULTS_DIR / EXPERIMENT
