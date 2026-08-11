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


def experiment_paths(run: str, model: str, tag: str):
    """Nested experiment layout, organized by run then model:
        checkpoints/<run>/<model>/<tag>.pt
        results/<run>/<model>/<tag>/
    Returns (ckpt_path, report_dir) and creates the parent dirs. One place so every
    training entry point (and the future unified train.py) organizes identically."""
    ckpt   = CHECKPOINTS_DIR / run / model / f"{tag}.pt"
    report = RESULTS_DIR / run / model / tag
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    return ckpt, report

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
WHEELBASE = 0.10                         # bicycle-model wheelbase (world units): yaw rate = v/WHEELBASE * tan(delta)
# Frames are stored as uint8 round(render*FRAME_SCALE) and divided by it on read. 254 rather than the
# usual 255 so the renderer's exact levels (0, 0.5, 1) round-trip BIT-EXACTLY -- 0.5*255 = 127.5 does
# not. Arbitrary values in [0,1] still quantize just as finely (max error 1/508).
FRAME_SCALE = 254.0
L = 0.12                                 # triangle length (altitude) in world units
W = 0.06                                 # triangle base width in world units
WORLD_BOUNDS = ((0.0, 1.0), (0.0, 1.0))  # ((x_min, x_max), (y_min, y_max))

# --- Heading marker --------------------------------------------------------
# Optional cue painted on the robot to break front/back visual symmetry, so
# heading is unambiguous in pixels. "none" = the classic binary triangle.
# "dot" = a bright disc near the nose, which makes the frame GRAYSCALE (body at
# BODY_VALUE, dot at NOSE_VALUE). Like L/W this is baked into every frame, so a
# change needs a fresh RUN + re-collection. Extend RENDER_MARKER for more cues.
RENDER_MARKER = "none"   # "none" | "dot" | "ring"
BODY_VALUE    = 0.5      # triangle-body intensity when a marker is active
NOSE_VALUE    = 1.0      # nose-dot intensity
NOSE_RADIUS   = 0.03     # nose-dot radius in world units
NOSE_OFFSET   = 0.5      # dot center as a fraction from centroid toward the apex
# "ring" marker: a WHITE body with a thin GREY circle outline encircling the front
# tip. A small, subtle heading cue (you still see the white apex, ringed by the grey
# indicator), replacing the big nose blob that dominated the frame and confounded
# position with heading.
RING_VALUE     = 0.5     # grey ring intensity (body is rendered white = 1.0)
RING_RADIUS    = 0.030   # ring radius in world units (circle around the leading edge)
RING_THICKNESS = 0.016   # ring band thickness in world units (~1px at grid 64)
RING_OFFSET    = 1     # ring center as a fraction from centroid toward the apex (white tip pokes through the front)

# --- Data collection -------------------------------------------------------
N_EPISODES = 20000
MAX_STEPS  = 100
HOLD_K     = 4
GRID_SIZE  = 128        # render resolution; must match the dataset's stored grid_size
PRED_STEP  = 1          # transition horizon in steps (1 = single step). Must divide HOLD_K so each transition spans one action.

# Action-sampling policy (random-walk). Not frozen; collect_dataset can override.
V_MEAN = 0.18      # mean linear velocity
V_STD = 0.05       # linear-velocity spread
OMEGA_MEAN = 0.0   # mean angular velocity (no turn bias)
OMEGA_STD = 0.6    # angular-velocity spread

# --- Actuator model (deliberately UNMODELED effect for the gray-box test) ---
# The commanded speed is not fully applied: the collector STORES the commanded v as
# the action but STEPS the sim with ACTUATOR_GAIN * v (a fixed, unknown efficiency).
# The model is given the command and must recover the gain as its learnable a_v
# coefficient. 1.0 = a perfect actuator, which reproduces the original datasets exactly.
ACTUATOR_GAIN = 1.0

# --- Model architecture ----------------------------------------------------
IN_CHANNELS      = 1
N_FRAMES         = 1               # frames stacked as encoder input channels (>1 = history; needed for hidden velocity)
LATENT_DIM       = 128
ENCODER_CHANNELS = (32, 64, 128)   # output channels per stride-2 conv stage
PREDICTOR_HIDDEN = 256
ACTION_DIM       = 2               # (v, omega)
PREDICTOR_MODE   = "residual"      # "mlp" | "residual" | "physics"
PHYSICS_LOCK_POSE = False          # physics mode: if True, the MLP cannot correct dims 0,1,2 (pose is pure kinematics)

# --- Grounded physics block (models/grounded.py) ---------------------------
# A separate, interpretable latent block holding [x, y, cos th, sin th]. It has
# its own raw linear head (NO BatchNorm) and is EXEMPT from SIGReg, so it can
# hold real-scale state instead of being smeared isotropic. The known kinematics
# run on it; the rest of the latent is the usual SIGReg'd free block.
# Grounding is label-free: the locked kinematic prediction + the prediction loss
# force the block to track true pose (no state supervision). See grounded.py.
PHYSICS_BLOCK_DIM      = 4       # dims 0..3 = x, y, cos th, sin th
GROUNDED_LOCK_BLOCK    = True    # True: block evolves by pure kinematics (no MLP). False: gray-box.
GROUNDED_BLOCK_BUDGET  = 0.0     # gray-box only (lock=False): max scale of the learned block correction
LAM_RECON              = 0.0     # optional decoder: reconstruct frame from the block alone (0 = off)
RECON_FG_WEIGHT        = 0.0     # foreground weighting of recon. 0 = plain MSE. >0: recon = bg_mean + w*fg_mean, averaging error over background vs lit (triangle/nose) pixels SEPARATELY so the shape is not drowned by the ~99% black background; the decoder must then draw a sharp, correctly-oriented shape, forcing heading into the block.
PRED_BLOCK_WEIGHT      = 1.0     # weight on the block dims in the prediction loss (pred = free_mean + w*block_mean). The block's prediction is the locked kinematics, so this is the physics-consistency strength. The block is 4/128 dims, so a plain MSE buries it; raise this to make the kinematics bite (the heading lever).

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
