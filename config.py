#!/usr/bin/env python
# coding: utf-8

# # Config
# 
# Experiment paths and the active-run pointer. Kept separate from `constants.py`, which holds frozen physics. Change `RUN` here and the whole project (collect, train, probe) repoints at that dataset and its checkpoint.
# 
# > To generate `config.py`, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python config.ipynb
# > ```

# In[ ]:


from pathlib import Path

# Project root: the folder holding constants.py. Found by walking up from cwd
# so this resolves no matter where a notebook kernel was started.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "constants.py").exists())

# The active run. Changing this one string repoints every path below.
RUN = "run03"

DATASETS_DIR    = ROOT / "data" / "datasets"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"

DATA_PATH = DATASETS_DIR / f"{RUN}.h5"
CKPT_PATH = CHECKPOINTS_DIR / f"{RUN}_jepa.pt"

# Shared RNG seed. train and probe must use the SAME seed so the probe scores
# on the exact held-out episodes the encoder never trained on.
SEED = 0

