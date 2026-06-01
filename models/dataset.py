#!/usr/bin/env python
# coding: utf-8

# # Transition dataset
# 
# Turns collected episodes into transition pairs for JEPA training. One training example: `(frame_t, action_t, frame_{t+1})`.
# 
# Two-representation rule: the model only ever sees `frame` and `action`. True state `(x, y, theta)` is returned only when `return_state=True`, for probing/eval only. Never feed it to the model.
# 
# > To generate `models/dataset.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python models/dataset.ipynb
# > ```

# ## Path setup
# 
# Same root-finding pattern as the files in `data`.

# In[ ]:


import sys
from pathlib import Path

# Walk up from the current dir to find the project root (the folder with constants.py).
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "constants.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ## Imports

# In[ ]:


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union


# ## RobotTransitions
# 
# PyTorch Dataset over the HDF5 file. Loads everything into RAM on init (40x40 frames are ~180 MB) so DataLoader workers don't fight over an open HDF5 handle.
# 
# Pass a list of episode indices to `episodes` for train/val subsets. The splitting itself lives in `make_dataloaders`.

# In[ ]:


class RobotTransitions(Dataset):
    """Transition pairs (frame_t, action_t, frame_{t+1}) from an HDF5 dataset.

    Parameters
    ----------
    h5_path : str or Path
        Path to a dataset file from collect_dataset.
    episodes : array-like of int, optional
        Subset of episode indices to include. Defaults to all episodes.
    return_state : bool
        If True, also return true (x, y, theta) as state/next_state.
        Answer key only. Never pass to the model.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        episodes=None,
        return_state: bool = False,
    ):
        with h5py.File(h5_path, "r") as f:
            self.frames    = f["frames"][:]
            self.actions   = f["actions"][:]
            self.states    = f["states"][:] if return_state else None
            starts         = f["episode_starts"][:]
            lengths        = f["episode_lengths"][:]
            self.grid_size = int(f.attrs["grid_size"])

        if episodes is not None:
            episodes = np.asarray(episodes, dtype=np.int64)
            starts, lengths = starts[episodes], lengths[episodes]

        self.index        = RobotTransitions._build_transition_index(starts, lengths)
        self.return_state = return_state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        i = int(self.index[idx])

        frame      = torch.from_numpy(self.frames[i]).to(torch.float32).unsqueeze(0)
        next_frame = torch.from_numpy(self.frames[i + 1]).to(torch.float32).unsqueeze(0)
        action     = torch.from_numpy(self.actions[i]).to(torch.float32)

        sample = {"frame": frame, "action": action, "next_frame": next_frame}

        if self.return_state:
            sample["state"]      = torch.from_numpy(self.states[i]).to(torch.float32)
            sample["next_state"] = torch.from_numpy(self.states[i + 1]).to(torch.float32)

        return sample

    @staticmethod
    def _build_transition_index(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Frame indices where both frame[i] and frame[i+1] are in the same episode.

        Parameters
        ----------
        starts : np.ndarray, shape (E,)
            Flat-array row where each episode begins.
        lengths : np.ndarray, shape (E,)
            Number of frames per episode.

        Returns
        -------
        np.ndarray of int64, shape (n_transitions,)
            Episodes shorter than 2 contribute nothing.
        """
        starts  = np.asarray(starts,  dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        chunks = [
            np.arange(s, s + n - 1)
            for s, n in zip(starts, lengths)
            if n >= 2
        ]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)


# ## make_dataloaders
# 
# Shuffles episode indices with a seeded RNG, splits into train/val, and returns two DataLoaders. Split is at the episode level so consecutive frames from the same trajectory never straddle train and val.

# In[ ]:


def make_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = 128,
    val_frac: float = 0.1,
    seed: int = 0,
    return_state: bool = False,
    num_workers: int = 0,
) -> tuple:
    """Train/val DataLoaders with an episode-level split.

    Parameters
    ----------
    h5_path : str or Path
        Input .h5 file from collect_dataset.
    batch_size : int
    val_frac : float
        Fraction of episodes held out for validation.
    seed : int
        RNG seed for the episode shuffle.
    return_state : bool
        Passed through to RobotTransitions. Eval/probe use only.
    num_workers : int
        DataLoader worker processes.

    Returns
    -------
    tuple of (train_loader, val_loader)
    """
    with h5py.File(h5_path, "r") as f:
        n_ep = len(f["episode_starts"])

    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_ep)
    n_val = int(round(val_frac * n_ep))
    val_ep, train_ep = perm[:n_val], perm[n_val:]

    train_ds = RobotTransitions(h5_path, episodes=train_ep, return_state=return_state)
    val_ds   = RobotTransitions(h5_path, episodes=val_ep,   return_state=return_state)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


# ## Tests
# 
# Sanity checks for the index logic, tensor shapes, the episode-boundary guarantee, and DataLoader batch shapes.

# In[ ]:


def _test_dataset():
    """Sanity tests for RobotTransitions and make_dataloaders."""
    h5_path = ROOT / "data" / "datasets" / "run01.h5"
    assert h5_path.exists(), f"dataset not found: {h5_path}"

    ds = RobotTransitions(h5_path)
    H = W = ds.grid_size
    print(f"transitions: {len(ds)}  ({H}x{W})")

    # shape and dtype
    sample = ds[0]
    assert sample["frame"].shape      == (1, H, W), sample["frame"].shape
    assert sample["next_frame"].shape == (1, H, W)
    assert sample["action"].shape     == (2,)
    assert sample["frame"].dtype      == torch.float32
    assert set(torch.unique(sample["frame"]).tolist()) <= {0.0, 1.0}

    # no last-frame of an episode used as frame_t
    with h5py.File(h5_path, "r") as f:
        starts  = f["episode_starts"][:]
        lengths = f["episode_lengths"][:]
    last_frames = set((starts + lengths - 1).tolist())
    assert not (set(ds.index.tolist()) & last_frames), "index crosses an episode seam!"

    # return_state path
    ds_s = RobotTransitions(h5_path, return_state=True)
    s = ds_s[0]
    assert s["state"].shape == (3,) and s["next_state"].shape == (3,)

    # DataLoader batch shapes
    train_dl, val_dl = make_dataloaders(h5_path, batch_size=64)
    batch = next(iter(train_dl))
    assert batch["frame"].shape  == (64, 1, H, W), batch["frame"].shape
    assert batch["action"].shape == (64, 2)

    n_train, n_val = len(train_dl.dataset), len(val_dl.dataset)
    print(f"split: {n_train} train + {n_val} val = {n_train + n_val} transitions")
    print("All dataset tests passed.")


# In[ ]:


if __name__ == "__main__":
    _test_dataset()

