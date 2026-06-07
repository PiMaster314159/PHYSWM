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

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH, BATCH_SIZE, SEED, PRED_STEP


# ## Imports

# In[ ]:


import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union


# ## RobotTransitions
# 
# PyTorch Dataset over the HDF5 file. Loads everything into RAM on init so DataLoader workers don't fight over an open HDF5 handle. Pass a list of episode indices to `episodes` for train/val subsets (the split itself lives in `make_dataloaders`).
# 
# `step` sets the prediction horizon. `step=1` is single-step (`frame_t -> frame_{t+1}`). `step>1` forms hold-aligned, non-overlapping windows so each transition spans one constant action; it requires `step` to divide the dataset's `hold_k`. Over a window the same action is applied `step` times, so `frame_{t+step}` is where that one action takes the robot, and the accumulated rotation is large enough that heading finally matters to the prediction.

# In[ ]:


class RobotTransitions(Dataset):
    """Transition pairs (frame_t, action_t, frame_{t+step}) from an HDF5 dataset.

    Parameters
    ----------
    h5_path : str or Path
        Path to a dataset file from collect_dataset.
    episodes : array-like of int, optional
        Subset of episode indices to include. Defaults to all episodes.
    return_state : bool
        If True, also return true (x, y, theta) for the start and end of each
        transition as state/next_state. Answer key only. Never pass to the model.
    step : int
        Transition horizon in sim steps. 1 = single step. For step > 1 the
        dataset's hold_k must be a multiple of step, so each (non-overlapping,
        hold-aligned) window spans a single constant action.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        episodes : Optional[npt.NDArray[np.int64]] = None,
        return_state: bool = False,
        step: int = PRED_STEP,
    ):
        with h5py.File(h5_path, "r") as f:
            self.frames    = f["frames"][:]
            self.actions   = f["actions"][:]
            self.states    = f["states"][:] if return_state else None
            starts         = f["episode_starts"][:]
            lengths        = f["episode_lengths"][:]
            self.grid_size = int(f.attrs["grid_size"])
            hold_k         = int(f.attrs["hold_k"])

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        if step > 1 and hold_k % step != 0:
            raise ValueError(
                f"step {step} must divide the dataset's hold_k {hold_k} so each "
                f"transition spans a single action"
            )

        if episodes is not None:
            episodes = np.asarray(episodes, dtype=np.int64)
            starts, lengths = starts[episodes], lengths[episodes]

        self.step         = step
        self.index        = RobotTransitions._build_transition_index(starts, lengths, step)
        self.return_state = return_state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        i = int(self.index[idx])
        j = i + self.step

        frame      = torch.from_numpy(self.frames[i]).to(torch.float32).unsqueeze(0)
        next_frame = torch.from_numpy(self.frames[j]).to(torch.float32).unsqueeze(0)
        action     = torch.from_numpy(self.actions[i]).to(torch.float32)

        sample = {"frame": frame, "action": action, "next_frame": next_frame}

        if self.return_state:
            sample["state"]      = torch.from_numpy(self.states[i]).to(torch.float32)
            sample["next_state"] = torch.from_numpy(self.states[j]).to(torch.float32)

        return sample

    @staticmethod
    def _build_transition_index(starts: np.ndarray, lengths: np.ndarray, step: int) -> np.ndarray:
        """Start indices of non-overlapping length-`step` windows inside episodes.

        For each entry i, (frames[i], actions[i], frames[i+step]) is a valid
        transition that stays inside one episode.

        Parameters
        ----------
        starts : np.ndarray, shape (E,)
            Flat-array row where each episode begins.
        lengths : np.ndarray, shape (E,)
            Number of frames per episode.
        step : int
            Transition horizon in steps.

        Returns
        -------
        np.ndarray of int64, shape (n_transitions,)
            Episodes shorter than step+1 contribute nothing.
        """
        starts  = np.asarray(starts,  dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        chunks = []
        for s, n in zip(starts, lengths):
            n_windows = (n - 1) // step          # full step-windows that fit in the episode
            if n_windows >= 1:
                chunks.append(s + np.arange(n_windows) * step)
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)


# ## make_dataloaders
# 
# Shuffles episode indices with a seeded RNG, splits into train/val, and returns two DataLoaders. Split is at the episode level so consecutive frames from the same trajectory never straddle train and val.

# In[ ]:


def make_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.1,
    seed: int = SEED,
    return_state: bool = False,
    num_workers: int = 0,
    step: int = PRED_STEP,
) -> tuple:
    """Train/val DataLoaders with an episode-level split.

    Parameters
    ----------
    h5_path : str or Path
        Input .h5 file from collect_dataset.
    batch_size : int
        Default from config.
    val_frac : float
        Fraction of episodes held out for validation.
    seed : int
        RNG seed for the episode shuffle. Default from config.
    return_state : bool
        Passed through to RobotTransitions. Eval/probe use only.
    num_workers : int
        DataLoader worker processes.
    step : int
        Transition horizon. Default from config (PRED_STEP).

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

    train_ds = RobotTransitions(h5_path, episodes=train_ep, return_state=return_state, step=step)
    val_ds   = RobotTransitions(h5_path, episodes=val_ep,   return_state=return_state, step=step)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


# ## Tests
# 
# Sanity checks for the index logic, tensor shapes, the episode-boundary guarantee, and DataLoader batch shapes.

# In[ ]:


def _test_dataset():
    """Sanity tests for RobotTransitions and make_dataloaders."""
    h5_path = DATA_PATH
    assert h5_path.exists(), f"dataset not found: {h5_path}"

    ds = RobotTransitions(h5_path)   # single-step (step=PRED_STEP default)
    H = W = ds.grid_size
    print(f"transitions: {len(ds)}  ({H}x{W})  step={ds.step}")

    # shape and dtype
    sample = ds[0]
    assert sample["frame"].shape      == (1, H, W), sample["frame"].shape
    assert sample["next_frame"].shape == (1, H, W)
    assert sample["action"].shape     == (2,)
    assert sample["frame"].dtype      == torch.float32
    # binary datasets are {0,1}; marker datasets are grayscale. Both live in [0,1].
    fmin, fmax = sample["frame"].min().item(), sample["frame"].max().item()
    assert 0.0 <= fmin and fmax <= 1.0, f"frame values out of [0,1]: [{fmin}, {fmax}]"

    # no last-frame of an episode used as frame_t
    with h5py.File(h5_path, "r") as f:
        starts  = f["episode_starts"][:]
        lengths = f["episode_lengths"][:]
        hold_k  = int(f.attrs["hold_k"])
    last_frames = set((starts + lengths - 1).tolist())
    assert not (set(ds.index.tolist()) & last_frames), "index crosses an episode seam!"

    # return_state path
    ds_s = RobotTransitions(h5_path, return_state=True)
    s = ds_s[0]
    assert s["state"].shape == (3,) and s["next_state"].shape == (3,)

    # multi-step (hold-aligned): step must divide hold_k; gives fewer transitions
    ds_k = RobotTransitions(h5_path, step=hold_k)
    assert len(ds_k) < len(ds), "multi-step should yield fewer transitions than single-step"
    i0 = int(ds_k.index[0])
    assert np.array_equal(ds_k[0]["next_frame"][0].numpy(), ds_k.frames[i0 + hold_k]), \
        "next_frame is not `step` frames ahead"
    # a step that does not divide hold_k must be rejected
    try:
        RobotTransitions(h5_path, step=hold_k + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for step not dividing hold_k")
    print(f"multi-step OK: step={hold_k} -> {len(ds_k)} transitions (vs {len(ds)} single-step)")

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

