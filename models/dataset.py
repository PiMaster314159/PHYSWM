# # Transition dataset
# 
# Turns collected episodes into transition pairs for JEPA training. One training example: `(frame_t, action_t, frame_{t+1})`.
# 
# Two-representation rule: the model only ever sees `frame` and `action`. True state `(x, y, theta)` is returned only when `return_state=True`, for probing/eval only. Never feed it to the model.
# 

# ## Path setup
# 
# Same root-finding pattern as the files in `data`.


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH, BATCH_SIZE, SEED, PRED_STEP


# ## Imports


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
            self.velocities = f["velocities"][:] if ("velocities" in f and return_state) else None  # bicycle: hidden v
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
            if self.velocities is not None:
                sample["velocity"]      = torch.tensor([float(self.velocities[i])], dtype=torch.float32)
                sample["next_velocity"] = torch.tensor([float(self.velocities[j])], dtype=torch.float32)

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


class RolloutWindows(Dataset):
    """Windows for multi-step rollout training: a start frame, K held actions, and the true
    poses at horizons 0..K. The model encodes the frame once and rolls K dynamics steps.
    Window starts are hold-aligned (stride `step`); episodes too short for K transitions
    contribute nothing. Only the FIRST frame is returned (the rollout is open-loop)."""

    def __init__(self, h5_path, episodes=None, K: int = 4, step: int = PRED_STEP):
        with h5py.File(h5_path, "r") as f:
            self.frames  = f["frames"][:]
            self.actions = f["actions"][:]
            self.states  = f["states"][:]
            starts  = f["episode_starts"][:]
            lengths = f["episode_lengths"][:]
            self.grid_size = int(f.attrs["grid_size"])
            hold_k = int(f.attrs["hold_k"])
        if step > 1 and hold_k % step != 0:
            raise ValueError(f"step {step} must divide hold_k {hold_k}")
        if episodes is not None:
            episodes = np.asarray(episodes, dtype=np.int64)
            starts, lengths = starts[episodes], lengths[episodes]
        self.K, self.step = K, step
        chunks = []
        for s, n in zip(starts, lengths):
            n_win = (n - 1) // step - K + 1            # length-K windows that fit in the episode
            if n_win >= 1:
                chunks.append(int(s) + np.arange(n_win) * step)
        self.index = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        b, step, K = int(self.index[idx]), self.step, self.K
        frame = torch.from_numpy(self.frames[b]).to(torch.float32).unsqueeze(0)
        next_frame = torch.from_numpy(self.frames[b + step]).to(torch.float32).unsqueeze(0)   # after action_0
        acts  = torch.stack([torch.from_numpy(self.actions[b + k * step]).to(torch.float32) for k in range(K)])
        poses = torch.stack([torch.from_numpy(self.states[b + k * step]).to(torch.float32) for k in range(K + 1)])
        return {"frame": frame, "next_frame": next_frame, "actions": acts, "poses": poses}


def make_rollout_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.1,
    seed: int = SEED,
    K: int = 4,
    num_workers: int = 0,
    step: int = PRED_STEP,
) -> tuple:
    """Train/val rollout-window loaders with the SAME episode-level split as make_dataloaders."""
    with h5py.File(h5_path, "r") as f:
        n_ep = len(f["episode_starts"])
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_ep)
    n_val = int(round(val_frac * n_ep))
    val_ep, train_ep = perm[:n_val], perm[n_val:]
    train_ds = RolloutWindows(h5_path, episodes=train_ep, K=K, step=step)
    val_ds   = RolloutWindows(h5_path, episodes=val_ep,   K=K, step=step)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


class HistoryTransitions(Dataset):
    """Like RobotTransitions but each example carries a STACK of `stack` frames (at stride
    `step`) ending at frame_t, plus the same stack ending at frame_{t+step}. The encoder reads
    pose off the newest frame and the hidden gain off the motion across the stack. Also returns
    the true (x,y,theta) for t and t+step, and the per-episode `gain` (the hidden parameter)."""

    def __init__(self, h5_path, episodes=None, stack: int = 4, step: int = PRED_STEP):
        with h5py.File(h5_path, "r") as f:
            self.frames  = f["frames"][:]
            self.actions = f["actions"][:]
            self.states  = f["states"][:]
            self.gains   = f["gains"][:] if "gains" in f else None
            self.velocities = f["velocities"][:] if "velocities" in f else None   # bicycle: hidden v
            starts  = f["episode_starts"][:]
            lengths = f["episode_lengths"][:]
            self.grid_size = int(f.attrs["grid_size"])
            hold_k = int(f.attrs["hold_k"])
        if step > 1 and hold_k % step != 0:
            raise ValueError(f"step {step} must divide hold_k {hold_k}")
        if episodes is not None:
            episodes = np.asarray(episodes, dtype=np.int64)
            starts, lengths = starts[episodes], lengths[episodes]
        self.stack, self.step = stack, step
        need = (stack - 1) * step                       # history needed before frame_t
        chunks = []
        for s, n in zip(starts, lengths):
            lo, hi = int(s) + need, int(s) + n - 1 - step   # t needs both history and a successor
            if hi >= lo:
                chunks.append(np.arange(lo, hi + 1, step))
        self.index = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def _stack_at(self, end: int) -> torch.Tensor:
        step, K = self.step, self.stack
        frs = [self.frames[end - (K - 1 - k) * step] for k in range(K)]   # oldest .. newest(end)
        return torch.from_numpy(np.stack(frs)).to(torch.float32)          # (stack, H, W)

    def _actions_at(self, end: int) -> torch.Tensor:
        """The commanded actions aligned with the frame stack (same positions as _stack_at). The
        gain head needs these: the per-episode gain is observed-motion / commanded-v, so without
        the actions a_v is unidentifiable (displacement a_v*v*dt confounds a_v with the unseen v)."""
        step, K = self.step, self.stack
        acts = [self.actions[end - (K - 1 - k) * step] for k in range(K)]  # oldest .. newest(end)
        return torch.from_numpy(np.stack(acts)).to(torch.float32)          # (stack, ACTION_DIM)

    def __getitem__(self, idx: int) -> dict:
        i, step = int(self.index[idx]), self.step
        sample = {
            "frame":            self._stack_at(i),
            "next_frame":       self._stack_at(i + step),
            "action_hist":      self._actions_at(i),
            "next_action_hist": self._actions_at(i + step),
            "action":           torch.from_numpy(self.actions[i]).to(torch.float32),
            "state":            torch.from_numpy(self.states[i]).to(torch.float32),
            "next_state":       torch.from_numpy(self.states[i + step]).to(torch.float32),
        }
        if self.gains is not None:
            sample["gain"] = torch.tensor([float(self.gains[i])], dtype=torch.float32)   # (1,)
        if self.velocities is not None:
            sample["velocity"]      = torch.tensor([float(self.velocities[i])], dtype=torch.float32)
            sample["next_velocity"] = torch.tensor([float(self.velocities[i + step])], dtype=torch.float32)
        return sample


def make_history_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.1,
    seed: int = SEED,
    stack: int = 4,
    num_workers: int = 0,
    step: int = PRED_STEP,
) -> tuple:
    """Train/val frame-stack loaders with the SAME episode-level split as make_dataloaders."""
    with h5py.File(h5_path, "r") as f:
        n_ep = len(f["episode_starts"])
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_ep)
    n_val = int(round(val_frac * n_ep))
    val_ep, train_ep = perm[:n_val], perm[n_val:]
    train_ds = HistoryTransitions(h5_path, episodes=train_ep, stack=stack, step=step)
    val_ds   = HistoryTransitions(h5_path, episodes=val_ep,   stack=stack, step=step)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


class HistoryRolloutWindows(Dataset):
    """Stack-aware open-loop rollout windows for the hidden-ego model. Each example carries a STACK
    of `stack` frames (at stride `step`) ending at the start, plus the action history over that
    stack (to encode pose_0 AND the hidden gain a_v); then K held actions to roll the dynamics
    forward, the true poses at horizons 0..K, and the per-episode true gain. Window starts need
    both the (stack-1)*step history before them and K*step successors after."""

    def __init__(self, h5_path, episodes=None, stack: int = 4, K: int = 8, step: int = PRED_STEP):
        with h5py.File(h5_path, "r") as f:
            self.frames  = f["frames"][:]
            self.actions = f["actions"][:]
            self.states  = f["states"][:]
            self.gains   = f["gains"][:] if "gains" in f else None
            starts  = f["episode_starts"][:]
            lengths = f["episode_lengths"][:]
            self.grid_size = int(f.attrs["grid_size"])
            hold_k = int(f.attrs["hold_k"])
        if step > 1 and hold_k % step != 0:
            raise ValueError(f"step {step} must divide hold_k {hold_k}")
        if episodes is not None:
            episodes = np.asarray(episodes, dtype=np.int64)
            starts, lengths = starts[episodes], lengths[episodes]
        self.stack, self.K, self.step = stack, K, step
        need = (stack - 1) * step                        # history before the start
        chunks = []
        for s, n in zip(starts, lengths):
            lo, hi = int(s) + need, int(s) + n - 1 - K * step   # start needs history AND K successors
            if hi >= lo:
                chunks.append(np.arange(lo, hi + 1, step))
        self.index = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def _stack_at(self, end: int) -> torch.Tensor:
        step, K = self.step, self.stack
        frs = [self.frames[end - (K - 1 - k) * step] for k in range(K)]
        return torch.from_numpy(np.stack(frs)).to(torch.float32)          # (stack, H, W)

    def _actions_at(self, end: int) -> torch.Tensor:
        step, K = self.step, self.stack
        acts = [self.actions[end - (K - 1 - k) * step] for k in range(K)]
        return torch.from_numpy(np.stack(acts)).to(torch.float32)          # (stack, ACTION_DIM)

    def __getitem__(self, idx: int) -> dict:
        b, step, K = int(self.index[idx]), self.step, self.K
        roll_actions = torch.stack([torch.from_numpy(self.actions[b + k * step]).to(torch.float32)
                                    for k in range(K)])                     # (K, ACTION_DIM)
        poses = torch.stack([torch.from_numpy(self.states[b + k * step]).to(torch.float32)
                             for k in range(K + 1)])                        # (K+1, 3) true (x,y,theta)
        sample = {
            "frame":        self._stack_at(b),
            "action_hist":  self._actions_at(b),
            "roll_actions": roll_actions,
            "poses":        poses,
        }
        if self.gains is not None:
            sample["gain"] = torch.tensor([float(self.gains[b])], dtype=torch.float32)   # (1,)
        return sample


def make_history_rollout_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.1,
    seed: int = SEED,
    stack: int = 4,
    K: int = 8,
    num_workers: int = 0,
    step: int = PRED_STEP,
) -> tuple:
    """Stack-aware rollout loaders with the SAME episode-level split as make_history_dataloaders
    (so the val episodes match the held-out set the model was selected on)."""
    with h5py.File(h5_path, "r") as f:
        n_ep = len(f["episode_starts"])
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n_ep)
    n_val = int(round(val_frac * n_ep))
    val_ep, train_ep = perm[:n_val], perm[n_val:]
    train_ds = HistoryRolloutWindows(h5_path, episodes=train_ep, stack=stack, K=K, step=step)
    val_ds   = HistoryRolloutWindows(h5_path, episodes=val_ep,   stack=stack, K=K, step=step)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


_DEFAULT_N_FRAMES = 1   # module default so eval/probe make_dataloaders() calls inherit --n-frames without threading


def set_default_n_frames(k: int) -> None:
    """Set the process-wide default frame-stack depth. train.py calls this once at startup, so every
    later make_dataloaders() (training, eval, probe) returns K-frame stacks without passing n_frames."""
    global _DEFAULT_N_FRAMES
    _DEFAULT_N_FRAMES = int(k)


def make_dataloaders(
    h5_path: Union[str, Path],
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.1,
    seed: int = SEED,
    return_state: bool = False,
    num_workers: int = 0,
    step: int = PRED_STEP,
    n_frames: int = None,
) -> tuple:
    """Train/val DataLoaders with an episode-level split.

    n_frames > 1 returns HISTORY stacks (K frames as channels, newest last) via make_history_dataloaders,
    so the same call site works for single- and multi-frame training. Defaults to the process-wide
    set_default_n_frames() value.

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
    n = _DEFAULT_N_FRAMES if n_frames is None else n_frames
    if n > 1:                                  # history stacks: frame is (K, H, W), pose at the newest frame
        return make_history_dataloaders(h5_path, batch_size=batch_size, val_frac=val_frac,
                                        seed=seed, stack=n, num_workers=num_workers, step=step)
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


if __name__ == "__main__":
    _test_dataset()

