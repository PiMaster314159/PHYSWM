"""Phase 1: a "spin in place" dataset that isolates heading from position.

Each episode parks the robot at a fixed point and spins it (v=0, constant omega),
so within any transition the ONLY thing that changes frame-to-frame is heading.
The spin center is jittered between episodes (not within) so position still has
some variance for a sane R^2, but never moves during a spin. Heading is covered
across the full circle by random per-episode start angles.

Writes the same HDF5 schema as data/collect.py so make_dataloaders / probe /
diagnose read it unchanged. Two flavors, matching the two encoder families:
  spin_64x64.h5       marker="none"  -> probe the binary-trained run02 models
  spin_64x64_nose.h5  marker="dot"   -> probe the nose-dot run04 models

Run from the PHYSWM root:
    python make_spin_dataset.py
"""

import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import h5py

from config import DT, WORLD_BOUNDS, DATASETS_DIR
from sim.render import render_frame
from sim.dynamics import step


def make_spin_dataset(
    path: Path,
    marker: str,
    grid_size: int,
    n_episodes: int = 20000,
    steps_per_ep: int = 24,
    seed: int = 0,
    center_lo: float = 0.35,
    center_hi: float = 0.65,
    omega_min: float = 0.3,
    omega_max: float = float(np.pi / 2),
) -> None:
    """Generate one spin-in-place dataset.

    Parameters
    ----------
    path : Path
        Output .h5 file.
    marker : {"none", "dot"}
        Heading cue baked into every frame. Must match the encoder being probed.
    grid_size : int
        Render resolution. Must match the encoder's training grid.
    n_episodes : int
        Number of spins. Each is one fixed location.
    steps_per_ep : int
        Frames per spin. Action (0, omega) is constant across the whole episode,
        so the stored hold_k equals this (any step divisor stays single-action).
    seed : int
        RNG seed.
    center_lo, center_hi : float
        Spin-center box. Kept well inside the arena so no heading clips a wall.
    omega_min, omega_max : float
        Per-episode angular-speed magnitude (sign is random).
    """
    rng = np.random.default_rng(seed)
    frame_dtype = np.uint8 if marker == "none" else np.float32

    frames, states, actions = [], [], []
    starts, lengths, term = [], [], []
    total = 0

    for _ in range(n_episodes):
        cx = rng.uniform(center_lo, center_hi)
        cy = rng.uniform(center_lo, center_hi)
        theta = rng.uniform(-np.pi, np.pi)
        omega = rng.uniform(omega_min, omega_max) * rng.choice([-1.0, 1.0])
        action = np.array([0.0, omega], dtype=np.float32)   # v=0: pure rotation

        s = np.array([cx, cy, theta], dtype=float)
        for _t in range(steps_per_ep):
            states.append(s.astype(np.float32))
            frames.append(render_frame(s, grid_size=grid_size, marker=marker).astype(frame_dtype))
            actions.append(action.copy())
            s = step(s, action, dt=DT)   # v=0 keeps (x, y) fixed; theta advances

        starts.append(total)
        lengths.append(steps_per_ep)
        term.append(1)                  # timeout (spins never hit a wall)
        total += steps_per_ep

    frames  = np.asarray(frames,  dtype=frame_dtype)
    states  = np.asarray(states,  dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames",  data=frames,  chunks=(64, grid_size, grid_size), compression="gzip")
        f.create_dataset("states",  data=states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("episode_starts",  data=np.asarray(starts,  dtype=np.int64))
        f.create_dataset("episode_lengths", data=np.asarray(lengths, dtype=np.int64))
        f.create_dataset("termination",     data=np.asarray(term,    dtype=np.uint8))

        f.attrs["grid_size"]   = grid_size
        f.attrs["world_bounds"] = np.asarray(WORLD_BOUNDS, dtype=np.float64)
        f.attrs["dt"]          = DT
        f.attrs["hold_k"]      = steps_per_ep   # action constant across the whole spin
        f.attrs["render_marker"] = marker
        f.attrs["kind"]        = "spin_in_place"
        f.attrs["seed"]        = seed
        f.attrs["n_episodes_kept"] = n_episodes
        f.attrs["total_transitions"] = total
        f.attrs["indexing_convention"] = (
            "fixed (x,y) per episode, v=0 so position never moves within a spin; "
            "only heading changes frame-to-frame"
        )

    print(f"wrote {path}  ({total} frames, {n_episodes} spins, marker={marker}, {grid_size}x{grid_size})")


if __name__ == "__main__":
    make_spin_dataset(DATASETS_DIR / "spin_64x64.h5",      marker="none", grid_size=64, seed=0)
    make_spin_dataset(DATASETS_DIR / "spin_64x64_nose.h5", marker="dot",  grid_size=64, seed=0)
