# # Data collection
# 
# Run random-action episodes through the dynamics simulator and stream them to an HDF5 file. Models never see the continuous state `(x, y, theta)`; we keep it as the answer key for probing.
# 
# Indexing convention (all three arrays have the same length `L` per episode):
# 
# - `states[t]`  : continuous `(x, y, theta)` at step `t`. Answer key, not fed to the model.
# - `frames[t]`  : render of `states[t]`.
# - `actions[t]` : action applied at `states[t]`. Links `states[t]` to `states[t+1]` for `t < L-1`. The final action has no recorded successor.

# ## Path setup
# 
# `collect.ipynb` lives in `data/`, one level below the project root. Add the root to `sys.path` so `config` and `sim/` import cleanly, wherever the kernel's working directory happens to be.


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH


# ## Imports


import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Optional, Union

from config import (
    DT, WORLD_BOUNDS, V_MEAN, V_STD, OMEGA_MEAN, OMEGA_STD,
    GRID_SIZE, HOLD_K, MAX_STEPS, N_EPISODES, SEED, RENDER_MARKER, NOSE_RADIUS,
    ACTUATOR_GAIN, WHEELBASE,
)
from sim.render import render_frame
from sim.environment import env_step, bounding_radius, out_of_bounds
from sim.dynamics import bicycle_step, wrap_theta
from sim.visualize import show_frame_strip, show_trajectory


# ## Action sampling
# 
# `v ~ N(v_mean, v_std)` clipped to `[0, 1]` (forward only). `omega ~ N(omega_mean, omega_std)` clipped to `[-pi/2, pi/2]` for reasonable single-step turns. Defaults come from `config.py`; override per call if needed.


def random_normal_clipped(
    mean: float, std: float, low: float, high: float, rng: np.random.Generator
) -> float:
    """Draw one scalar from N(mean, std), clipped to [low, high].

    Parameters
    ----------
    mean, std : float
        Gaussian parameters.
    low, high : float
        Inclusive clip range.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    float
        The clipped sample.
    """
    return float(np.clip(rng.normal(mean, std), low, high))


def sample_action(
    rng: np.random.Generator,
    v_mean: float = V_MEAN,
    v_std: float = V_STD,
    omega_mean: float = OMEGA_MEAN,
    omega_std: float = OMEGA_STD,
) -> np.ndarray:
    """Sample one random (v, omega) action.

    v is clipped to [0, 1] (forward only); omega to [-pi/2, pi/2].

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    v_mean, v_std : float
        Linear-velocity Gaussian.
    omega_mean, omega_std : float
        Angular-velocity Gaussian.

    Returns
    -------
    np.ndarray, shape (2,)
        (v, omega).
    """
    v = random_normal_clipped(v_mean, v_std, 0.0, 1.0, rng)
    omega = random_normal_clipped(omega_mean, omega_std, -np.pi / 2, np.pi / 2, rng)
    return np.array([v, omega])


# ## Initial-state sampling
# 
# `(x, y)` uniform inside the arena, kept a `margin` away from the walls so the spawn pose is legal. Margin defaults to `bounding_radius()`, the same value `out_of_bounds` uses. `theta` uniform on `[-pi, pi)`.


def sample_initial_state(
    rng: np.random.Generator,
    world_bounds: tuple = WORLD_BOUNDS,
    margin: Optional[float] = None,
) -> np.ndarray:
    """Random legal start pose inside the arena.

    (x, y) uniform inside the arena with a margin so the spawn is never already
    out of bounds; theta uniform on [-pi, pi).

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.
    margin : float or None
        Spawn margin. If None, defaults to the triangle's bounding radius (the
        same value `out_of_bounds` uses).

    Returns
    -------
    np.ndarray, shape (3,)
        (x, y, theta).
    """
    if margin is None:
        margin = bounding_radius()
    (x_min, x_max), (y_min, y_max) = world_bounds
    if x_max - 2 * margin <= x_min or y_max - 2 * margin <= y_min:
        raise ValueError(f"margin {margin} too large for world_bounds {world_bounds}")

    x = rng.uniform(x_min + margin, x_max - margin)
    y = rng.uniform(y_min + margin, y_max - margin)
    theta = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, theta])


# ## Episode loop
# 
# Roll out one episode and return the three equal-length arrays plus a termination cause.
# 
# Both termination modes give matching array lengths naturally:
# 
# - **wall**: action appended, `env_step` flips `done`, we break without appending the out-of-bounds state/frame. `len(states) == len(actions) == L`. The final action drove the robot into the wall (no successor).
# - **timeout**: loop ends with `len(actions) == len(states) - 1`. We pad one unused action so all three arrays are length `L`.
# 
# `hold_k` holds each sampled action constant for K steps before resampling. Default 1 resamples every step; 4-5 gives visibly smoother arcs.


def run_episode(
    rng: np.random.Generator,
    max_steps: int = 50,
    grid_size: int = 40,
    world_bounds: tuple = WORLD_BOUNDS,
    dt: float = DT,
    hold_k: int = 1,
    v_mean: float = V_MEAN,
    v_std: float = V_STD,
    omega_mean: float = OMEGA_MEAN,
    omega_std: float = OMEGA_STD,
    marker: str = RENDER_MARKER,
    nose_radius: float = NOSE_RADIUS,
    actuator_gain: float = 1.0,
    drag_c: float = 0.0,
) -> tuple:
    """Run one random-action episode.

    Returns parallel (frames, states, actions) plus the termination cause. All
    three arrays have the same length L.

    Indexing convention
    -------------------
    frames[t]  : render of states[t]
    states[t]  : continuous (x, y, theta) at step t (answer key, never the model's input)
    actions[t] : action applied at states[t]; links states[t] -> states[t+1] for t < L-1.
                 The last action has no recorded successor.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    max_steps : int
        Step cap. If reached, termination is "timeout".
    grid_size : int
        Render resolution. Must match the HDF5 dataset shape.
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.
    dt : float
        Sim timestep.
    hold_k : int
        Hold each sampled action for K steps before resampling. Default 1.
    v_mean, v_std, omega_mean, omega_std : float
        Action-sampling distribution.
    marker : {"none", "dot"}
        Heading cue passed to render_frame. "none" -> binary uint8 frames;
        anything else -> grayscale float32 frames.

    Returns
    -------
    frames : np.ndarray, shape (L, grid_size, grid_size); uint8 if marker="none", else float32
    states : np.ndarray, shape (L, 3), float32
    actions : np.ndarray, shape (L, 2), float32
    termination : {"wall", "timeout"}
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if hold_k < 1:
        raise ValueError(f"hold_k must be >= 1, got {hold_k}")

    frame_dtype = np.uint8 if marker == "none" else np.float32

    def _render(s):
        return render_frame(s, grid_size=grid_size, marker=marker, nose_radius=nose_radius).astype(frame_dtype)

    state = sample_initial_state(rng, world_bounds=world_bounds)
    states = [state]
    frames = [_render(state)]
    actions = []
    termination = "timeout"

    action = sample_action(rng, v_mean, v_std, omega_mean, omega_std)
    steps_held = 0

    for _ in range(max_steps - 1):
        if steps_held >= hold_k:
            action = sample_action(rng, v_mean, v_std, omega_mean, omega_std)
            steps_held = 0
        actions.append(action)              # store the COMMANDED action (what we think we sent)
        steps_held += 1

        # Non-ideal actuation applied to the TRUTH; the stored action keeps the full commanded v,
        # so the model sees the command and must recover the effect.
        #   actuator_gain: constant efficiency (applied = gain * v)   -> a constant a_v recovers it
        #   drag_c:        aerodynamic drag, a v^2 speed loss          -> needs a v^2 residual term
        # gain=1, drag_c=0 leaves env_step's input byte-identical to before.
        applied = action.copy()
        applied[0] = max(0.0, action[0] * actuator_gain - drag_c * action[0] ** 2)
        next_state, done = env_step(state, applied, dt=dt, world_bounds=world_bounds)
        if done:
            termination = "wall"
            break
        state = next_state
        states.append(state)
        frames.append(_render(state))

    # Timeout case: pad one unused action so all three arrays have length L.
    if len(actions) < len(states):
        actions.append(sample_action(rng, v_mean, v_std, omega_mean, omega_std))

    return (
        np.asarray(frames, dtype=frame_dtype),
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        termination,
    )


# ## Bicycle / throttle episode
#
# Kinematic bicycle with a throttle. Action is (a, delta) = (acceleration, steering angle); VELOCITY is a
# hidden STATE (integrates a), so it never appears in a single frame -- the model must read it off the last
# few frames (history). A target-speed controller drives v toward a resampled cruise target (clean speed
# coverage); steering wanders in the interior and turns back toward center near the walls (long episodes).
# We store the true velocity in its own array so eval can check whether the model RECOVERED speed.


def run_bicycle_episode(
    rng: np.random.Generator,
    max_steps: int = 500,
    grid_size: int = GRID_SIZE,
    world_bounds: tuple = WORLD_BOUNDS,
    dt: float = DT,
    hold_k: int = 4,
    v_mean: float = V_MEAN,
    v_std: float = V_STD,
    v_max: float = 0.6,
    a_max: float = 1.5,
    kp_a: float = 3.0,
    delta_std: float = 0.4,
    delta_max: float = 0.6,
    k_center: float = 1.5,
    wheelbase: float = WHEELBASE,
    marker: str = RENDER_MARKER,
    nose_radius: float = NOSE_RADIUS,
) -> tuple:
    """One bicycle-with-throttle episode. Returns (frames, states(x,y,theta), actions(a,delta), termination,
    velocities(v)). Velocity is the hidden state; it is returned separately for eval, never rendered."""
    frame_dtype = np.uint8 if marker == "none" else np.float32

    def _render(pose):
        return render_frame(pose, grid_size=grid_size, marker=marker, nose_radius=nose_radius).astype(frame_dtype)

    (x_min, x_max), (y_min, y_max) = world_bounds
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half = 0.5 * min(x_max - x_min, y_max - y_min)

    pose0 = sample_initial_state(rng, world_bounds=world_bounds)
    v0 = float(np.clip(rng.normal(v_mean, v_std), 0.0, v_max))     # start at a plausible cruise speed
    state = np.array([pose0[0], pose0[1], pose0[2], v0], float)    # (x, y, theta, v)
    states, vels, frames, actions = [state[:3].copy()], [state[3]], [_render(state[:3])], []
    termination = "timeout"

    def _targets():
        return (float(np.clip(rng.normal(v_mean, v_std), 0.0, v_max)),
                float(np.clip(rng.normal(0.0, delta_std), -delta_max, delta_max)))
    v_tgt, delta_rand = _targets()
    held = 0

    for _ in range(max_steps - 1):
        if held >= hold_k:
            v_tgt, delta_rand = _targets(); held = 0
        x, y, theta, v = state
        a = float(np.clip(kp_a * (v_tgt - v), -a_max, a_max))       # throttle -> drive v toward target
        r = np.hypot(x - cx, y - cy) / half                        # steering: wander inside, turn back at walls
        if r > 0.6:
            head_err = wrap_theta(np.arctan2(cy - y, cx - x) - theta)
            delta = float(np.clip(k_center * head_err, -delta_max, delta_max))
        else:
            delta = delta_rand
        action = np.array([a, delta], float)
        actions.append(action); held += 1
        nxt = bicycle_step(state, action, dt=dt, wheelbase=wheelbase, drag_c=0.0, v_min=0.0, v_max=v_max)
        if out_of_bounds(nxt[:3], world_bounds=world_bounds):
            termination = "wall"; break
        state = nxt
        states.append(state[:3].copy()); vels.append(state[3]); frames.append(_render(state[:3]))

    if len(actions) < len(states):
        actions.append(np.array([0.0, 0.0], float))
    return (np.asarray(frames, frame_dtype), np.asarray(states, np.float32),
            np.asarray(actions, np.float32), termination, np.asarray(vels, np.float32))


# ## Sanity check: 10 episodes
#
# Run 10 episodes, print lengths and termination reasons, then plot all 10 trajectories. Look for: (1) varied start poses, (2) episodes that don't all die on step 1, (3) smooth curves rather than noise.


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    episodes = [run_episode(rng, max_steps=500, grid_size=40, hold_k=4) for _ in range(10)]

    print("episode lengths:")
    for i, (_, s, _, term) in enumerate(episodes):
        print(f"  ep {i}: L={len(s):3d}  termination={term}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (_, s, _, term) in zip(axes.flat, episodes):
        arrow_every = max(1, len(s) // 12)
        show_trajectory(s, ax=ax, title=f"L={len(s)} ({term})", arrow_every=arrow_every)
    fig.tight_layout()
    plt.show()


# Pick the longest episode and eyeball a strip of its rendered frames. Confirms the renderer and dynamics agree.


if __name__ == "__main__":
    longest = max(episodes, key=lambda e: len(e[1]))
    frames, states, _, term = longest
    L = len(states)
    key_indices = np.linspace(0, L - 1, 6, dtype=int)
    show_frame_strip(
        [frames[i] for i in key_indices],
        titles=[f"t={i}" for i in key_indices],
        figsize=(15, 3),
    )
    plt.show()


# ## Dataset collection
# 
# Stream episodes straight to one HDF5 file at constant memory. Datasets are resizable and chunked, with gzip on frames only (states/actions are tiny). Episodes shorter than `min_length` are dropped to avoid degenerate spawn-and-crash runs.
# 
# Every generative parameter is stored as an HDF5 attr, so the file is self-describing: `h5py.File(...).attrs` tells you exactly what produced it.


def collect_dataset(
    path: Union[str, Path],
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    grid_size: int = GRID_SIZE,
    world_bounds: tuple = WORLD_BOUNDS,
    dt: float = DT,
    hold_k: int = HOLD_K,
    v_mean: float = V_MEAN,
    v_std: float = V_STD,
    omega_mean: float = OMEGA_MEAN,
    omega_std: float = OMEGA_STD,
    min_length: int = 10,
    seed: int = SEED,
    marker: str = RENDER_MARKER,
    nose_radius: float = NOSE_RADIUS,
    actuator_gain: float = ACTUATOR_GAIN,
    gain_range: tuple = None,
    drag_c: float = 0.0,
    dynamics: str = "unicycle",
    v_max: float = 0.6,
    a_max: float = 1.5,
    delta_max: float = 0.6,
    wheelbase: float = WHEELBASE,
    progress_every: int = 100,
) -> None:
    """Stream `n_episodes` random-action episodes to an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Output .h5 file. Parent directory created if missing.
    n_episodes : int
        Number of episodes to attempt.
    max_steps, grid_size, world_bounds, dt, hold_k : see `run_episode`.
    v_mean, v_std, omega_mean, omega_std : see `sample_action`.
    min_length : int
        Episodes with `len(states) < min_length` are dropped.
    seed : int
        Seeds the `np.random.default_rng` used for everything.
    marker : {"none", "dot"}
        Heading cue baked into every frame. "none" stores binary uint8 frames;
        a marker stores grayscale float32 frames. Default from config.
    progress_every : int
        Print a progress line every N attempted episodes.

    n_episodes, max_steps, grid_size, hold_k, seed, marker default from config.py.

    Notes
    -----
    Stored at the top level of the HDF5 file:
      frames           (T, H, W) uint8 (binary) or float32 (marker)  gzip-compressed
      states           (T, 3)    float32
      actions          (T, 2)    float32
      episode_starts   (E,)      int64   index into the flat arrays
      episode_lengths  (E,)      int64
      termination      (E,)      uint8   0=wall, 1=timeout
    Plus the generative parameters as `.attrs` (including render_marker).
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")

    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames_dtype = "uint8" if marker == "none" else "float32"

    with h5py.File(path, "w") as f:
        frames_ds = f.create_dataset(
            "frames",
            shape=(0, grid_size, grid_size),
            maxshape=(None, grid_size, grid_size),
            dtype=frames_dtype,
            chunks=(64, grid_size, grid_size),
            compression="gzip",
        )
        states_ds = f.create_dataset(
            "states", shape=(0, 3), maxshape=(None, 3),
            dtype="float32", chunks=(512, 3),
        )
        actions_ds = f.create_dataset(
            "actions", shape=(0, 2), maxshape=(None, 2),
            dtype="float32", chunks=(512, 2),
        )
        gains_ds = f.create_dataset(   # per-step actuator gain (constant within an episode)
            "gains", shape=(0,), maxshape=(None,),
            dtype="float32", chunks=(512,),
        )
        velocities_ds = None
        if dynamics == "bicycle":      # true (hidden) speed per step -- eval-only; the model must infer it
            velocities_ds = f.create_dataset(
                "velocities", shape=(0,), maxshape=(None,), dtype="float32", chunks=(512,),
            )

        episode_starts = []
        episode_lengths = []
        term_codes = []  # 0 = wall, 1 = timeout
        total = 0
        kept = 0

        for ep in range(n_episodes):
            # per-episode actuator gain: a fresh draw from gain_range each episode (the hidden
            # parameter the model must infer), or the fixed actuator_gain when range is None.
            ep_gain = float(rng.uniform(*gain_range)) if gain_range is not None else actuator_gain
            velocities = None
            if dynamics == "bicycle":
                frames, states, actions, termination, velocities = run_bicycle_episode(
                    rng, max_steps=max_steps, grid_size=grid_size, world_bounds=world_bounds, dt=dt,
                    hold_k=hold_k, v_mean=v_mean, v_std=v_std, v_max=v_max, a_max=a_max,
                    delta_max=delta_max, wheelbase=wheelbase, marker=marker, nose_radius=nose_radius,
                )
            else:
                frames, states, actions, termination = run_episode(
                    rng,
                    max_steps=max_steps, grid_size=grid_size,
                    world_bounds=world_bounds, dt=dt, hold_k=hold_k,
                    v_mean=v_mean, v_std=v_std,
                    omega_mean=omega_mean, omega_std=omega_std,
                    marker=marker, nose_radius=nose_radius,
                    actuator_gain=ep_gain, drag_c=drag_c,
                )
            if len(states) < min_length:
                continue

            L = len(states)
            for ds, block in ((frames_ds, frames), (states_ds, states), (actions_ds, actions)):
                ds.resize(ds.shape[0] + L, axis=0)
                ds[-L:] = block
            gains_ds.resize(gains_ds.shape[0] + L, axis=0)
            gains_ds[-L:] = np.full(L, ep_gain, dtype=np.float32)
            if velocities_ds is not None:
                velocities_ds.resize(velocities_ds.shape[0] + L, axis=0)
                velocities_ds[-L:] = velocities

            episode_starts.append(total)
            episode_lengths.append(L)
            term_codes.append(0 if termination == "wall" else 1)
            total += L
            kept += 1

            if (ep + 1) % progress_every == 0:
                print(f"{ep + 1}/{n_episodes} attempted, {kept} kept, {total} transitions")

        f.create_dataset("episode_starts",  data=np.asarray(episode_starts,  dtype=np.int64))
        f.create_dataset("episode_lengths", data=np.asarray(episode_lengths, dtype=np.int64))
        f.create_dataset("termination",     data=np.asarray(term_codes,      dtype=np.uint8))

        # Generative parameters: file is self-describing.
        f.attrs["grid_size"] = grid_size
        f.attrs["world_bounds"] = np.asarray(world_bounds, dtype=np.float64)
        f.attrs["dt"] = dt
        f.attrs["hold_k"] = hold_k
        f.attrs["max_steps"] = max_steps
        f.attrs["min_length"] = min_length
        f.attrs["seed"] = seed
        f.attrs["render_marker"] = marker
        f.attrs["nose_radius"] = nose_radius
        f.attrs["actuator_gain"] = actuator_gain   # truth for the gray-box test: model's a_v should recover this
        f.attrs["drag_c"] = drag_c                  # aerodynamic drag coefficient (applied v -= drag_c * v^2)
        f.attrs["dynamics"] = dynamics              # "unicycle" (action v,omega) or "bicycle" (action a,delta; v hidden)
        if dynamics == "bicycle":
            f.attrs["wheelbase"] = wheelbase
            f.attrs["v_max"] = v_max
            f.attrs["a_max"] = a_max
            f.attrs["delta_max"] = delta_max
        # per-episode gain range (the hidden parameter the model must infer); equal bounds = fixed gain
        f.attrs["gain_lo"] = float(gain_range[0]) if gain_range is not None else actuator_gain
        f.attrs["gain_hi"] = float(gain_range[1]) if gain_range is not None else actuator_gain
        f.attrs["v_mean"] = v_mean
        f.attrs["v_std"] = v_std
        f.attrs["omega_mean"] = omega_mean
        f.attrs["omega_std"] = omega_std
        f.attrs["n_episodes_attempted"] = n_episodes
        f.attrs["n_episodes_kept"] = kept
        f.attrs["total_transitions"] = total
        f.attrs["indexing_convention"] = (
            "frames[t] = render of states[t]; "
            "actions[t] applied at states[t] -> states[t+1] for t < L-1; "
            "final action has no recorded successor"
        )

    print(f"Done. {kept}/{n_episodes} episodes kept, {total} transitions -> {path}")


# ## Generate the dataset
# 
# The real run. With the constants' sampling policy and `hold_k=4`, episodes average tens of steps before hitting a wall. Bump `hold_k` higher if they come out too short.


if __name__ == "__main__":
    out = DATA_PATH
    collect_dataset(out, n_episodes=N_EPISODES, max_steps=MAX_STEPS, hold_k=HOLD_K, seed=SEED)

    # Write coverage diagnostics next to the dataset.
    from data.coverage import plot_coverage
    plot_coverage(out, show=False)


# ## Inspect run
# 
# Quick read-back to confirm shapes, attrs, and an example trajectory survived the round-trip.


if __name__ == "__main__":
    with h5py.File(DATA_PATH, "r") as f:
        print("datasets:")
        for name, ds in f.items():
            print(f"  {name:20s} shape={ds.shape}  dtype={ds.dtype}")
        print("\nattrs:")
        for k, v in f.attrs.items():
            print(f"  {k:25s} = {v}")

        lens = f["episode_lengths"][:]
        print(f"\nepisode lengths: min={lens.min()}, median={int(np.median(lens))}, max={lens.max()}, mean={lens.mean():.1f}")

        # Plot one trajectory by re-slicing into the flat array.
        starts = f["episode_starts"][:]
        idx = int(np.argmax(lens))  # longest episode
        s, e = starts[idx], starts[idx] + lens[idx]
        states = f["states"][s:e]

    show_trajectory(states, title=f"longest episode (L={len(states)})", arrow_every=max(1, len(states) // 15))
    plt.show()


