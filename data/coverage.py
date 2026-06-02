#!/usr/bin/env python
# coding: utf-8

# # Dataset coverage diagnostics
# 
# Read an HDF5 dataset from `collect.ipynb` and check it's worth training on. Builds a 6-plot summary panel, then saves it plus every plot as its own PNG into a `<name>_coverage/` folder next to the dataset.
# 
# What we look at:
# 
# - **Action sampler**: empirical `v` and `ω` histograms against the Gaussians we configured.
# - **Heading coverage**: for each `(x, y)` bin, how spread-out the observed headings are. This is the one that actually matters. High everywhere means the robot has been seen facing every direction across the whole arena.
# - **Per-step changes**: within-episode `Δθ` and `Δposition`, to confirm the transitions are learnable.
# - **Episode lengths**: split by whether the episode ended at a wall or timed out.
# 
# Two extra plots (`(x, y)` occupancy and the global `θ` histogram) are saved as standalone images but kept out of the panel. They mostly reflect the uniform spawn, so they are reference more than headline.
# 
# > To regenerate `data/coverage.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python data/coverage.ipynb
# > ```

# ## Path setup
# 
# Make the project root importable, same as `collect.ipynb`.

# In[ ]:


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "constants.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH


# ## Imports
# 
# Plain matplotlib, no seaborn.

# In[ ]:


import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import h5py
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Mapping, Optional, Union


# ## Helpers
# 
# `gaussian_pdf` is the analytic curve we overlay on the action histograms. `wrap_angle` puts angle differences in `[-pi, pi)` so a step across the `+-pi` seam reads as a small turn, not a full loop. `per_step_diffs` collects within-episode `Δθ` and `Δposition`, never crossing an episode boundary.

# In[ ]:


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal PDF N(mu, sigma) evaluated at x."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def wrap_angle(theta: npt.ArrayLike) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi). Same convention as `sim.dynamics.wrap_theta`."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def per_step_diffs(
    states: np.ndarray, episode_starts: np.ndarray, episode_lengths: np.ndarray
) -> tuple:
    """Within-episode per-step Δθ and Δposition. Never crosses an episode boundary.

    Parameters
    ----------
    states : np.ndarray, shape (T, 3)
    episode_starts : np.ndarray, shape (E,)
    episode_lengths : np.ndarray, shape (E,)

    Returns
    -------
    d_theta : np.ndarray, shape (T - E,)
        Wrapped angle differences within episodes.
    d_pos : np.ndarray, shape (T - E,)
        Euclidean position step magnitudes within episodes.
    """
    d_theta_chunks = []
    d_pos_chunks = []
    for s, L in zip(episode_starts, episode_lengths):
        if L < 2:
            continue
        ep = states[s : s + L]
        dx = np.diff(ep[:, 0])
        dy = np.diff(ep[:, 1])
        dtheta = wrap_angle(np.diff(ep[:, 2]))
        d_theta_chunks.append(dtheta)
        d_pos_chunks.append(np.sqrt(dx ** 2 + dy ** 2))
    if not d_theta_chunks:
        return np.array([]), np.array([])
    return np.concatenate(d_theta_chunks), np.concatenate(d_pos_chunks)


# ## Action plots
# 
# One histogram each for `v` and `ω`, with the configured Gaussian drawn on top. If the sampler is doing its job the bars track the curve. A small pile-up at `v = 0` or `v = 1` is just the clip and is expected.

# In[ ]:


def plot_v_hist(actions: np.ndarray, attrs: Mapping, ax: Axes) -> None:
    """Histogram of linear velocity v, with its configured Gaussian overlaid."""
    v = actions[:, 0]
    v_mean, v_std = float(attrs["v_mean"]), float(attrs["v_std"])
    ax.hist(v, bins=50, density=True, alpha=0.6, color="steelblue")
    xs = np.linspace(0.0, 1.0, 200)
    ax.plot(xs, gaussian_pdf(xs, v_mean, v_std), "r--", lw=1.5,
            label=f"N({v_mean:.2f}, {v_std:.2f})")
    ax.set_title("linear velocity (v)")
    ax.set_xlabel("v")
    ax.legend(fontsize=8)


def plot_omega_hist(actions: np.ndarray, attrs: Mapping, ax: Axes) -> None:
    """Histogram of turn rate omega, with its configured Gaussian overlaid."""
    omega = actions[:, 1]
    om_mean, om_std = float(attrs["omega_mean"]), float(attrs["omega_std"])
    ax.hist(omega, bins=50, density=True, alpha=0.6, color="steelblue")
    xs = np.linspace(omega.min(), omega.max(), 200)
    ax.plot(xs, gaussian_pdf(xs, om_mean, om_std), "r--", lw=1.5,
            label=f"N({om_mean:.2f}, {om_std:.2f})")
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_title("turn rate (ω)")
    ax.set_xlabel("ω (rad)")
    ax.legend(fontsize=8)


# ## State plots
# 
# The headline is the heading-coverage map: for each `(x, y)` bin, the entropy of the headings seen there. High and even means the robot faced all directions everywhere; cold spots are heading-biased regions where probes will do unevenly.
# 
# The `(x, y)` occupancy and global `θ` histogram (saved as extras, not in the panel) mostly just reflect the uniform spawn, so don't read too much into them.

# In[ ]:


def plot_xy_hist(states: np.ndarray, world_bounds: tuple, ax: Axes) -> None:
    """2D histogram of visited (x, y) positions."""
    (x_min, x_max), (y_min, y_max) = world_bounds
    _, _, _, im = ax.hist2d(
        states[:, 0], states[:, 1],
        bins=30, range=[[x_min, x_max], [y_min, y_max]], cmap="viridis",
    )
    ax.set_title("where the robot spent time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")


def plot_theta_marginal(states: np.ndarray, ax: Axes) -> None:
    """Histogram of headings theta across the whole dataset."""
    theta = wrap_angle(states[:, 2])
    ax.hist(theta, bins=36, density=True, alpha=0.6, color="steelblue")
    ax.axhline(1 / (2 * np.pi), color="r", ls="--", lw=1, label="uniform")
    ax.set_title("heading distribution")
    ax.set_xlabel("θ (rad)")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.legend(fontsize=8)


def plot_theta_entropy_map(
    states: np.ndarray, world_bounds: tuple, ax: Axes,
    n_xy_bins: int = 20, n_theta_bins: int = 12, min_samples: int = 10,
) -> None:
    """Heatmap of heading spread (entropy) for each (x, y) bin.

    For every spatial bin, compute the entropy of the headings observed there.
    High and uniform means the robot has faced all directions everywhere. Bins
    with fewer than `min_samples` samples are left blank. Max entropy is
    `log(n_theta_bins)` (perfectly uniform headings).
    """
    (x_min, x_max), (y_min, y_max) = world_bounds
    x_edges = np.linspace(x_min, x_max, n_xy_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_xy_bins + 1)
    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)

    theta = wrap_angle(states[:, 2])
    H, _ = np.histogramdd(
        np.stack([states[:, 0], states[:, 1], theta], axis=-1),
        bins=[x_edges, y_edges, theta_edges],
    )  # shape (n_x, n_y, n_theta)
    counts_xy = H.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = H / counts_xy[..., None]
        p_log_p = np.where(p > 0, p * np.log(p), 0.0)
        entropy = -p_log_p.sum(axis=-1)
    entropy = np.where(counts_xy >= min_samples, entropy, np.nan)

    # entropy is (n_x, n_y). Transpose for imshow so rows index y and origin
    # "lower" puts low-y at the bottom.
    max_h = np.log(n_theta_bins)
    im = ax.imshow(
        entropy.T, extent=(x_min, x_max, y_min, y_max), origin="lower",
        vmin=0, vmax=max_h, cmap="viridis", interpolation="nearest",
    )
    ax.set_title("heading coverage by location")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"heading entropy (max log {n_theta_bins} ≈ {max_h:.2f})")


# ## Transition + episode-length plots
# 
# `Δθ` and `Δposition` are measured within episodes only and angle-wrapped. Their spread should sit around `ω_std·dt` and `v_mean·dt`. If `Δθ` collapses to a spike at 0 the frames are too redundant (raise `dt` or `hold_k`).
# 
# The episode-length histogram is stacked by termination. Mostly walls is good; if timeouts dominate, `max_steps` is too low or the robot can't reach the walls.

# In[ ]:


def plot_dtheta(d_theta: np.ndarray, attrs: Mapping, ax: Axes) -> None:
    """Within-episode per-step heading change."""
    expected_std = float(attrs["omega_std"]) * float(attrs["dt"])
    ax.hist(d_theta, bins=50, density=True, alpha=0.6, color="steelblue")
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_title(f"per-step turn (Δθ),  σ ≈ {expected_std:.3f}")
    ax.set_xlabel("Δθ (rad)")


def plot_dpos(d_pos: np.ndarray, attrs: Mapping, ax: Axes) -> None:
    """Within-episode per-step distance moved."""
    expected_mean = float(attrs["v_mean"]) * float(attrs["dt"])
    ax.hist(d_pos, bins=50, density=True, alpha=0.6, color="steelblue")
    ax.axvline(expected_mean, color="r", ls="--", lw=1,
               label=f"v_mean·dt = {expected_mean:.3f}")
    ax.set_title("per-step distance moved")
    ax.set_xlabel("|Δposition|")
    ax.legend(fontsize=8)


def plot_episode_lengths(
    episode_lengths: np.ndarray, termination: np.ndarray, attrs: Mapping, ax: Axes
) -> None:
    """Stacked histogram of episode lengths, split by how the episode ended."""
    walls = episode_lengths[termination == 0]
    timeouts = episode_lengths[termination == 1]
    min_length = int(attrs["min_length"])
    max_steps = int(attrs["max_steps"])

    bins = np.linspace(min_length, max(max_steps, episode_lengths.max() + 1), 30)
    ax.hist(
        [walls, timeouts], bins=bins, stacked=True,
        label=[f"wall ({len(walls)})", f"timeout ({len(timeouts)})"],
        color=["steelblue", "orange"], alpha=0.8,
    )
    median = int(np.median(episode_lengths))
    ax.axvline(median, color="k", ls="--", lw=1, label=f"median = {median}")
    ax.axvline(min_length, color="red", ls=":", lw=1, label=f"min_length = {min_length}")
    ax.set_title("episode lengths")
    ax.set_xlabel("length (steps)")
    ax.legend(fontsize=8)


# ## Summary + entry point
# 
# `print_summary` is the quick numeric readout. `plot_coverage` is the one to call: give it a dataset path and it prints the summary, builds the 6-plot panel, and writes the panel plus each plot's own PNG into `<name>_coverage/` next to the dataset.

# In[ ]:


def print_summary(h5) -> None:
    """Print a short numeric summary of the dataset (an open h5py.File)."""
    lens = h5["episode_lengths"][:]
    term = h5["termination"][:]
    actions = h5["actions"][:]
    n_walls = int((term == 0).sum())
    n_timeouts = int((term == 1).sum())

    print(f"episodes:       {len(lens)} kept of {int(h5.attrs.get('n_episodes_attempted', -1))} attempted")
    print(f"transitions:    {int(h5.attrs.get('total_transitions', actions.shape[0]))}")
    print(f"length stats:   min={lens.min()}  median={int(np.median(lens))}  mean={lens.mean():.1f}  max={lens.max()}")
    print(f"terminations:   {n_walls} wall, {n_timeouts} timeout")
    print(f"v     empirical: mean={actions[:, 0].mean():.3f}  std={actions[:, 0].std():.3f}  "
          f"(attrs: {float(h5.attrs['v_mean']):.3f}, {float(h5.attrs['v_std']):.3f})")
    print(f"omega empirical: mean={actions[:, 1].mean():.3f}  std={actions[:, 1].std():.3f}  "
          f"(attrs: {float(h5.attrs['omega_mean']):.3f}, {float(h5.attrs['omega_std']):.3f})")


def plot_coverage(h5_path: Union[str, Path], save_dir: Optional[Union[str, Path]] = None,
                  show: bool = True) -> Figure:
    """Read a dataset and build its coverage diagnostics.

    Prints the summary, builds a 6-plot panel, and saves the panel plus each
    individual plot as its own PNG.

    Parameters
    ----------
    h5_path : str or Path
        Input .h5 file (from `collect_dataset`).
    save_dir : str, Path, or None
        Output folder. If None, uses `<h5_parent>/<stem>_coverage/`.
    show : bool
        Call `plt.show()` at the end (otherwise the figure is closed).

    Returns
    -------
    matplotlib Figure
        The summary panel.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"no such file: {h5_path}")
    if save_dir is None:
        save_dir = h5_path.parent / f"{h5_path.stem}_coverage"
    save_dir = Path(save_dir)

    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        states = f["states"][:]
        actions = f["actions"][:]
        episode_starts = f["episode_starts"][:]
        episode_lengths = f["episode_lengths"][:]
        termination = f["termination"][:]

        print(f"=== {h5_path.name} ===")
        print_summary(f)
        print()

    world_bounds = tuple(map(tuple, np.asarray(attrs["world_bounds"])))
    d_theta, d_pos = per_step_diffs(states, episode_starts, episode_lengths)

    # (filename, draw-on-an-axis). The first six make up the panel; the last
    # two are saved as standalone images only.
    panel_plots = [
        ("velocity",         lambda ax: plot_v_hist(actions, attrs, ax)),
        ("turn_rate",        lambda ax: plot_omega_hist(actions, attrs, ax)),
        ("heading_coverage", lambda ax: plot_theta_entropy_map(states, world_bounds, ax)),
        ("step_turn",        lambda ax: plot_dtheta(d_theta, attrs, ax)),
        ("step_distance",    lambda ax: plot_dpos(d_pos, attrs, ax)),
        ("episode_lengths",  lambda ax: plot_episode_lengths(episode_lengths, termination, attrs, ax)),
    ]
    extra_plots = [
        ("xy_occupancy",     lambda ax: plot_xy_hist(states, world_bounds, ax)),
        ("heading_hist",     lambda ax: plot_theta_marginal(states, ax)),
    ]

    # Summary panel: the six core plots in a 2x3 grid.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for (_, draw), ax in zip(panel_plots, axes.flat):
        draw(ax)
    fig.suptitle(f"{h5_path.name}  coverage", fontsize=14, y=1.0)
    fig.tight_layout()

    # Save the panel and each plot on its own.
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "panel.png", dpi=120, bbox_inches="tight")
    for name, draw in panel_plots + extra_plots:
        f1, ax1 = plt.subplots(figsize=(5, 4))
        draw(ax1)
        f1.tight_layout()
        f1.savefig(save_dir / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(f1)
    print(f"saved panel + {len(panel_plots) + len(extra_plots)} plots -> {save_dir}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ## Demo
# 
# Runs on the dataset `collect.ipynb` just produced. Guarded with `if __name__ == "__main__":` so it doesn't fire when this notebook is exported and imported elsewhere.

# In[ ]:


if __name__ == "__main__":
    plot_coverage(DATA_PATH)

