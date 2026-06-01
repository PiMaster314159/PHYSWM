#!/usr/bin/env python
# coding: utf-8

# # Visualization
# 
# Matplotlib helpers for eyeballing frames. Human sanity-checking only. The model never sees matplotlib output, only raw binary frames from `render_frame`.
# 
# Good sanity check: before generating thousands of episodes, render a few frames at your training resolution, blow them up with nearest-neighbor interpolation, and confirm by eye that the triangle points the way theta says it should.
# 
# > To generate `visualize.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python sim/visualize.ipynb
# > ```

# ## Imports

# In[ ]:


import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Sequence

from constants import WORLD_BOUNDS


# ## Input-validation helper
# 
# Convert frame to numpy array, check it is 2D.

# In[ ]:


def _as_frame(frame: npt.ArrayLike) -> np.ndarray:
    """Validate and convert a frame to a 2D numpy array.

    Parameters
    ----------
    frame : array-like
        A binary frame.

    Returns
    -------
    np.ndarray, shape (H, W)
        Validated 2D array.

    Raises
    ------
    ValueError
        If the frame is not 2D.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {frame.shape}")
    return frame


# ## Single frame
# 
# Draw one binary frame. Nearest-neighbor interpolation so individual cells stay visible at low resolutions like 40x40.

# In[ ]:


def show_frame(
    frame: npt.ArrayLike,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Display one binary frame, blown up with nearest-neighbor interpolation.

    Parameters
    ----------
    frame : array-like, shape (H, W)
        A frame from `render_frame`.
    ax : matplotlib Axes, optional
        Where to draw. If None, a new figure is created.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib Axes
        The axis the frame was drawn on.
    """
    frame = _as_frame(frame)

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    ax.imshow(frame, cmap="gray", interpolation="nearest")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ## Strip of frames
# 
# Frames side-by-side. Good for spot-checking a single rolled-out episode.

# In[ ]:


def show_frame_strip(
    frames: npt.ArrayLike,
    titles: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
) -> Figure:
    """Display a sequence of frames side-by-side.

    Parameters
    ----------
    frames : sequence of (H, W) arrays, or (T, H, W) array
        Frames to display.
    titles : sequence of str, optional
        One title per frame. Defaults to "t=0", "t=1", ...
    figsize : (width, height), optional
        Defaults to (2.5 * T, 2.5).

    Returns
    -------
    matplotlib Figure
        The figure holding the strip.
    """
    frames = [_as_frame(f) for f in frames]
    n = len(frames)
    if n == 0:
        raise ValueError("frames is empty")
    if titles is not None and len(titles) != n:
        raise ValueError(f"titles has length {len(titles)}, expected {n}")

    if figsize is None:
        figsize = (2.5 * n, 2.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        title = titles[i] if titles is not None else f"t={i}"
        show_frame(frame, ax=ax, title=title)
    fig.tight_layout()
    return fig


# ## Trajectory plot
# 
# Plot continuous `(x, y, theta)` in world coordinates, with arrows for heading at each step and a red box for the world bounds. Useful for debugging episode collection (e.g. confirming an `env_step` loop drove the robot as expected before termination).
# 
# Works on the continuous state, not on rendered frames. The model never sees this view.

# In[ ]:


def show_trajectory(
    states: npt.ArrayLike,
    world_bounds: tuple = WORLD_BOUNDS,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    arrow_every: int = 1,
    figsize: tuple = (4, 4),
) -> Axes:
    """Plot a continuous (x, y, theta) trajectory with heading arrows.

    Path drawn in (x, y) world coordinates as a faint black line; an arrow at
    each step shows heading. World bounds drawn as a red box so you can see how
    close the robot gets to the walls.

    Debugging tool only. This is the answer key the model never sees.

    Parameters
    ----------
    states : array-like, shape (T, 3)
        Sequence of (x, y, theta) states, e.g. from `step_rollout` or an
        `env_step` loop.
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        Drawn as a red rectangle. Defaults to the unit square.
    ax : matplotlib Axes, optional
        Where to draw. If None, a new figure is created at `figsize`.
    title : str, optional
        Plot title.
    arrow_every : int
        Draw a heading arrow every Nth state. Bump this up if the plot gets
        cluttered for long trajectories.
    figsize : (width, height)
        Used only when `ax` is None.

    Returns
    -------
    matplotlib Axes
        The axis the trajectory was drawn on.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError(f"states must have shape (T, 3), got {states.shape}")
    if arrow_every < 1:
        raise ValueError(f"arrow_every must be >= 1, got {arrow_every}")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    (x_min, x_max), (y_min, y_max) = world_bounds

    # Path.
    ax.plot(states[:, 0], states[:, 1], "k-", alpha=0.4, lw=1)

    # Heading arrows (subsampled if requested).
    arrows = states[::arrow_every]
    ax.quiver(
        arrows[:, 0], arrows[:, 1],
        np.cos(arrows[:, 2]), np.sin(arrows[:, 2]),
        angles="xy", scale_units="xy", scale=20, width=0.005,
    )

    # World-bounds box.
    ax.add_patch(plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="red", lw=1,
    ))

    # 5% padding around the world box so the arrows aren't clipped.
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)
    return ax


# ## Tests
# 
# Tests only confirm the helpers don't crash on real frames. Graphs are a visual sanity check, not assertions.

# In[ ]:


def _test_visualize():
    """Smoke test. Confirms the helpers don't crash on real frames."""
    from sim.render import render_frame

    frame = render_frame([0.5, 0.5, 0.0], grid_size=32)
    ax = show_frame(frame, title="single")
    assert ax is not None
    plt.close("all")

    frames = [render_frame([0.5, 0.5, t], grid_size=32)
              for t in np.linspace(0, np.pi, 4)]
    fig = show_frame_strip(frames)
    assert fig is not None
    plt.close("all")

    # Title-length mismatch should complain.
    try:
        show_frame_strip(frames, titles=["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched titles length")

    # show_trajectory smoke test.
    states = np.array([
        [0.2, 0.5, 0.0],
        [0.3, 0.5, 0.1],
        [0.4, 0.55, 0.2],
        [0.5, 0.6, 0.3],
    ])
    ax = show_trajectory(states, title="trajectory test")
    assert ax is not None
    plt.close("all")

    # Bad shape should complain.
    try:
        show_trajectory(np.zeros((4, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong-shape states")

    print("All visualize tests passed.")


# In[14]:


if __name__ == "__main__":
    _test_visualize()

