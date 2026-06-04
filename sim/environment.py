#!/usr/bin/env python
# coding: utf-8

# # Environment: boundary detection + `env_step`
# 
# `env_step` wraps the dynamics step and adds a `done` flag. Decides when an episode ends. The data-collection loop calls `env_step` until `done` is True.
# 
# Boundary rule: episode ends as soon as the robot's bounding circle (centroid + radius) crosses any world edge.
# 
# > To generate `environment.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python sim/environment.ipynb
# > ```

# ## Imports
# 
# `step` and `_as_state` from `sim.dynamics`; the fixed constants `DT`, `L`, `W`, `WORLD_BOUNDS` from `config.py`.
# 
# Run from the `PHYSWM` root so both `config` and `sim.*` resolve.

# In[ ]:


import numpy as np
import numpy.typing as npt
from typing import Optional

from config import DT, L, W, WORLD_BOUNDS
from sim.dynamics import step, _as_state


# ## Bounding-circle radius
# 
# Max distance from the centered triangle's centroid to any of its vertices.
# 
# After centering in `make_triangle`, the tip sits at `+2L/3` along local-x and the back corners at `(-L/3, ±W/2)`. Whichever is farther bounds the triangle from the outside. With `L=0.12, W=0.06` this is `0.08`.

# In[ ]:


def bounding_radius(L: float = L, W: float = W) -> float:
    """Max distance from the centered triangle's centroid to any vertex.

    Parameters
    ----------
    L : float
        Triangle altitude (length) in world units.
    W : float
        Triangle base width in world units.

    Returns
    -------
    float
        Bounding-circle radius.
    """
    return float(max(2 * L / 3, np.sqrt((L / 3) ** 2 + (W / 2) ** 2)))


# ## Boundary check
# 
# `out_of_bounds` is `True` iff the bounding circle crosses any world edge. Heading-independent by construction (only looks at `x`, `y`).

# In[ ]:


def out_of_bounds(
    state: npt.ArrayLike,
    world_bounds: tuple = WORLD_BOUNDS,
    margin: Optional[float] = None,
    L: float = L,
    W: float = W,
) -> bool:
    """True iff the robot's bounding circle crosses any world edge.

    Centroid + margin is a deliberate over-approximation: it triggers `done`
    as soon as any part of the triangle could exit the arena, regardless of
    heading.

    Parameters
    ----------
    state : array-like, shape (3,)
        (x, y, theta).
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.
    margin : float or None
        If None, defaults to `bounding_radius(L, W)`.
    L, W : float
        Triangle length/base, used only when `margin` is None.

    Returns
    -------
    bool
        True if out of bounds (episode should end).
    """
    state = _as_state(state)

    if margin is None:
        margin = bounding_radius(L=L, W=W)

    (x_min, x_max), (y_min, y_max) = world_bounds
    x, y, _ = state
    return bool(
        x - margin < x_min
        or x + margin > x_max
        or y - margin < y_min
        or y + margin > y_max
    )


# ## `env_step`
# 
# Wrapper around `step` + `out_of_bounds`. The data-collection loop should never have to know the boundary rule directly.

# In[ ]:


def env_step(
    state: npt.ArrayLike,
    action: npt.ArrayLike,
    dt: float = DT,
    world_bounds: tuple = WORLD_BOUNDS,
    margin: Optional[float] = None,
) -> tuple[np.ndarray, bool]:
    """Advance the robot one step and report whether the episode ended.

    Parameters
    ----------
    state : array-like, shape (3,)
        (x, y, theta).
    action : array-like, shape (2,)
        (v, omega).
    dt : float
        Timestep size.
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.
    margin : float or None
        Boundary margin. If None, defaults to the triangle's bounding radius.

    Returns
    -------
    next_state : np.ndarray, shape (3,)
        State after one step.
    done : bool
        True if the new state is out of bounds.
    """
    next_state = step(state, action, dt=dt)
    done = out_of_bounds(next_state, world_bounds=world_bounds, margin=margin)
    return next_state, done


# ## Tests
# 
# Sanity tests for boundary detection.

# In[5]:


def _test_environment():
    """Sanity tests for boundary detection."""
    r = bounding_radius()
    assert 0 < r < 0.5, f"bounding_radius looks wrong: {r}"

    # Center of arena: well inside.
    assert not out_of_bounds([0.5, 0.5, 0.0])

    # Far past every edge.
    assert out_of_bounds([1.5, 0.5, 0.0])
    assert out_of_bounds([-0.5, 0.5, 0.0])
    assert out_of_bounds([0.5, 1.5, 0.0])
    assert out_of_bounds([0.5, -0.5, 0.0])

    # Centroid right on the wall: out (margin pushes us past).
    assert out_of_bounds([1.0, 0.5, 0.0])
    assert out_of_bounds([0.0, 0.5, 0.0])

    # Heading should not affect the bounding-circle test.
    for theta in np.linspace(-np.pi, np.pi, 9):
        assert out_of_bounds([1.0, 0.5, theta]) == out_of_bounds([1.0, 0.5, 0.0])

    # Stepping into a wall flips done True.
    _, done = env_step([0.95, 0.5, 0.0], [1.0, 0.0], dt=0.1)
    assert done, "Expected done after stepping into the right wall"

    # Stepping in the middle leaves done False.
    _, done = env_step([0.5, 0.5, 0.0], [1.0, 0.0], dt=0.1)
    assert not done

    # Walking straight into a wall must terminate within a reasonable horizon.
    state = np.array([0.5, 0.5, 0.0])
    action = np.array([1.0, 0.0])
    hit = False
    for _ in range(50):
        state, done = env_step(state, action)
        if done:
            hit = True
            break
    assert hit, "Robot driving forward should have hit the right wall"

    print("All environment tests passed.")


# In[6]:


if __name__ == "__main__":
    _test_environment()

