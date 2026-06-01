#!/usr/bin/env python
# coding: utf-8

# # Renderer
# 
# Binary occupancy grid renderer. Robot drawn as isosceles triangle (altitude `L`, base `W`) so heading is visually distinguishable.
# 
# Resolution-agnostic: computes world coords of each grid cell's center, tests inclusion via half-plane (sign-of-cross-product) test. No fixed pixel sprite.
# 
# World convention: row 0 is upper `y` boundary, col 0 is left `x` boundary.
# 
# > To generate `render.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python sim/render.ipynb
# > ```

# ## Imports & triangle constants
# 
# Triangle dimensions `L`, `W` and `WORLD_BOUNDS` come from the project-wide `constants.py` (single source for the whole project).

# In[ ]:


import numpy as np
import numpy.typing as npt

from constants import L, W, WORLD_BOUNDS


# ## Triangle in local coordinates
# 
# Triangle built once in the robot's body frame, centered on its centroid. `(x, y)` in world coordinates maps to the geometric center of the shape.

# In[7]:


def make_triangle(L: float = L, W: float = W) -> np.ndarray:
    """Triangle polygon in local (body-frame) coordinates, centered on centroid.

    Parameters
    ----------
    L : float
        Triangle altitude (length) in world units.
    W : float
        Triangle base width in world units.

    Returns
    -------
    np.ndarray, shape (3, 2)
        Vertices with tip toward +x, base spanning +-W/2 along y.
    """
    vertices = np.array([
        [L,   0.0],
        [0.0,  W / 2],
        [0.0, -W / 2],
    ])
    return vertices - vertices.mean(axis=0)


# ## Coordinate transforms
# 
# Converts world coordinates to the robot's local (body) frame. Cheaper than rotating the triangle every call: transform the grid points into body frame once, test against the untransformed triangle.

# In[8]:


def world_to_triangle_coords(points: np.ndarray, state: npt.ArrayLike) -> np.ndarray:
    """Map world-frame points into the robot's local (body) frame.

    Parameters
    ----------
    points : np.ndarray, shape (..., 2)
        World-coordinate points to transform.
    state : array-like, shape (3,)
        (x, y, theta).

    Returns
    -------
    np.ndarray, shape (..., 2)
        Points in the robot's body frame.
    """
    x, y, theta = state
    # Rotation by -theta = transpose of CCW rotation by +theta.
    R_inv = np.array([
        [ np.cos(theta),  np.sin(theta)],
        [-np.sin(theta),  np.cos(theta)],
    ])
    return (points - np.array([x, y])) @ R_inv.T


# ## Point-in-triangle test
# 
# Half-plane test: a point is inside the triangle iff the signs of all three edge cross products agree. Handles both winding orders so vertex ordering from `make_triangle` does not matter.

# In[9]:


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D scalar cross product a x b, vectorized.

    Parameters
    ----------
    a, b : np.ndarray, shape (..., 2)

    Returns
    -------
    np.ndarray, shape (...)
        Scalar cross product at each position.
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# In[10]:


def points_in_triangle(points: np.ndarray, triangle_vertices: np.ndarray) -> np.ndarray:
    """Test whether each point lies inside (or on the edge of) the triangle.

    For each edge, the sign of the cross product indicates which side the point
    is on. A point is inside iff all three signs agree. Handles both winding orders.

    Parameters
    ----------
    points : np.ndarray, shape (..., 2)
    triangle_vertices : np.ndarray, shape (3, 2)

    Returns
    -------
    np.ndarray of bool, shape (...)
    """
    a, b, c = triangle_vertices
    ab = b - a
    bc = c - b
    ca = a - c

    cross1 = cross2d(ab, points - a)
    cross2 = cross2d(bc, points - b)
    cross3 = cross2d(ca, points - c)

    all_nonneg = (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0)
    all_nonpos = (cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0)
    return all_nonneg | all_nonpos


# ## Grid cell centers
# 
# Computes world `(x, y)` at each cell center of a `grid_size x grid_size` image. Used in `render_frame` to test which cells fall inside the triangle.

# In[ ]:


def grid_cell_centers(
    grid_size: int,
    world_bounds: tuple = WORLD_BOUNDS,
) -> np.ndarray:
    """World-coordinate centers of every cell in a (grid_size x grid_size) image.

    Parameters
    ----------
    grid_size : int
        Number of cells per side.
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.

    Returns
    -------
    np.ndarray, shape (grid_size, grid_size, 2)
        centers[i, j] = (x, y) world coords of cell at row i, col j.
        Row 0 corresponds to high y; col 0 to low x.
    """
    (x_min, x_max), (y_min, y_max) = world_bounds
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size

    x_centers = x_min + (np.arange(grid_size) + 0.5) * dx
    y_centers = y_max - (np.arange(grid_size) + 0.5) * dy

    xx, yy = np.meshgrid(x_centers, y_centers)
    return np.stack([xx, yy], axis=-1)


# ## Input-validation helper
# 
# Reuse `_as_state` from `dynamics.py` rather than duplicating validation logic.

# In[ ]:


from sim.dynamics import _as_state


# ## Main render function
# 
# Core of the module: converts a robot state to a binary occupancy frame. This is what the model's encoder receives as input.

# In[ ]:


def render_frame(
    state: npt.ArrayLike,
    grid_size: int = 64,
    world_bounds: tuple = WORLD_BOUNDS,
    L: float = L,
    W: float = W,
    dtype: npt.DTypeLike = np.uint8,
) -> np.ndarray:
    """Render the robot at `state` to a binary (grid_size x grid_size) frame.

    Cells whose centers fall inside the rotated/translated triangle are 1,
    everything else is 0.

    Parameters
    ----------
    state : array-like, shape (3,)
        (x, y, theta).
    grid_size : int
        Image side length in pixels (resolution knob).
    world_bounds : tuple, ((x_min, x_max), (y_min, y_max))
        World extents.
    L, W : float
        Triangle length and base width in world units.
    dtype : dtype-like
        Output array dtype (default uint8).

    Returns
    -------
    np.ndarray, shape (grid_size, grid_size)
        Binary occupancy frame.
    """
    state = _as_state(state)

    triangle_local = make_triangle(L=L, W=W)
    centers_world = grid_cell_centers(grid_size, world_bounds=world_bounds)
    centers_local = world_to_triangle_coords(centers_world, state)

    inside = points_in_triangle(centers_local, triangle_local)
    return inside.astype(dtype)


# ## Vertices in world coordinates
# 
# Helper for matplotlib overlays (e.g. triangle outline on a debug plot). Not used by the renderer itself.

# In[14]:


def triangle_world_vertices(state: npt.ArrayLike, L: float = L, W: float = W) -> np.ndarray:
    """Triangle vertices in world coordinates.

    Useful for matplotlib overlays. Not used by the renderer itself.

    Parameters
    ----------
    state : array-like, shape (3,)
        (x, y, theta).
    L, W : float
        Triangle length and base width in world units.

    Returns
    -------
    np.ndarray, shape (3, 2)
        Triangle vertices in world coordinates.
    """
    state = _as_state(state)

    x, y, theta = state
    triangle_local = make_triangle(L=L, W=W)
    R = np.array([
        [ np.cos(theta), -np.sin(theta)],
        [ np.sin(theta),  np.cos(theta)],
    ])
    return triangle_local @ R.T + np.array([x, y])


# ## Tests
# 
# Sanity tests for the renderer.

# In[15]:


def _test_renderer():
    """Sanity tests for the renderer."""
    state = np.array([0.5, 0.5, 0.0])
    frame = render_frame(state, grid_size=64)
    assert frame.shape == (64, 64), f"Unexpected shape: {frame.shape}"
    assert np.all((frame == 0) | (frame == 1)), "Frame is not binary."
    assert frame.sum() > 0, "Triangle rendered with zero occupied pixels."

    # Different orientations should generally produce different binary patterns.
    frame2 = render_frame([0.5, 0.5, np.pi / 2], grid_size=64)
    assert frame2.sum() > 0, "Rotated triangle rendered with zero occupied pixels."
    assert not np.array_equal(frame, frame2), "0 and 90 degree renders should differ."

    verts = triangle_world_vertices(state)
    assert verts.shape == (3, 2), f"Unexpected vertex shape: {verts.shape}"

    print("All renderer tests passed.")


# In[16]:


if __name__ == "__main__":
    _test_renderer()

