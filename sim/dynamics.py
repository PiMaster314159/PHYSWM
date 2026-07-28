# # Dynamics
# 
# First-order kinematics simulator. Advances robot state `(x, y, theta)` given action `(v, omega)`. No inertia.
# 
# Semi-implicit Euler: theta updated first, then position uses the new theta. More stable than forward Euler.
# 
# Equations:
# - `theta_{t+1} = wrap(theta_t + omega * dt)` (bounded to `[-pi, pi)`)
# - `x_{t+1}     = x_t + v * cos(theta_{t+1}) * dt`
# - `y_{t+1}     = y_t + v * sin(theta_{t+1}) * dt`
# 

# # Imports


import numpy as np
import numpy.typing as npt
from config import DT, WHEELBASE


# ## Angle wrapping
# 
# Bounds angle to `[-pi, pi)`. Wraps if outside range.


def wrap_theta(theta: float) -> float:
    """Wrap an angle (radians) into [-pi, pi).

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in [-pi, pi).
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


# ## Input-validation helpers
# 
# Convert array-like input to numpy float array. Validates shape and finiteness.


def _as_state(state: npt.ArrayLike) -> np.ndarray:
    """Validate and convert state to a float numpy array.

    Parameters
    ----------
    state : array-like
        (x, y, theta) in world coordinates.

    Returns
    -------
    np.ndarray, shape (3,)
        Validated float array.

    Raises
    ------
    ValueError
        If shape is not (3,) or values are not finite.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (3,):
        raise ValueError(f"state must have shape (3,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"state must be finite, got {state}")
    return state


def _as_action(action: npt.ArrayLike) -> np.ndarray:
    """Validate and convert action to a float numpy array.

    Parameters
    ----------
    action : array-like
        (v, omega).

    Returns
    -------
    np.ndarray, shape (2,)
        Validated float array.

    Raises
    ------
    ValueError
        If shape is not (2,) or values are not finite.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"action must be finite, got {action}")
    return action


# # Single Step Dynamics


def step(state: npt.ArrayLike, action: npt.ArrayLike, dt: float = DT) -> np.ndarray:
    """Advance the robot state by one timestep.

    Parameters
    ----------
    state : array-like, shape (3,)
        (x, y, theta) in world coordinates; theta in radians.
    action : array-like, shape (2,)
        (v, omega). v is linear velocity (world units/s), omega is angular velocity (rad/s).
    dt : float
        Timestep size.

    Returns
    -------
    np.ndarray, shape (3,)
        (x, y, theta) after one step, with theta wrapped to [-pi, pi).
    """
    state = _as_state(state)
    action = _as_action(action)

    x, y, theta = state
    v, omega = action

    # Wrap theta in case input is outside range [-pi, pi).
    theta = wrap_theta(theta)
    theta_new = wrap_theta(theta + omega * dt)
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt
    return np.array([x_new, y_new, theta_new])


# # Bicycle / throttle dynamics
#
# Kinematic bicycle with a throttle. Unlike the unicycle, VELOCITY IS A STATE (it integrates the throttle),
# so state grows to (x, y, theta, v) and the action becomes (a, delta) = (acceleration, steering angle).
# Because v is a hidden state, it is NOT visible in a single rendered frame -> the model must infer it from
# the last few frames (history). At v=0 the car cannot turn (yaw rate scales with speed), a real bicycle trait.
#
#   v_new     = clip(v + (gain*a - drag_c*v^2) * dt, v_min, v_max)   throttle: your v_dot = a, + optional aero drag
#   theta_new = wrap(theta + (v_new / L) * tan(delta) * dt)          steering: yaw rate = v/L * tan(delta)
#   x_new     = x + v_new * cos(theta_new) * dt
#   y_new     = y + v_new * sin(theta_new) * dt
# Semi-implicit (v -> theta -> position), matching the unicycle step.


def bicycle_step(state: npt.ArrayLike, action: npt.ArrayLike, dt: float = DT,
                 wheelbase: float = WHEELBASE, drag_c: float = 0.0, gain: float = 1.0,
                 v_min: float = 0.0, v_max: float = None) -> np.ndarray:
    """One kinematic-bicycle-with-throttle step.

    state  : (x, y, theta, v)         velocity v is a STATE (integrates the throttle)
    action : (a, delta)               a = commanded acceleration (throttle), delta = steering angle (rad)
    drag_c : v_dot = gain*a - drag_c*v^2   (drag_c=0 -> your prof's clean v_dot = a; >0 -> aero drag on velocity)
    Returns (x, y, theta, v) with theta wrapped to [-pi, pi).
    """
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.shape != (4,):
        raise ValueError(f"bicycle state must have shape (4,) = (x,y,theta,v), got {state.shape}")
    if action.shape != (2,):
        raise ValueError(f"bicycle action must have shape (2,) = (a,delta), got {action.shape}")

    x, y, theta, v = state
    a, delta = action
    v_new = v + (gain * a - drag_c * v * v) * dt          # throttle -> velocity (semi-implicit: use v_new below)
    v_new = max(v_min, v_new) if v_min is not None else v_new
    v_new = min(v_max, v_new) if v_max is not None else v_new
    theta_new = wrap_theta(theta + (v_new / wheelbase) * np.tan(delta) * dt)   # yaw rate scales with speed
    x_new = x + v_new * np.cos(theta_new) * dt
    y_new = y + v_new * np.sin(theta_new) * dt
    return np.array([x_new, y_new, theta_new, v_new])


# ## Multi-step rollout
#
# Convenience wrapper. Takes a sequence of actions and returns the full state trajectory, shape `(T+1, 3)`.


def step_rollout(initial_state: npt.ArrayLike, actions: npt.ArrayLike, dt: float = DT) -> np.ndarray:
    """Execute a sequence of actions from an initial state.

    Parameters
    ----------
    initial_state : array-like, shape (3,)
        Starting (x, y, theta).
    actions : array-like, shape (T, 2)
        Sequence of (v, omega) actions.
    dt : float
        Timestep size.

    Returns
    -------
    np.ndarray, shape (T+1, 3)
        Initial state followed by the state after each action.
    """
    initial_state = _as_state(initial_state)
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape (T, 2), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must be finite")

    states = [initial_state]
    for action in actions:
        states.append(step(states[-1], action, dt))
    return np.array(states)


# ## Tests
# 
# Sanity tests for the dynamics.


def _test_step_dynamics():
    """Sanity tests for the dynamics."""
    # Facing right, move forward.
    next_state = step([0.5, 0.5, 0.0], [1.0, 0.0], dt=0.1)
    expected = np.array([0.6, 0.5, 0.0])
    assert np.allclose(next_state, expected), f"Expected {expected}, got {next_state}"

    # Facing up, move forward.
    next_state = step([0.5, 0.5, np.pi / 2], [1.0, 0.0], dt=0.1)
    expected = np.array([0.5, 0.6, np.pi / 2])
    assert np.allclose(next_state, expected), f"Expected {expected}, got {next_state}"

    # Rotate in place.
    next_state = step([0.5, 0.5, 0.0], [0.0, 2.0], dt=0.1)
    expected = np.array([0.5, 0.5, 0.2])
    assert np.allclose(next_state, expected), f"Expected {expected}, got {next_state}"

    # Rollout with constant forward velocity.
    actions = np.array([[1.0, 0.0]] * 3)
    states = step_rollout([0.0, 0.0, 0.0], actions, dt=0.1)
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])
    assert np.allclose(states, expected), f"Expected {expected}, got {states}"

    # Testing defensive wrap - an unwrapped input theta should still produce wrapped output.
    next_state = step([0.0, 0.0, 3 * np.pi], [0.0, 0.0], dt=0.1)
    assert -np.pi <= next_state[2] < np.pi, f"theta not wrapped: {next_state[2]}"

    print("All dynamics tests passed.")


if __name__ == "__main__":
    _test_step_dynamics()

