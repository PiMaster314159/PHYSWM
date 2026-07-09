"""Model Predictive Path Integral (MPPI) planner, numpy, model-agnostic.

One planning step: sample K action sequences around a nominal, roll them through `dynamics`,
score each with `running_cost` + `terminal_cost`, softmax-weight by cost, and return the new
nominal sequence. The caller applies its first action, advances the real system, warm-starts
(shift the nominal), and calls again -> receding-horizon MPC.

`dynamics` and the costs are VECTORIZED over the K samples so a full plan is a handful of numpy
ops. Swap `dynamics` for the neural world model's step later; the planner does not change.
"""
import numpy as np


def mppi_plan(s0, a_nom, dynamics, running_cost, terminal_cost=None,
              n_samples=1000, sigma=0.5, lam=1.0, a_low=None, a_high=None, rng=None):
    """
    s0           : (d,)      current state
    a_nom        : (H, m)    nominal action sequence (warm start)
    dynamics     : (K,d),(K,m) -> (K,d)     one vectorized step
    running_cost : (K,d),(K,m) -> (K,)      per-step cost
    terminal_cost: (K,d) -> (K,)            optional cost on the final state
    returns (a_new (H,m), info dict)
    """
    rng = np.random.default_rng() if rng is None else rng
    H, m = a_nom.shape
    K = n_samples

    eps = rng.normal(0.0, sigma, size=(K, H, m))
    A = a_nom[None] + eps                                  # (K, H, m)
    if a_low is not None:
        A = np.clip(A, a_low, a_high)

    S = np.tile(np.asarray(s0, float), (K, 1))             # (K, d)
    J = np.zeros(K)
    for t in range(H):
        J += running_cost(S, A[:, t])
        S = dynamics(S, A[:, t])
    if terminal_cost is not None:
        J += terminal_cost(S)

    w = np.exp(-(J - J.min()) / lam)                       # softmax weights (min-shift for stability)
    w /= w.sum() + 1e-12
    a_new = (w[:, None, None] * A).sum(axis=0)             # (H, m) weighted-average sequence
    return a_new, {"J_min": float(J.min()), "J_mean": float(J.mean()), "w_max": float(w.max())}
