import torch
import torch.nn as nn
import torch.nn.functional as F

def sigreg_loss(
    z: torch.Tensor,
    n_projections: int = 1024,
    n_quad: int = 17,
    t_max: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sketched-Isotropic-Gaussian Regularizer (SIGReg).

    Projects z onto n_projections random unit directions and applies the
    univariate Epps-Pulley normality test to each projection.

    SIGReg(Z) -> 0 iff P_Z -> N(0, I)  (Cramer-Wold theorem).

    Matches the reference le-wm implementation: quadrature on [0, 3] with 17
    knots, 1024 projections, statistic scaled by batch size B.

    Parameters
    ----------
    z : (B, D)
    n_projections : int
        Number of random unit directions. Reference uses 1024.
    n_quad : int
        Quadrature nodes. Reference uses 17 on [0, 3].
    t_max : float
        Upper integration limit. Reference uses 3.0.
    eps : float
        Small constant for direction normalization.
    """
    B, D = z.shape

    t  = torch.linspace(0.0, t_max, n_quad, device=z.device, dtype=z.dtype)
    dt = t_max / (n_quad - 1)
    w  = torch.full((n_quad,), 2.0 * dt, device=z.device, dtype=z.dtype)
    w[0] = dt;  w[-1] = dt
    phi0             = torch.exp(-t ** 2 / 2)
    combined_weights = w * phi0

    dirs = torch.randn(D, n_projections, device=z.device, dtype=z.dtype)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + eps)

    x_t    = (z @ dirs).unsqueeze(-1) * t
    ecf_re = x_t.cos().mean(0)
    ecf_im = x_t.sin().mean(0)

    err    = (ecf_re - phi0) ** 2 + ecf_im ** 2
    T_stat = (err @ combined_weights) * B
    return T_stat.mean()


def state_to_target(states: torch.Tensor) -> torch.Tensor:
    """(N,3) (x,y,theta) -> (N,4) (x,y,cos,sin). Shared anchor-target builder."""
    x, y, th = states[:, 0], states[:, 1], states[:, 2]
    return torch.stack([x, y, torch.cos(th), torch.sin(th)], dim=1)

def img_loss(pred: torch.Tensor, target: torch.Tensor, fg_weight: float = 0.0) -> torch.Tensor:
    """Reconstruction loss. fg_weight>0 averages error over lit (foreground) and black
    (background) pixels SEPARATELY and re-weights: bg_mean + fg_weight * fg_mean. The
    robot is ~0.4% of pixels, so plain MSE is dominated by the black background and
    never bothers to reconstruct pose; this makes the loss actually about the object.
    """
    if fg_weight <= 0:
        return F.mse_loss(pred, target)
    err2 = (pred - target) ** 2
    fg = (target > 0).float()
    fg_mean = (err2 * fg).sum() / fg.sum().clamp(min=1.0)
    bg_mean = (err2 * (1.0 - fg)).sum() / (1.0 - fg).sum().clamp(min=1.0)
    return bg_mean + fg_weight * fg_mean

def pose_stats(states):
    """(N,3) -> standardization mean/std over [x,y,cosθ,sinθ]"""
    T = state_to_target(states)
    return T.mean(0), T.std(0) + 1e-6

def standardize(t, mean, std):   return (t - mean) / std
def unstandardize(t, mean, std): return t * std + mean