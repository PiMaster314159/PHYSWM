#!/usr/bin/env python
# coding: utf-8

# # JEPA model
# 
# Encoder + Predictor for LeWM training.
# 
# Pipeline: `(frame_t, action_t, frame_{t+1})` -> encoder -> predictor -> latent prediction. The encoder maps pixels to latents. The predictor maps `(z_t, action_t)` to a predicted `z_{t+1}`. Training minimizes the distance between that prediction and the actual encoded next frame.
# 
# > To generate `models/jepa.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python models/jepa.ipynb
# > ```

# ## Path setup
# 
# Same root-finding pattern as `dataset.py`.

# In[ ]:


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DATA_PATH, DT, PRED_STEP,
    GRID_SIZE, IN_CHANNELS, LATENT_DIM, ENCODER_CHANNELS,
    PREDICTOR_HIDDEN, ACTION_DIM, PREDICTOR_MODE,
)


# ## Imports

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Encoder
# 
# CNN: `(B, 1, H, W)` -> `(B, latent_dim)`.
# 
# Three stride-2 conv layers halve spatial dims and grow channel count at each stage. Flattened size is measured once at init via a dummy forward pass so the same class works at any grid size without changes.

# In[ ]:


class Encoder(nn.Module):
    """CNN: binary frame -> latent vector.

    Parameters
    ----------
    grid_size : int
        Side length of the square input frame.
    in_channels : int
        Input channels. 1 for single-channel binary frames.
    latent_dim : int
        Dimension of the output latent z.
    channels : tuple of int
        Output channels for each conv layer. Length sets the number of
        stride-2 downsampling stages.

    Defaults pull from config.py.

    Notes
    -----
    BatchNorm1d on the projection head is intentional, not just for
    normalization. The EP-based SIGReg loss has near-zero gradients when
    latents are near zero (sin(t*h) -> 0 as h -> 0). BatchNorm keeps
    latents off zero by construction, preventing gradient vanishing and
    allowing SIGReg to shape the distribution effectively. This mirrors
    the le-wm paper's design: "This step is necessary because [LayerNorm]
    prevents our anti-collapse objective from being optimized effectively."
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        in_channels: int = IN_CHANNELS,
        latent_dim: int = LATENT_DIM,
        channels: tuple = ENCODER_CHANNELS,
    ):
        super().__init__()
        self.grid_size  = grid_size
        self.latent_dim = latent_dim

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, grid_size, grid_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, H, W)

        Returns
        -------
        z : torch.Tensor, shape (B, latent_dim)
        """
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self.head(self.conv(x).flatten(1))


# ## Predictor
# 
# Predicts the next latent as `base(z, a) + mlp(z, a)`. A `mode` switch picks the base, which is the only thing that differs between the three variants:
# 
# - `mlp`: base = 0, predict the next latent from scratch.
# - `residual`: base = z, predict the change; identity (next = current) is the default at init.
# - `physics`: base = z + a learnable-scaled unicycle kinematic step on the first 3 latent dims (read as x, y, theta).
# 
# The physics base uses only the latent's own dims 0, 1, 2 and the action; no ground-truth pose leaks in. Two learnable scalars (`a_pos`, `a_theta`) bridge latent units and physical units: BatchNorm forces every latent dim to ~unit variance, but the kinematics want position in world units and heading in radians, so `a_pos` amplifies the tiny physical step (`dt·v ≈ 0.018`) to a latent-scale change and `a_theta` reads `z2` as a true angle. This pegs dims 0-2 to pose: to make the base accurate the encoder must put a faithful heading in `z2`. The learned MLP corrects the base and handles the other dims. Set the mode in `config.py` (`PREDICTOR_MODE`).

# In[ ]:


class Predictor(nn.Module):
    """Predicts the next latent as base(z, a) + a learned MLP correction.

    The `mode` selects the base, the only difference between the three variants:

      "mlp"      base = 0       predict the next latent from scratch
      "residual" base = z       predict the change; identity default
      "physics"  base = z + a learnable-scaled unicycle step on dims 0, 1, 2

    The physics base reads dims 0, 1, 2 as (x, y, theta):

        theta = a_theta * z2
        z0_next = z0 + a_pos * dt * v * cos(theta)
        z1_next = z1 + a_pos * dt * v * sin(theta)
        z2_next = z2 + (dt * omega) / a_theta

    using only the latent's own values and the action (no ground-truth state
    leaks in). a_pos and a_theta are learnable scalars (kept in log-space so they
    stay positive; both start at 1). They bridge latent units and physical units:
    BatchNorm forces unit-variance latents, but the kinematics want z0,z1 in world
    units and z2 in radians. a_pos amplifies the tiny physical step (dt*v ~ 0.018)
    to a latent-scale change; a_theta lets z2 sit at unit variance yet be read as a
    true heading. Without them the kinematic term is a ~2% nudge on identity and
    barely shapes anything. To make the base accurate the encoder must put a
    faithful heading in z2, which is the pressure the plain objective lacks.

    Parameters
    ----------
    latent_dim : int
    action_dim : int
        Raw action size. 2 for (v, omega).
    hidden : int
        MLP hidden width.
    mode : {"mlp", "residual", "physics"}
        Which base to add the learned correction to.
    dt : float
        Timestep for the physics base. Defaults to PRED_STEP * DT, so a multi-step
        transition integrates over its full horizon (a single Euler step of that
        size; the learned MLP corrects the approximation).

    Defaults pull from config.py.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = PREDICTOR_HIDDEN,
        mode: str = PREDICTOR_MODE,
        dt: float = PRED_STEP * DT,
    ):
        super().__init__()
        if mode not in ("mlp", "residual", "physics"):
            raise ValueError(f"mode must be mlp/residual/physics, got {mode!r}")
        if mode == "physics" and latent_dim < 3:
            raise ValueError("physics mode needs latent_dim >= 3 (uses dims 0,1,2)")
        self.mode = mode
        self.dt   = dt
        self.net  = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        if mode == "physics":
            # learnable unit-bridging scales, log-space so they stay positive (=1 at init)
            self.log_a_pos   = nn.Parameter(torch.zeros(()))
            self.log_a_theta = nn.Parameter(torch.zeros(()))

    def _physics_base(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z + learnable-scaled unicycle step on dims 0,1,2 (read as x, y, theta)."""
        a_pos   = self.log_a_pos.exp()
        a_theta = self.log_a_theta.exp()
        v     = action[:, 0:1]              # (B, 1)
        omega = action[:, 1:2]
        theta = a_theta * z[:, 2:3]
        dx   = a_pos * self.dt * v * torch.cos(theta)
        dy   = a_pos * self.dt * v * torch.sin(theta)
        dth  = (self.dt * omega) / a_theta
        rest = torch.zeros_like(z[:, 3:])   # other dims: no kinematic change
        delta = torch.cat([dx, dy, dth, rest], dim=1)
        return z + delta

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(z: (B, D), action: (B, action_dim)) -> predicted next latent (B, D)."""
        learned = self.net(torch.cat([z, action], dim=-1))
        if self.mode == "mlp":
            return learned
        if self.mode == "residual":
            return z + learned
        return self._physics_base(z, action) + learned


# ## JEPA
# 
# Wires encoder and predictor together. Forward returns everything the loss needs: current latent, predicted next latent, and actual encoded next latent.
# 
# Stop-gradient on the target is applied in `jepa_loss`, not here, keeping this a pure forward pass.

# In[ ]:


class JEPA(nn.Module):
    """Vanilla LeWM: encoder + latent-space predictor.

    Parameters
    ----------
    grid_size : int
        Input frame side length.
    latent_dim : int
        Latent z size.
    encoder_channels : tuple of int
        Conv channel widths for the encoder.
    predictor_hidden : int
        Predictor MLP hidden width.
    predictor_mode : {"mlp", "residual", "physics"}
        Predictor base. See Predictor.

    Defaults pull from config.py.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        latent_dim: int = LATENT_DIM,
        encoder_channels: tuple = ENCODER_CHANNELS,
        predictor_hidden: int = PREDICTOR_HIDDEN,
        predictor_mode: str = PREDICTOR_MODE,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(
            grid_size=grid_size,
            latent_dim=latent_dim,
            channels=encoder_channels,
        )
        self.predictor  = Predictor(
            latent_dim=latent_dim,
            action_dim=ACTION_DIM,
            hidden=predictor_hidden,
            mode=predictor_mode,
        )

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, latent_dim)."""
        return self.encoder(frame)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(z: (B, D), action: (B, 2)) -> predicted next latent (B, D)."""
        return self.predictor(z, action)

    def forward(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> dict:
        """Full predict-vs-target pass.

        Returns
        -------
        dict with keys
            z : (B, D)  encoding of frame_t
            pred_next_z : (B, D)  predictor forecast of z_{t+1}
            target_next_z : (B, D)  encoder output for next frame
        """
        z             = self.encode(frame)
        pred_next_z   = self.predict(z, action)
        target_next_z = self.encode(next_frame)
        return {"z": z, "pred_next_z": pred_next_z, "target_next_z": target_next_z}


# ## Loss
# 
# Two terms, following the LeWM paper exactly.
# 
# **Prediction loss**: MSE between the predicted next latent and the encoded next frame. Stop-gradient on the target prevents both ends of the shared encoder collapsing to the same constant and trivially zeroing the loss.
# 
# **SIGReg**: the Sketched-Isotropic-Gaussian Regularizer. Projects the batch of latents onto M random unit directions, then applies the univariate Epps-Pulley (EP) normality test to each projection. By the Cramer-Wold theorem, matching all 1D projections to N(0,1) implies the full joint distribution matches N(0, I). This is a provable anti-collapse guarantee.

# In[ ]:


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


def jepa_loss(out: dict, lam: float = 1.0) -> tuple:
    """LeWM training loss: prediction MSE + SIGReg.

    Parameters
    ----------
    out : dict
        Output of JEPA.forward(). Keys: z, pred_next_z, target_next_z.
    lam : float
        Weight on SIGReg. Lambda in the paper.

    Returns
    -------
    total : scalar loss with grad
    parts : dict with float entries pred, sigreg, total (detached, for logging)
    """
    pred  = F.mse_loss(out["pred_next_z"], out["target_next_z"].detach())
    sig   = sigreg_loss(out["z"])
    total = pred + lam * sig
    parts = {"pred": pred.item(), "sigreg": sig.item(), "total": total.item()}
    return total, parts


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ## Tests
# 
# Shape checks on random and real batches, resolution-agnosticism, action-sensitivity, gradient wiring, and loss finiteness.

# In[ ]:


def _test_jepa():
    torch.manual_seed(0)
    B, D = 8, 128

    model = JEPA(grid_size=40, latent_dim=D)

    # param breakdown
    for name, m in [("encoder", model.encoder), ("predictor", model.predictor)]:
        print(f"{name:12s}  {count_parameters(m):>9,} params")
    print(f"{'TOTAL':12s}  {count_parameters(model):>9,} params")

    # shapes on random input
    frame      = torch.rand(B, 1, 40, 40)
    action     = torch.rand(B, 2)
    next_frame = torch.rand(B, 1, 40, 40)
    out = model(frame, action, next_frame)
    for key in ("z", "pred_next_z", "target_next_z"):
        assert out[key].shape == (B, D), (key, out[key].shape)
        assert torch.isfinite(out[key]).all(), f"{key} contains non-finite values"

    # resolution-agnostic: same class works at 40, 64, 84
    for g in (40, 64, 84):
        m = JEPA(grid_size=g, latent_dim=32)
        o = m(torch.rand(2, 1, g, g), torch.rand(2, 2), torch.rand(2, 1, g, g))
        assert o["pred_next_z"].shape == (2, 32), (g, o["pred_next_z"].shape)

    # predictor modes: all three build + run; physics scales receive gradient
    for mode in ("mlp", "residual", "physics"):
        mm = JEPA(grid_size=40, latent_dim=D, predictor_mode=mode)
        oo = mm(frame, action, next_frame)
        assert oo["pred_next_z"].shape == (B, D), (mode, oo["pred_next_z"].shape)
        if mode == "physics":
            ll, _ = jepa_loss(oo)
            mm.zero_grad()
            ll.backward()
            assert mm.predictor.log_a_pos.grad   is not None, "a_pos got no gradient"
            assert mm.predictor.log_a_theta.grad is not None, "a_theta got no gradient"
    print("predictor modes mlp/residual/physics build + run OK (physics scales trainable)")

    # action sensitivity: different action -> different prediction
    z  = model.encode(frame)
    p1 = model.predict(z, torch.zeros(B, 2))
    p2 = model.predict(z, torch.ones(B, 2))
    assert not torch.allclose(p1, p2), "predictor ignores the action!"

    # wrong-rank frame is rejected
    try:
        model.encode(torch.rand(1, 40, 40))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 3D input")

    # sigreg: low for N(0,1), high for collapsed and inflated distributions
    z_normal   = torch.randn(256, D)
    z_collapse = torch.zeros(256, D) + 0.01 * torch.randn(1, D)
    z_inflated = 5.0 * torch.randn(256, D)
    s_n = sigreg_loss(z_normal).item()
    s_c = sigreg_loss(z_collapse).item()
    s_i = sigreg_loss(z_inflated).item()
    print(f"sigreg: normal={s_n:.4f}  collapsed={s_c:.4f}  inflated={s_i:.4f}")
    assert s_c > s_n, "sigreg should penalise collapsed latents more than normal"
    assert s_i > s_n, "sigreg should penalise inflated latents more than normal"

    # gradient wiring: loss must reach every parameter
    out = model(frame, action, next_frame)
    loss, parts = jepa_loss(out)
    assert torch.isfinite(loss), "loss is non-finite"
    model.zero_grad()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
    print(f"gradient wiring OK  pred={parts['pred']:.4f}  sigreg={parts['sigreg']:.4f}")

    # real batch from the DataLoader
    if DATA_PATH.exists():
        try:
            from models.dataset import make_dataloaders
        except ImportError:
            from dataset import make_dataloaders
        train_dl, _ = make_dataloaders(DATA_PATH, batch_size=16)
        batch = next(iter(train_dl))
        g     = train_dl.dataset.grid_size
        model = JEPA(grid_size=g, latent_dim=D)
        out   = model(batch["frame"], batch["action"], batch["next_frame"])
        assert out["pred_next_z"].shape == (16, D)
        loss, parts = jepa_loss(out)
        assert torch.isfinite(loss)
        print(f"real batch {tuple(batch['frame'].shape)} -> "
              f"pred_next_z {tuple(out['pred_next_z'].shape)}  "
              f"loss={parts['total']:.4f}  pred={parts['pred']:.4f}  sigreg={parts['sigreg']:.4f}")
    else:
        print(f"(skipping real-batch test: {DATA_PATH} not found)")

    print("All JEPA tests passed.")


# In[ ]:


if __name__ == "__main__":
    _test_jepa()

