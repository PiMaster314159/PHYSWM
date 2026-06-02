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

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "constants.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH


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
        grid_size: int = 40,
        in_channels: int = 1,
        latent_dim: int = 128,
        channels: tuple = (32, 64, 128),
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
# MLP: `(z_t, action_t)` -> predicted `z_{t+1}`.
# 
# Concatenates the current latent and raw `(v, omega)` action, passes through three linear layers. No action encoder needed. `(v, omega)` are already clean physical signals and the MLP learns to weight them.

# In[ ]:


class Predictor(nn.Module):
    """MLP: (z_t, action_t) -> predicted next latent.

    Parameters
    ----------
    latent_dim : int
        Size of z, both input and output.
    action_dim : int
        Raw action size. 2 for (v, omega).
    hidden : int
        Hidden layer width.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        hidden: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)
        action : (B, action_dim)

        Returns
        -------
        (B, latent_dim)
        """
        return self.net(torch.cat([z, action], dim=-1))


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
    """

    def __init__(
        self,
        grid_size: int = 40,
        latent_dim: int = 128,
        encoder_channels: tuple = (32, 64, 128),
        predictor_hidden: int = 256,
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
            action_dim=2,
            hidden=predictor_hidden,
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

