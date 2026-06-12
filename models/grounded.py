"""Grounded-physics-block JEPA.

Splits the latent into two parts:
  - physics block (dims 0..K-1 = x, y, cos th, sin th): its OWN raw linear head
    (no BatchNorm), EXEMPT from SIGReg, evolved by the known unicycle kinematics.
  - free block (dims K..): the usual BatchNorm'd head, SIGReg'd, residual MLP.

Grounding is label-free. The block's forward prediction is pure kinematics
(MLP masked off it), so the ordinary prediction loss restricted to the block IS
the physics-consistency constraint: the encoder can only lower it by reading true
pose out of the pixels (it never sees the action). The known equations, written
in explicit (x, y, cos, sin) form, pin the dims into those roles. No state labels.

Gray-box knob: set lock_block=False and a block_budget > 0 to allow a bounded
learned correction on the block (rough physics + friction/slip residual), for
systems whose exact dynamics are unknown (RC car, cloth).

Optional decoder: reconstruct the frame from the block alone (lam_recon > 0).
Off by default; useful when the dynamics are degenerate (v=0) or only approximate.

Reuses sigreg_loss from models.jepa. Run experiments via run_grounded.py.
"""

import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    GRID_SIZE, LATENT_DIM, ACTION_DIM, ENCODER_CHANNELS, PREDICTOR_HIDDEN,
    IN_CHANNELS, DT, PHYSICS_BLOCK_DIM,
)

try:
    from models.jepa import sigreg_loss
except ImportError:
    from jepa import sigreg_loss


# ## Encoder with a split head

class GroundedEncoder(nn.Module):
    """Conv trunk + two heads: a raw physics block and a BatchNorm'd free block.

    The physics head has NO BatchNorm so the block can hold real-scale state
    (x ~ [0,1], heading a unit vector). The free head keeps BatchNorm for the
    anti-collapse SIGReg machinery. Outputs are concatenated as [block | free].
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        latent_dim: int = LATENT_DIM,
        block_dim: int = PHYSICS_BLOCK_DIM,
        in_channels: int = IN_CHANNELS,
        channels: tuple = ENCODER_CHANNELS,
    ):
        super().__init__()
        if block_dim >= latent_dim:
            raise ValueError(f"block_dim {block_dim} must be < latent_dim {latent_dim}")
        self.block_dim = block_dim
        self.free_dim  = latent_dim - block_dim

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, grid_size, grid_size)).flatten(1).shape[1]

        self.phys_head = nn.Linear(flat_dim, block_dim)            # raw, no BatchNorm
        self.free_head = nn.Sequential(
            nn.Linear(flat_dim, self.free_dim),
            nn.BatchNorm1d(self.free_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, latent_dim), block dims first."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        feat = self.conv(x).flatten(1)
        return torch.cat([self.phys_head(feat), self.free_head(feat)], dim=1)


# ## Predictor: kinematics on the block, residual MLP on the free dims

def unicycle_step(block: torch.Tensor, action: torch.Tensor, dt: float, eps: float = 1e-4) -> torch.Tensor:
    """Advance a (B, 4) block [x, y, cos th, sin th] by one unicycle step.

    Semi-implicit Euler, matching sim/dynamics.py: rotate heading first, then move
    along the NEW heading. Heading stays on the unit circle by construction (a
    rotation), and the prediction loss pushes the encoder's raw heading dims onto
    it too. No angle wrapping needed since heading is carried as (cos, sin).
    """
    x, y   = block[:, 0:1], block[:, 1:2]
    c, s   = block[:, 2:3], block[:, 3:4]
    n      = torch.sqrt(c * c + s * s) + eps          # defensive normalize of the input heading
    c, s   = c / n, s / n
    v      = action[:, 0:1]
    omega  = action[:, 1:2]

    cw, sw = torch.cos(omega * dt), torch.sin(omega * dt)
    c_new  = c * cw - s * sw                          # rotate the heading vector by omega*dt
    s_new  = s * cw + c * sw
    x_new  = x + v * c_new * dt                       # move along the new heading
    y_new  = y + v * s_new * dt
    return torch.cat([x_new, y_new, c_new, s_new], dim=1)


class GroundedPredictor(nn.Module):
    """Next latent = [ kinematics(block) (+budget*corr) | free + mlp_free ].

    lock_block=True: the block is pure kinematics (no learned correction), so the
    prediction loss on the block becomes the physics-consistency constraint.
    lock_block=False: add a bounded learned correction tanh(.)*block_budget to the
    block (gray-box: rough physics + learned residual for unknown dynamics).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        block_dim: int = PHYSICS_BLOCK_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = PREDICTOR_HIDDEN,
        dt: float = DT,
        lock_block: bool = True,
        block_budget: float = 0.0,
    ):
        super().__init__()
        self.block_dim    = block_dim
        self.dt           = dt
        self.lock_block   = lock_block
        self.block_budget = block_budget
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        K          = self.block_dim
        learned    = self.net(torch.cat([z, action], dim=-1))
        block_pred = unicycle_step(z[:, :K], action, self.dt)
        if not self.lock_block and self.block_budget > 0:
            block_pred = block_pred + self.block_budget * torch.tanh(learned[:, :K])
        free_pred  = z[:, K:] + learned[:, K:]        # residual on the free dims
        return torch.cat([block_pred, free_pred], dim=1)


# ## Optional decoder: frame from the block alone

class BlockDecoder(nn.Module):
    """Reconstruct the frame from ONLY the physics block (grounding booster)."""

    def __init__(self, block_dim: int = PHYSICS_BLOCK_DIM, grid_size: int = GRID_SIZE, hidden: int = 512):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(block_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, grid_size * grid_size),
            nn.Sigmoid(),                              # frames live in [0, 1]
        )

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        g = self.grid_size
        return self.net(block).view(-1, 1, g, g)


# ## GroundedJEPA

class GroundedJEPA(nn.Module):
    """Split-latent JEPA: grounded physics block + free SIGReg block."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        latent_dim: int = LATENT_DIM,
        block_dim: int = PHYSICS_BLOCK_DIM,
        encoder_channels: tuple = ENCODER_CHANNELS,
        predictor_hidden: int = PREDICTOR_HIDDEN,
        dt: float = DT,
        lock_block: bool = True,
        block_budget: float = 0.0,
        use_decoder: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.block_dim  = block_dim
        self.encoder = GroundedEncoder(grid_size, latent_dim, block_dim, channels=encoder_channels)
        self.predictor = GroundedPredictor(
            latent_dim, block_dim, ACTION_DIM, predictor_hidden, dt, lock_block, block_budget,
        )
        self.decoder = BlockDecoder(block_dim, grid_size) if use_decoder else None

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, action)

    def forward(self, frame: torch.Tensor, action: torch.Tensor, next_frame: torch.Tensor) -> dict:
        z             = self.encode(frame)
        pred_next_z   = self.predict(z, action)
        target_next_z = self.encode(next_frame)
        out = {"z": z, "pred_next_z": pred_next_z, "target_next_z": target_next_z}
        if self.decoder is not None:
            out["recon"] = self.decoder(z[:, :self.block_dim])
        return out


def grounded_loss(out: dict, frame: torch.Tensor, block_dim: int = PHYSICS_BLOCK_DIM,
                  lam: float = 0.005, lam_recon: float = 0.0) -> tuple:
    """Prediction MSE (all dims) + SIGReg (FREE dims only) + optional recon.

    The block part of the prediction loss is the label-free physics-consistency
    constraint (because the block forward is locked kinematics). SIGReg is applied
    only to dims block_dim.. so the block is free to hold real-scale state.
    """
    pred  = F.mse_loss(out["pred_next_z"], out["target_next_z"].detach())
    sig   = sigreg_loss(out["z"][:, block_dim:])
    total = pred + lam * sig
    parts = {"pred": pred.item(), "sigreg": sig.item()}

    if lam_recon > 0 and "recon" in out:
        recon = F.mse_loss(out["recon"], frame)
        total = total + lam_recon * recon
        parts["recon"] = recon.item()

    parts["total"] = total.item()
    return total, parts


# ## Smoke test
#
# Shapes + gradient wiring on random data. Run on the tower BEFORE a full train:
#   python -c "from models.grounded import _test_grounded; _test_grounded()"

def _test_grounded():
    torch.manual_seed(0)
    B, D, K, g = 8, 128, 4, 64

    for lock, budget, dec in [(True, 0.0, False), (False, 0.1, True)]:
        m = GroundedJEPA(grid_size=g, latent_dim=D, block_dim=K,
                         lock_block=lock, block_budget=budget, use_decoder=dec)
        frame      = torch.rand(B, 1, g, g)
        action     = torch.rand(B, 2)
        next_frame = torch.rand(B, 1, g, g)
        out = m(frame, action, next_frame)

        for key in ("z", "pred_next_z", "target_next_z"):
            assert out[key].shape == (B, D), (key, out[key].shape)
            assert torch.isfinite(out[key]).all(), key
        if dec:
            assert out["recon"].shape == (B, 1, g, g)

        total, parts = grounded_loss(out, frame, block_dim=K, lam=0.005, lam_recon=(0.1 if dec else 0.0))
        assert torch.isfinite(total)
        m.zero_grad()
        total.backward()

        # encoder must receive gradient through BOTH heads
        gphys = m.encoder.phys_head.weight.grad
        gfree = m.encoder.free_head[0].weight.grad
        assert gphys is not None and gphys.abs().sum() > 0, "physics head got no gradient"
        assert gfree is not None and gfree.abs().sum() > 0, "free head got no gradient"

        # the locked block must carry NO learned-correction gradient path into net's block slice
        print(f"lock={lock} budget={budget} decoder={dec}  "
              f"total={parts['total']:.4f} pred={parts['pred']:.4f} sigreg={parts['sigreg']:.4f}"
              + (f" recon={parts['recon']:.4f}" if 'recon' in parts else ""))

    # unicycle_step sanity: facing +x (cos=1,sin=0), v=1, omega=0, dt=0.1 -> x advances 0.1
    b = torch.tensor([[0.5, 0.5, 1.0, 0.0]])
    nb = unicycle_step(b, torch.tensor([[1.0, 0.0]]), dt=0.1)
    assert torch.allclose(nb, torch.tensor([[0.6, 0.5, 1.0, 0.0]]), atol=1e-5), nb
    # pure rotation: omega such that omega*dt = pi/2 -> heading (1,0) becomes (0,1)
    nb = unicycle_step(b, torch.tensor([[0.0, (torch.pi / 2) / 0.1]]), dt=0.1)
    assert torch.allclose(nb[:, 2:], torch.tensor([[0.0, 1.0]]), atol=1e-5), nb
    print("All grounded tests passed.")


if __name__ == "__main__":
    _test_grounded()
