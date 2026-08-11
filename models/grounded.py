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
    IN_CHANNELS, DT, PHYSICS_BLOCK_DIM, WHEELBASE,
)

from models.base import WorldModel
from models.components import sigreg_loss, conv_trunk, constrain_pose, unicycle_step, bicycle_step, mlp, MLPDecoder, img_loss, make_residual


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
        if block_dim not in (4, 5):
            raise ValueError(f"block_dim must be 4 [x,y,cos,sin] or 5 [+v bicycle]; got {block_dim}")
        self.block_dim = block_dim
        self.free_dim  = latent_dim - block_dim

        self.conv, flat_dim = conv_trunk(grid_size, in_channels, channels)
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
        phys = constrain_pose(self.phys_head(feat))   # block -> [0,1] pos + unit-circle heading
        return torch.cat([phys, self.free_head(feat)], dim=1)


# ## Predictor: kinematics on the block, residual MLP on the free dims

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
        learn_coeffs: bool = False,
        residual_mode: str = "none",
        dynamics: str = "unicycle",
        wheelbase: float = WHEELBASE,
    ):
        super().__init__()
        self.block_dim    = block_dim
        self.dt           = dt
        self.dynamics     = dynamics
        self.wheelbase    = wheelbase
        self.lock_block   = lock_block
        self.block_budget = block_budget
        # gray-box speed/turn scales on the block kinematics. Learnable lets the block
        # absorb an unmodeled actuator gain (else locked at 1 = pure known kinematics).
        # Learn the SPEED scale a_v only (it absorbs the actuator gain). a_omega stays LOCKED:
        # the locked rotation is what grounds heading, and unlike the ego model the grounded
        # JEPA has no anchor_pred to constrain a learnable a_omega, so making it learnable lets
        # the encoder and a_omega collude into an under-rotating escape hatch (heading collapses).
        # Locked gains are the constant 1.0, non-persistent so they stay out of the checkpoint.
        if learn_coeffs:
            self.log_a_v = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_a_v", torch.zeros(()), persistent=False)
        self.register_buffer("log_a_omega", torch.zeros(()), persistent=False)
        self.net = mlp([latent_dim + action_dim, hidden, hidden, latent_dim])
        # higher-order gray-box residual on the block, conditioned on the FREE latent. 'basis' builds a
        # structured physics residual whose per-context coefficients c_k(free) are a linear readout of the
        # 124 free dims; 'mlp' feeds those same free dims into a free-form net. Toggle with residual_mode.
        self.residual = make_residual(residual_mode, dt, block_budget, free_dim=latent_dim - block_dim, dynamics=dynamics)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        K          = self.block_dim
        learned    = self.net(torch.cat([z, action], dim=-1))
        if self.dynamics == "bicycle":                # 5-D block, action (a, delta); a_v = throttle gain
            block_pred = bicycle_step(z[:, :K], action, self.dt, self.wheelbase, a_accel=self.log_a_v.exp())
        else:
            block_pred = unicycle_step(z[:, :K], action, self.dt,
                                       self.log_a_v.exp(), self.log_a_omega.exp())
        if self.residual is not None:                 # structured physics-basis correction on the block
            block_pred = block_pred + self.residual(z[:, :K], action, free=z[:, K:])
        elif not self.lock_block and self.block_budget > 0:
            block_pred = block_pred + self.block_budget * torch.tanh(learned[:, :K])
        free_pred  = z[:, K:] + learned[:, K:]        # residual on the free dims
        return torch.cat([block_pred, free_pred], dim=1)


# ## Optional decoder: frame from the block alone

# ## GroundedJEPA

class GroundedJEPA(WorldModel):
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
        learn_coeffs: bool = False,
        residual_mode: str = "none",
        in_channels: int = IN_CHANNELS,
        dynamics: str = "unicycle",
        wheelbase: float = WHEELBASE,
    ):
        if dynamics == "bicycle":
            block_dim = 5                                 # block carries velocity: [x,y,cos,sin,v]
        super().__init__(pose_dim=block_dim)
        self.latent_dim = latent_dim
        self.block_dim  = block_dim
        self.dynamics   = dynamics
        self.encoder = GroundedEncoder(grid_size, latent_dim, block_dim, channels=encoder_channels, in_channels=in_channels)
        self.predictor = GroundedPredictor(
            latent_dim, block_dim, ACTION_DIM, predictor_hidden, dt, lock_block, block_budget,
            learn_coeffs=learn_coeffs, residual_mode=residual_mode, dynamics=dynamics, wheelbase=wheelbase,
        )
        self.decoder = MLPDecoder(block_dim, grid_size) if use_decoder else None

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, action)

    def decode_pose(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent) -> (B, 4) real-unit pose: the block IS pose [x,y,cosθ,sinθ]."""
        return z[:, :self.block_dim]

    def extra_forward(self, frame, action, next_frame, out):
        """Optional block decode (grounding booster) when use_decoder=True."""
        if self.decoder is not None:
            return {"recon": self.decoder(out["z"][:, :self.block_dim])}
        return {}

    def representation_loss(self, out, batch, weights):
        """Grounded's own terms: block-weighted prediction MSE + SIGReg (free dims only) +
        optional block recon. anchor/anchor_pred are the shared WorldModel.pose_supervision
        (block IS pose, buffers 0/1 -> raw MSE on dims 0..block_dim)."""
        K    = self.block_dim
        err2 = (out["pred_next_z"] - out["target_next_z"].detach()) ** 2
        free_pred  = err2[:, K:].mean()
        block_pred = err2[:, :K].mean()
        pred = free_pred + weights.get("pred_block_weight", 1.0) * block_pred
        sig  = sigreg_loss(out["z"][:, K:])                     # SIGReg on the FREE dims only
        total = pred + weights.get("sigreg", 0.005) * sig
        parts = {"pred": pred.item(), "pred_block": block_pred.item(),
                 "pred_free": free_pred.item(), "sigreg": sig.item()}
        if "recon" in out and weights.get("recon", 0.0) > 0:
            recon = img_loss(out["recon"], batch["frame"][:, -1:], weights.get("fg_weight", 0.0))   # newest frame
            total = total + weights["recon"] * recon
            parts["recon"] = recon.item()
        if self.predictor.residual is not None and weights.get("l1", 0.0) > 0:   # sparsify basis coeffs
            l1 = self.predictor.residual.l1()
            total = total + weights["l1"] * l1; parts["l1"] = l1.item()
        return total, parts


def grounded_loss(model, out: dict, frame: torch.Tensor, block_dim: int = PHYSICS_BLOCK_DIM,
                  lam: float = 0.005, lam_recon: float = 0.0, recon_fg_weight: float = 0.0,
                  pred_block_weight: float = 1.0,
                  s_target: torch.Tensor = None, lam_anchor: float = 0.0,
                  s_next_target: torch.Tensor = None, lam_anchor_pred: float = 0.0) -> tuple:
    """Thin adapter: pack args into (batch, weights) and call model.loss. Terms live in
    GroundedJEPA.representation_loss (block-weighted pred + free-dim SIGReg + recon) +
    WorldModel.pose_supervision (block anchor/anchor_pred; buffers 0/1 -> raw MSE). block_dim
    comes from the model now, so the arg is ignored (kept for call-site compatibility)."""
    batch   = {"frame": frame, "s_target": s_target, "s_next_target": s_next_target}
    weights = {"sigreg": lam, "recon": lam_recon, "fg_weight": recon_fg_weight,
               "pred_block_weight": pred_block_weight,
               "anchor": lam_anchor, "anchor_pred": lam_anchor_pred}
    return model.loss(out, batch, weights)


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

        total, parts = grounded_loss(m, out, frame, block_dim=K, lam=0.005, lam_recon=(0.1 if dec else 0.0))
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
