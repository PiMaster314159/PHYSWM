"""Hidden-state ego world model: the 4-dim pose state PLUS K hidden dims that capture an
unobservable, per-episode physical parameter (here, the actuator gain a_v).

Latent = [ x, y, cos th, sin th | h_1 .. h_K ].
  - pose (4 dims): grounded by the anchor + reconstruction, exactly as the base ego model.
  - hidden (K dims): inferred from a SHORT STACK of frames (the gain is invisible in one frame;
    you only see it in how motion responds), and it SETS the speed coefficient via a_v = exp(head(h)).

So the hidden state literally modulates how the four pose values update. a_omega stays locked at 1
(the actuator only scales speed, and the locked rotation is the heading-grounding force).

Supervised vs unsupervised gain (a flag at the loss level via lam_gain):
  lam_gain > 0  -> anchor a_v=g(h) to the TRUE per-episode gain (privileged info, easy/clean).
  lam_gain = 0  -> a_v is forced only by the dynamics-consistency; probe h against truth afterward
                   (the stronger "discovered it on its own" result).
"""
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GRID_SIZE, DT, ENCODER_CHANNELS, ACTION_DIM
from models.state_ae import Renderer, _img_loss, STATE_DIM   # POSE dims = 4; reuse the renderer + recon

HIDDEN_DIM = 1   # default: one hidden dim for the scalar actuator gain
STACK      = 4   # frames per history stack (at stride = pred_step)


class HiddenStateEncoder(nn.Module):
    """Conv trunk over a STACK of frames -> [pose(4) | hidden(K)].

    The stack is `stack` frames at stride pred_step, so it spans real motion; pose reads off the
    most recent content, the hidden dims read off how the motion responds (which reveals the gain).
    """

    def __init__(self, grid_size: int = GRID_SIZE, channels: tuple = ENCODER_CHANNELS,
                 stack: int = STACK, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.stack, self.hidden_dim = stack, hidden_dim
        layers, c_in = [], stack                       # stacked frames go in as channels
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, stack, grid_size, grid_size)).flatten(1).shape[1]
        self.pose_head   = nn.Linear(flat, 4)
        self.hidden_head = nn.Linear(flat, hidden_dim)

    @staticmethod
    def constrain(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pos  = torch.sigmoid(raw[:, :2])
        head = F.normalize(raw[:, 2:4], dim=1, eps=eps)
        return torch.cat([pos, head], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"expected (B, stack, H, W), got {tuple(frames.shape)}")
        feat   = self.conv(frames).flatten(1)
        pose   = self.constrain(self.pose_head(feat))
        hidden = self.hidden_head(feat)
        return torch.cat([pose, hidden], dim=1)


def ego_step_hidden(state: torch.Tensor, action: torch.Tensor, dt: float,
                    a_v: torch.Tensor, a_omega: float = 1.0) -> torch.Tensor:
    """Semi-implicit unicycle on the pose dims with a PER-SAMPLE speed scale a_v (B,1); the hidden
    dims are carried forward unchanged (the per-episode gain is constant within an episode)."""
    x, y   = state[:, 0:1], state[:, 1:2]
    c, s   = state[:, 2:3], state[:, 3:4]
    hidden = state[:, 4:]
    v, omega = action[:, 0:1], action[:, 1:2]
    w  = a_omega * omega * dt
    cw, sw = torch.cos(w), torch.sin(w)
    c_new = c * cw - s * sw
    s_new = s * cw + c * sw
    x_new = x + a_v * v * c_new * dt
    y_new = y + a_v * v * s_new * dt
    return torch.cat([x_new, y_new, c_new, s_new, hidden], dim=1)


class HiddenEgoWorldModel(nn.Module):
    """Encoder(stack) -> [pose | hidden]; dynamics with a_v = exp(head(hidden)); renderer on pose."""

    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT, channels: tuple = ENCODER_CHANNELS,
                 stack: int = STACK, hidden_dim: int = HIDDEN_DIM, renderer_hidden: int = 512):
        super().__init__()
        self.dt, self.stack, self.hidden_dim = dt, stack, hidden_dim
        self.encoder  = HiddenStateEncoder(grid_size, channels, stack, hidden_dim)
        self.renderer = Renderer(STATE_DIM, grid_size, renderer_hidden)   # renders from the 4 pose dims
        # a_v = exp(head(hidden)); zero-init -> a_v = 1 at start (known kinematics), only learns to deviate
        self.av_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.av_head.weight); nn.init.zeros_(self.av_head.bias)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def gain(self, state: torch.Tensor) -> torch.Tensor:
        return self.av_head(state[:, 4:]).exp()        # (B,1) per-sample speed scale

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return self.renderer(state[:, :4])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return ego_step_hidden(state, action, self.dt, self.gain(state))

    def forward(self, frames: torch.Tensor, action: torch.Tensor, next_frames: torch.Tensor) -> dict:
        s      = self.encode(frames)
        s_next = self.encode(next_frames)
        s_pred = self.step(s, action)
        return {
            "s": s, "s_next": s_next, "s_pred": s_pred, "gain": self.gain(s),
            "recon": self.render(s), "pred_recon": self.render(s_pred),
        }


def hidden_ego_loss(out: dict, frame: torch.Tensor, next_frame: torch.Tensor,
                    s_target: torch.Tensor = None, s_next_target: torch.Tensor = None,
                    true_gain: torch.Tensor = None,
                    lam_recon: float = 1.0, lam_dyn: float = 1.0, lam_pred: float = 1.0,
                    recon_fg_weight: float = 5.0, lam_anchor: float = 1.0, lam_anchor_pred: float = 1.0,
                    lam_gain: float = 0.0) -> tuple:
    """Pose anchors + frame recon + dynamics consistency, plus an optional gain anchor.

    s_target / s_next_target are the 4-dim true pose [x,y,cos,sin] (frame_0 and the next transition).
    lam_gain > 0 supervises a_v=g(h) against the true per-episode gain; lam_gain = 0 leaves the gain
    to be forced purely by anchor_pred / dyn (unsupervised, then probe h).
    """
    recon = _img_loss(out["recon"], frame, recon_fg_weight)
    pred  = _img_loss(out["pred_recon"], next_frame, recon_fg_weight)
    dyn   = F.mse_loss(out["s_pred"], out["s_next"].detach())          # pose + hidden consistency
    zero  = out["s"].new_zeros(())
    anchor      = F.mse_loss(out["s"][:, :4], s_target)      if (lam_anchor > 0 and s_target is not None) else zero
    anchor_pred = F.mse_loss(out["s_pred"][:, :4], s_next_target) if (lam_anchor_pred > 0 and s_next_target is not None) else zero
    gain        = F.mse_loss(out["gain"], true_gain)         if (lam_gain > 0 and true_gain is not None) else zero
    total = (lam_recon * recon + lam_dyn * dyn + lam_pred * pred
             + lam_anchor * anchor + lam_anchor_pred * anchor_pred + lam_gain * gain)
    parts = {"recon": recon.item(), "dyn": dyn.item(), "pred_recon": pred.item(),
             "anchor": anchor.item(), "anchor_pred": anchor_pred.item(), "gain": gain.item(),
             "total": total.item()}
    return total, parts
