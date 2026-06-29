"""Hidden-state ego world model: the 4-dim pose state PLUS K hidden dims that capture an
unobservable, per-episode physical parameter (here, the actuator gain a_v).

Latent = [ x, y, cos th, sin th | h_1 .. h_K ].
  - pose (4 dims): grounded by the anchor + reconstruction, exactly as the base ego model.
  - hidden (K dims): inferred from a SHORT STACK of frames AND the commanded actions over that
    stack (the gain is invisible in one frame, and unidentifiable without the actions: observed
    motion alone confounds the gain with the unseen commanded velocity). The FIRST hidden dim IS
    the speed scale a_v directly (no coefficient head, no squashing): x' = x + h*v*cos*dt.

So the hidden state literally modulates how the four pose values update, and reads out as the gain
with no intermediate mapping. a_omega stays locked at 1 (the actuator only scales speed, and the
locked rotation is the heading-grounding force).

Supervised vs unsupervised gain (a flag at the loss level via lam_gain):
  lam_gain > 0  -> anchor a_v=h to the TRUE per-episode gain (privileged info, easy/clean).
  lam_gain = 0  -> a_v=h is forced only by the dynamics-consistency (anchor_pred / dyn); probe h
                   against truth afterward (the stronger "discovered it on its own" result).
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
    """DECOUPLED encoder -> [pose(4) | hidden(K)].

    Pose is read from the CURRENT (most recent) frame only, so it stays sharp, the same single-frame
    signal the base ego uses. The hidden gain is read from the FULL stack of frames PLUS the
    commanded action stack: motion across frames reveals the displacement, the actions reveal the
    command, and the gain is their ratio (so it is unidentifiable from frames alone). Splitting the
    two trunks keeps pose tight while still inferring the gain.
    """

    def __init__(self, grid_size: int = GRID_SIZE, channels: tuple = ENCODER_CHANNELS,
                 stack: int = STACK, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.stack, self.hidden_dim = stack, hidden_dim

        def trunk(c_in):
            layers, c = [], c_in
            for c_out in channels:
                layers += [nn.Conv2d(c, c_out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
                c = c_out
            return nn.Sequential(*layers)

        self.pose_conv = trunk(1)          # single current frame -> pose (sharp)
        self.gain_conv = trunk(stack)      # full stack -> hidden gain (needs motion)
        with torch.no_grad():
            flat_p = self.pose_conv(torch.zeros(1, 1, grid_size, grid_size)).flatten(1).shape[1]
            flat_g = self.gain_conv(torch.zeros(1, stack, grid_size, grid_size)).flatten(1).shape[1]
        self.pose_head   = nn.Linear(flat_p, 4)
        # gain head sees motion features AND the commanded action stack (flattened). a_v = first
        # hidden dim directly; init weight=0, bias=1 so a_v starts at 1.0 (known kinematics) and the
        # model only learns to read the per-episode gain off the (motion, action) pair.
        self.hidden_head = nn.Linear(flat_g + stack * ACTION_DIM, hidden_dim)
        nn.init.zeros_(self.hidden_head.weight); nn.init.ones_(self.hidden_head.bias)

    @staticmethod
    def constrain(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pos  = torch.sigmoid(raw[:, :2])
        head = F.normalize(raw[:, 2:4], dim=1, eps=eps)
        return torch.cat([pos, head], dim=1)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"expected frames (B, stack, H, W), got {tuple(frames.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"expected actions (B, stack, {ACTION_DIM}), got {tuple(actions.shape)}")
        current = frames[:, -1:, :, :]                          # most recent frame -> pose
        pose    = self.constrain(self.pose_head(self.pose_conv(current).flatten(1)))
        motion  = self.gain_conv(frames).flatten(1)             # displacement across the stack
        hidden  = self.hidden_head(torch.cat([motion, actions.flatten(1)], dim=1))
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
    """Encoder(stack, actions) -> [pose | hidden]; dynamics with a_v = the hidden dim itself
    (no coefficient head, no squashing); renderer on the 4 pose dims."""

    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT, channels: tuple = ENCODER_CHANNELS,
                 stack: int = STACK, hidden_dim: int = HIDDEN_DIM, renderer_hidden: int = 512):
        super().__init__()
        self.dt, self.stack, self.hidden_dim = dt, stack, hidden_dim
        self.encoder  = HiddenStateEncoder(grid_size, channels, stack, hidden_dim)
        self.renderer = Renderer(STATE_DIM, grid_size, renderer_hidden)   # renders from the 4 pose dims

    def encode(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames, actions)

    def gain(self, state: torch.Tensor) -> torch.Tensor:
        # a_v IS the first hidden latent dim, used directly as the velocity scale. No coefficient
        # head and no sigmoid/exp: the dynamics' anchor_pred pins it to the true displacement scale
        # (~[0.5,1]), so it cannot run away the way the unbounded/saturating coefficient did.
        return state[:, 4:5]                                    # (B,1)

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return self.renderer(state[:, :4])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return ego_step_hidden(state, action, self.dt, self.gain(state))

    def forward(self, frames: torch.Tensor, action_hist: torch.Tensor, action: torch.Tensor,
                next_frames: torch.Tensor, next_action_hist: torch.Tensor) -> dict:
        s      = self.encode(frames, action_hist)
        s_next = self.encode(next_frames, next_action_hist)
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
