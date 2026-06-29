"""Ego world model: a state-bottleneck autoencoder with gray-box dynamics.

This replaces the JEPA / grounded-block approach. The whole latent IS a small,
interpretable physical state s = [x, y, cos th, sin th]. There is no free block to
escape into, so the encoder is forced to put ALL of pose (heading included) into s
to reconstruct the frame. A gray-box dynamics advances the state with the KNOWN
kinematics plus learnable physical coefficients (and an optional bounded residual
for unmodeled effects). See MODEL.md for the architecture and rationale.

Three training signals (all label-free):
  recon       render(s_t)        ~ frame_t        grounds the state in pixels
  dyn         step(s_t, a)       ~ enc(frame_t1)  enforces the physics in state space
  pred_recon  render(step(s_t,a))~ frame_t1       ties the predicted state to pixels

No SIGReg, no decoder-free latent: the small bottleneck + reconstruction is the
anti-collapse mechanism (LeCun Fig 10c: a low-capacity AE cannot collapse).

Self-contained: depends only on config + torch. Train/eval via run_ego.py.
"""

import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GRID_SIZE, ENCODER_CHANNELS, IN_CHANNELS, ACTION_DIM, DT

STATE_DIM = 4   # x, y, cos th, sin th


# ## Encoder: pixels -> physical state

class StateEncoder(nn.Module): 
    """Conv trunk -> a STATE_DIM state, constrained to physical ranges.

    Position dims -> sigmoid -> [0,1] (arena bounds). Heading dims -> unit circle.
    No BatchNorm: the state must hold real-scale values, and the bottleneck +
    renderer (not a variance regularizer) prevents collapse.
    """

    def __init__(self, grid_size: int = GRID_SIZE, channels: tuple = ENCODER_CHANNELS,
                 in_channels: int = IN_CHANNELS):
        super().__init__()
        layers, c_in = [], in_channels
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, grid_size, grid_size)).flatten(1).shape[1]
        self.head = nn.Linear(flat, STATE_DIM)

    @staticmethod
    def constrain(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pos  = torch.sigmoid(raw[:, :2])
        head = F.normalize(raw[:, 2:4], dim=1, eps=eps)
        return torch.cat([pos, head], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        return self.constrain(self.head(self.conv(x).flatten(1)))


# ## Renderer (decoder): physical state -> frame

class Renderer(nn.Module):
    """state -> frame. Learns the simulator's renderer from 4 numbers."""

    def __init__(self, state_dim: int = STATE_DIM, grid_size: int = GRID_SIZE, hidden: int = 512):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),    nn.ReLU(inplace=True),
            nn.Linear(hidden, grid_size * grid_size), nn.Sigmoid(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        g = self.grid_size
        return self.net(s).view(-1, 1, g, g)


class SpatialBroadcastDecoder(nn.Module):
    """state -> frame via spatial broadcast (Watters et al. 2019).

    Tile the state across the H x W grid, append fixed (x, y) coordinate channels,
    then run small stride-1 convs. Unlike an MLP, this CAN place a small object at an
    arbitrary position, which is exactly what the MLP renderer fails at (it collapses
    to drawing the mean frame), so reconstruction can finally demand true position.
    """

    def __init__(self, state_dim: int = STATE_DIM, grid_size: int = GRID_SIZE, hidden: int = 64):
        super().__init__()
        self.grid_size = grid_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, grid_size),
                                torch.linspace(-1, 1, grid_size), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=0))   # (2, H, W), fixed
        self.net = nn.Sequential(
            nn.Conv2d(state_dim + 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),        nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),        nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, g = s.shape[0], self.grid_size
        s_map  = s.view(B, -1, 1, 1).expand(B, s.shape[1], g, g)        # broadcast state over the grid
        coords = self.coords.unsqueeze(0).expand(B, 2, g, g)
        return torch.sigmoid(self.net(torch.cat([s_map, coords], dim=1)))


# ## Gray-box dynamics

def ego_step(state: torch.Tensor, action: torch.Tensor, dt: float,
             a_v: torch.Tensor, a_omega: torch.Tensor) -> torch.Tensor:
    """Known unicycle step on [x, y, cos th, sin th] with learnable scales.

    Semi-implicit (rotate heading, then move along the new heading), matching
    sim/dynamics.py. a_v, a_omega are the gray-box coefficients: on the frictionless
    toy they should learn ~1; on a real car they absorb wheelbase / velocity scaling.
    Heading stays on the unit circle (a rotation), so no wrap handling is needed.
    """
    x, y = state[:, 0:1], state[:, 1:2]
    c, s = state[:, 2:3], state[:, 3:4]
    v, omega = action[:, 0:1], action[:, 1:2]
    w  = a_omega * omega * dt
    cw, sw = torch.cos(w), torch.sin(w)
    c_new = c * cw - s * sw
    s_new = s * cw + c * sw
    x_new = x + a_v * v * c_new * dt
    y_new = y + a_v * v * s_new * dt
    return torch.cat([x_new, y_new, c_new, s_new], dim=1)


class EgoWorldModel(nn.Module):
    """Encoder + renderer + gray-box dynamics.

    residual_budget > 0 enables a bounded learned correction on the dynamics
    (gray-box for unknown physics: friction/slip). 0 = pure known kinematics with
    learnable coefficients (right for the frictionless toy).
    """

    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT,
                 channels: tuple = ENCODER_CHANNELS, in_channels: int = IN_CHANNELS,
                 renderer_hidden: int = 512, residual_budget: float = 0.0,
                 learn_coeffs: bool = False, decoder: str = "mlp"):
        super().__init__()
        self.dt = dt
        self.residual_budget = residual_budget
        self.encoder  = StateEncoder(grid_size, channels, in_channels)
        # spatial-broadcast decoder can place an object at (x,y); the MLP cannot (it
        # collapses to the mean frame), so "broadcast" is the default.
        if decoder == "broadcast":
            self.renderer = SpatialBroadcastDecoder(STATE_DIM, grid_size)
        elif decoder == "mlp":
            self.renderer = Renderer(STATE_DIM, grid_size, renderer_hidden)
        else:
            raise ValueError(f"decoder must be 'broadcast' or 'mlp', got {decoder!r}")
        # gray-box physical coefficients (log-space, =1 at init). FROZEN by default:
        # the toy's kinematics are known exactly, and a learnable a_omega just collapses
        # to ~0 (a "don't rotate" escape hatch that lets the encoder leave heading
        # unencoded). Learn them only when the physics scaling is genuinely unknown.
        if learn_coeffs:
            self.log_a_v     = nn.Parameter(torch.zeros(()))
            self.log_a_omega = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_a_v",     torch.zeros(()))
            self.register_buffer("log_a_omega", torch.zeros(()))
        if residual_budget > 0:
            self.residual = nn.Sequential(
                nn.Linear(STATE_DIM + ACTION_DIM, 64), nn.ReLU(inplace=True),
                nn.Linear(64, STATE_DIM),
            )

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def render(self, s: torch.Tensor) -> torch.Tensor:
        return self.renderer(s)

    def step(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        nxt = ego_step(s, action, self.dt, self.log_a_v.exp(), self.log_a_omega.exp())
        if self.residual_budget > 0:
            nxt = nxt + self.residual_budget * torch.tanh(self.residual(torch.cat([s, action], dim=-1)))
        return nxt

    def forward(self, frame: torch.Tensor, action: torch.Tensor, next_frame: torch.Tensor) -> dict:
        s      = self.encode(frame)
        s_next = self.encode(next_frame)
        s_pred = self.step(s, action)
        return {
            "s": s, "s_next": s_next, "s_pred": s_pred,
            "recon": self.render(s),
            "pred_recon": self.render(s_pred),
        }


def _img_loss(pred: torch.Tensor, target: torch.Tensor, fg_weight: float = 0.0) -> torch.Tensor:
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


def ego_loss(out: dict, frame: torch.Tensor, next_frame: torch.Tensor,
             lam_dyn: float = 1.0, lam_pred: float = 1.0,
             lam_var: float = 0.0, var_gamma: float = 0.1,
             recon_fg_weight: float = 0.0, lam_recon: float = 1.0,
             s_target: torch.Tensor = None, s_next_target: torch.Tensor = None,
             lam_anchor: float = 0.0, lam_anchor_pred: float = 0.0) -> tuple:
    """recon (grounds state) + dyn (physics in state space) + pred_recon (predicted->pixels)
    + an optional variance floor that forbids any state dim from going dead
    + an optional supervised state anchor (privileged info: label-free at inference).

    The variance floor is the VICReg variance term: penalize a dim whose batch std drops
    below var_gamma. It is pure anti-collapse, it does NOT decorrelate the dims, so it
    keeps them interpretable, but it stops the encoder from zeroing out d0,d1 or caving
    the whole state to a constant under the collapse-prone dyn loss.

    ANCHOR (lam_anchor > 0): pull the encoded state toward a true-pose target s_target
    (=[x, y, cos th, sin th]) supplied only during training. This directly pins the 4
    dims to metric pose, so heading no longer relies on the weak dynamics signal, and the
    latent IS the state (no gauge freedom). s_next_target + lam_anchor_pred additionally
    supervises the PREDICTED next state, forcing the predictor to be real dynamics. At
    inference the encoder is camera-only; the anchor is a training-time crutch.
    lam_recon scales the frame reconstruction so it can be dropped (lam_recon=0) once the
    anchor, not the decoder, is the thing grounding the state.
    """
    recon = _img_loss(out["recon"], frame, recon_fg_weight)
    dyn   = F.mse_loss(out["s_pred"], out["s_next"].detach())
    pred  = _img_loss(out["pred_recon"], next_frame, recon_fg_weight)
    std   = (out["s"].var(0) + 1e-4).sqrt()                 # per-dim std over the batch (grad-safe)
    var   = F.relu(var_gamma - std).mean()                  # > 0 only for dims below the floor
    anchor      = F.mse_loss(out["s"], s_target) if (lam_anchor > 0 and s_target is not None) \
                  else torch.zeros((), device=out["s"].device)
    anchor_pred = F.mse_loss(out["s_pred"], s_next_target) if (lam_anchor_pred > 0 and s_next_target is not None) \
                  else torch.zeros((), device=out["s"].device)
    total = (lam_recon * recon + lam_dyn * dyn + lam_pred * pred + lam_var * var
             + lam_anchor * anchor + lam_anchor_pred * anchor_pred)
    parts = {"recon": recon.item(), "dyn": dyn.item(), "pred_recon": pred.item(),
             "var": var.item(), "anchor": anchor.item(), "anchor_pred": anchor_pred.item(),
             "total": total.item()}
    return total, parts


def ego_rollout_loss(model, frame: torch.Tensor, actions: torch.Tensor, poses: torch.Tensor,
                     lam_recon: float = 1.0, lam_anchor: float = 1.0, lam_rollout: float = 1.0,
                     recon_fg_weight: float = 0.0) -> tuple:
    """Multi-step rollout supervision (fixes the single-step a_v bias).

    Encode frame_0 ONCE, then roll the gray-box dynamics K steps over the action sequence
    (no re-encoding) and anchor EVERY rolled state to the true pose at that horizon. Unlike
    the single-step anchor_pred, a wrong a_v COMPOUNDS across the rollout, so the optimizer
    gets real gradient to pull a_v onto the true gain instead of leaving it loosely pinned.

    frame:   (B, 1, H, W)         the first frame
    actions: (B, K, 2)            the K held actions
    poses:   (B, K+1, 3)          true (x, y, theta) at horizons 0..K
    """
    def to_state(p):                                  # (B,3) -> (B,4) [x, y, cos, sin]
        return torch.stack([p[:, 0], p[:, 1], torch.cos(p[:, 2]), torch.sin(p[:, 2])], dim=1)

    s      = model.encode(frame)
    recon  = _img_loss(model.render(s), frame, recon_fg_weight)
    anchor = F.mse_loss(s, to_state(poses[:, 0]))     # ground the initial state
    K      = actions.shape[1]
    roll   = s.new_zeros(())
    for k in range(K):                                # open-loop rollout, anchored each step
        s = model.step(s, actions[:, k])
        roll = roll + F.mse_loss(s, to_state(poses[:, k + 1]))
    roll = roll / K
    total = lam_recon * recon + lam_anchor * anchor + lam_rollout * roll
    parts = {"recon": recon.item(), "anchor": anchor.item(), "rollout": roll.item(), "total": total.item()}
    return total, parts


# ## Smoke test
#   python -c "from models.state_ae import _test_state_ae; _test_state_ae()"

def _test_state_ae():
    torch.manual_seed(0)
    B, g = 8, 64
    for budget, learn, decoder in ((0.0, False, "broadcast"), (0.1, True, "mlp")):
        m = EgoWorldModel(grid_size=g, residual_budget=budget, learn_coeffs=learn, decoder=decoder)
        frame, action, nxt = torch.rand(B, 1, g, g), torch.rand(B, 2), torch.rand(B, 1, g, g)
        out = m(frame, action, nxt)
        assert out["s"].shape == (B, STATE_DIM)
        assert out["recon"].shape == (B, 1, g, g)
        # state is constrained: position in [0,1], heading unit norm
        assert (out["s"][:, :2] >= 0).all() and (out["s"][:, :2] <= 1).all()
        assert torch.allclose(out["s"][:, 2:].norm(dim=1), torch.ones(B), atol=1e-4)
        total, parts = ego_loss(out, frame, nxt, lam_var=1.0)
        m.zero_grad(); total.backward()
        assert m.encoder.head.weight.grad.abs().sum() > 0
        assert m.renderer.net[0].weight.grad.abs().sum() > 0      # decoder gets gradient
        if learn:                                  # coefficients get gradient only when learnable
            assert m.log_a_v.grad is not None
        print(f"decoder={decoder} budget={budget} learn_coeffs={learn}  total={parts['total']:.4f}  "
              f"recon={parts['recon']:.4f}  dyn={parts['dyn']:.4f}  var={parts['var']:.4f}")

    # ego_step sanity vs the simulator's convention
    s = torch.tensor([[0.5, 0.5, 1.0, 0.0]])               # at center, facing +x
    one = torch.ones(())
    nb = ego_step(s, torch.tensor([[1.0, 0.0]]), 0.1, one, one)
    assert torch.allclose(nb, torch.tensor([[0.6, 0.5, 1.0, 0.0]]), atol=1e-5), nb
    nb = ego_step(s, torch.tensor([[0.0, (torch.pi / 2) / 0.1]]), 0.1, one, one)  # quarter turn
    assert torch.allclose(nb[:, 2:], torch.tensor([[0.0, 1.0]]), atol=1e-5), nb
    print("All state_ae tests passed.")


if __name__ == "__main__":
    _test_state_ae()
