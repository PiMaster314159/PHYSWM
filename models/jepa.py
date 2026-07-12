# # JEPA model
# 
# Encoder + Predictor for LeWM training.
# 
# Pipeline: `(frame_t, action_t, frame_{t+1})` -> encoder -> predictor -> latent prediction. The encoder maps pixels to latents. The predictor maps `(z_t, action_t)` to a predicted `z_{t+1}`. Training minimizes the distance between that prediction and the actual encoded next frame.
# 

# ## Path setup
# 
# Same root-finding pattern as `dataset.py`.


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.base import WorldModel
from models.components import sigreg_loss, state_to_target, conv_trunk, mlp, unstandardize

from config import (
    DATA_PATH, DT, PRED_STEP,
    GRID_SIZE, IN_CHANNELS, LATENT_DIM, ENCODER_CHANNELS,
    PREDICTOR_HIDDEN, ACTION_DIM, PREDICTOR_MODE, PHYSICS_LOCK_POSE, LAM_PHYS,
)


# ## Imports


import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Encoder
# 
# CNN: `(B, 1, H, W)` -> `(B, latent_dim)`.
# 
# Three stride-2 conv layers halve spatial dims and grow channel count at each stage. Flattened size is measured once at init via a dummy forward pass so the same class works at any grid size without changes.


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

        self.conv, flat_dim = conv_trunk(grid_size, in_channels, channels)
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


class Predictor(nn.Module):
    """Predicts the next latent as base(z, a) + a learned MLP correction.

    The `mode` selects the base:
      "mlp"      base = 0       predict the next latent from scratch
      "residual" base = z       predict the change; identity default
      "physics"  base = z + a learnable-scaled unicycle step on dims 0, 1, 2

    Physics base (dims 0, 1, 2 read as x, y, theta), no ground-truth leak:
        theta = a_theta * z2
        z0_next = z0 + a_pos * dt * v * cos(theta)
        z1_next = z1 + a_pos * dt * v * sin(theta)
        z2_next = z2 + (dt * omega) / a_theta
    a_pos, a_theta are learnable scalars (log-space, =1 at init) bridging latent
    units (BatchNorm unit-variance) and physical units (world units / radians).

    lock_pose (physics only): if True, the MLP cannot correct dims 0, 1, 2, so
    the predicted pose is *purely* the kinematics. This is the hard architectural
    prior: the prediction loss can only be minimized if the encoder puts a
    faithful pose in dims 0-2, since the MLP is no longer allowed to paper over it.

    Parameters
    ----------
    latent_dim, action_dim, hidden : ints.
    mode : {"mlp", "residual", "physics"}
    dt : float
        Physics timestep. Defaults to PRED_STEP * DT (multi-step horizon).
    lock_pose : bool
        Physics only. Mask the MLP off dims 0, 1, 2.

    Defaults pull from config.py.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = PREDICTOR_HIDDEN,
        mode: str = PREDICTOR_MODE,
        dt: float = PRED_STEP * DT,
        lock_pose: bool = PHYSICS_LOCK_POSE,
    ):
        super().__init__()
        if mode not in ("mlp", "residual", "physics"):
            raise ValueError(f"mode must be mlp/residual/physics, got {mode!r}")
        if mode == "physics" and latent_dim < 3:
            raise ValueError("physics mode needs latent_dim >= 3 (uses dims 0,1,2)")
        self.mode = mode
        self.dt   = dt
        self.net  = mlp([latent_dim + action_dim, hidden, hidden, latent_dim])
        self.lock_pose = (mode == "physics" and lock_pose)
        if mode == "physics":
            # learnable unit-bridging scales, log-space so they stay positive (=1 at init)
            self.log_a_pos   = nn.Parameter(torch.zeros(()))
            self.log_a_theta = nn.Parameter(torch.zeros(()))
            if self.lock_pose:
                mask = torch.ones(latent_dim)
                mask[:3] = 0.0                       # MLP zeroed on the pose dims
                self.register_buffer("_pose_mask", mask)

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
        if self.lock_pose:
            learned = learned * self._pose_mask   # pose dims governed purely by kinematics
        return self._physics_base(z, action) + learned


# ## JEPA
# 
# Wires encoder and predictor together. Forward returns everything the loss needs: current latent, predicted next latent, and actual encoded next latent.
# 
# Stop-gradient on the target is applied in `jepa_loss`, not here, keeping this a pure forward pass.


class JEPA(WorldModel):
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
    predictor_lock_pose : bool
        Physics only. Lock dims 0,1,2 to pure kinematics (no MLP correction).

    Defaults pull from config.py.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        latent_dim: int = LATENT_DIM,
        encoder_channels: tuple = ENCODER_CHANNELS,
        predictor_hidden: int = PREDICTOR_HIDDEN,
        predictor_mode: str = PREDICTOR_MODE,
        predictor_lock_pose: bool = PHYSICS_LOCK_POSE,
        state_head: bool = False,
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
            lock_pose=predictor_lock_pose,
        )
        # optional LINEAR state-readout head (privileged supervision that does NOT fight
        # SIGReg: it reads pose out of the whole latent, so the code can stay distributed
        # and isotropic while still being linearly decodable to pose). Linear (not MLP) on
        # purpose, so a low readout loss means pose is genuinely linearly present.
        self.state_head = nn.Linear(latent_dim, 4) if state_head else None

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, latent_dim)."""
        return self.encoder(frame)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(z: (B, D), action: (B, 2)) -> predicted next latent (B, D)."""
        return self.predictor(z, action)

    def decode_pose(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent) -> (B, 4) real-unit pose [x,y,cosθ,sinθ]. The linear probe outputs
        STANDARDIZED pose; the stored stats bring it back to world units."""
        return unstandardize(self.state_head(z), self.pose_mean, self.pose_std)

    def extra_forward(self, frame, action, next_frame, out):
        """Physics mode only: the kinematic step of z (for the physics-consistency term).
        Pose readout is handled by decode_pose, so no state_hat here anymore."""
        if self.predictor.mode == "physics":
            return {"phys_next_z": self.predictor._physics_base(out["z"], action)}
        return {}

    def representation_loss(self, out, batch, weights):
        """JEPA's own terms: prediction MSE + SIGReg (+ physics consistency). The pose
        readout/readout_pred are the shared WorldModel.pose_supervision (via decode_pose)."""
        pred  = F.mse_loss(out["pred_next_z"], out["target_next_z"].detach())
        sig   = sigreg_loss(out["z"])
        total = pred + weights.get("sigreg", 1.0) * sig
        parts = {"pred": pred.item(), "sigreg": sig.item()}
        if "phys_next_z" in out:
            phys = F.mse_loss(out["phys_next_z"][:, :3], out["target_next_z"][:, :3].detach())
            total = total + weights.get("phys", 0.0) * phys
            parts["phys"] = phys.item()
        return total, parts


# ## Loss
# 
# Three terms (the third only in physics mode).
# 
# **Prediction loss**: MSE between the predicted next latent and the encoded next frame. Stop-gradient on the target prevents both ends of the shared encoder collapsing to a constant.
# 
# **SIGReg**: the Sketched-Isotropic-Gaussian Regularizer. Projects the latents onto random unit directions and applies the Epps-Pulley normality test; by Cramer-Wold, matching all 1D marginals to N(0,1) matches the joint to N(0, I). Provable anti-collapse.
# 
# **Physics consistency** (`lam_phys > 0`, physics mode): pulls the encoded *next* pose (`target_next_z` dims 0,1,2) toward the kinematic step of the *current* latent (`phys_next_z` dims 0,1,2). Minimizing it requires the next-frame encoding's position to equal `z + dt·v·cos(z2)...`, which requires `z2` to be a real heading. This is the soft version of the architectural pose lock, and the weight is the "physics ratio" you can crank.


def jepa_loss(model, out: dict, lam: float = 1.0, lam_phys: float = 0.0,
              s_target: torch.Tensor = None, lam_anchor: float = 0.0,
              lam_readout: float = 0.0, s_next_target: torch.Tensor = None,
              lam_readout_pred: float = 0.0) -> tuple:
    """Thin adapter: pack the args into (batch, weights) and call model.loss. The actual
    terms live in JEPA.representation_loss + WorldModel.pose_supervision. s_target/s_next_target
    are RAW pose [x,y,cosθ,sinθ]; pose_supervision standardizes them with the model's buffers,
    so JEPA's readout equals the old (state_head vs standardized-pose) loss exactly. Pass RAW
    targets and set the model's pose stats (model.set_pose_stats) before training.
    (lam_anchor, the direct dims-0..K pinning, is dropped: superseded by the readout path.)"""
    batch   = {"s_target": s_target, "s_next_target": s_next_target}
    weights = {"sigreg": lam, "phys": lam_phys, "anchor": lam_readout, "anchor_pred": lam_readout_pred}
    return model.loss(out, batch, weights)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ## Tests
# 
# Shape checks on random and real batches, resolution-agnosticism, action-sensitivity, gradient wiring, and loss finiteness.


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
            ll, _ = jepa_loss(mm, oo)
            mm.zero_grad()
            ll.backward()
            assert mm.predictor.log_a_pos.grad   is not None, "a_pos got no gradient"
            assert mm.predictor.log_a_theta.grad is not None, "a_theta got no gradient"
    print("predictor modes mlp/residual/physics build + run OK (physics scales trainable)")

    # physics: pose lock + consistency loss
    locked = JEPA(grid_size=40, latent_dim=D, predictor_mode="physics", predictor_lock_pose=True)
    o = locked(frame, action, next_frame)
    assert "phys_next_z" in o, "physics forward should expose phys_next_z"
    # locked: predicted pose dims equal the pure kinematic base (MLP masked off 0,1,2)
    assert torch.allclose(o["pred_next_z"][:, :3], o["phys_next_z"][:, :3], atol=1e-5), \
        "lock_pose leaked MLP into the pose dims"
    lp, parts = jepa_loss(locked, o, lam_phys=1.0)
    assert "phys" in parts and torch.isfinite(lp), "physics-consistency term failed"
    print(f"physics lock_pose + lam_phys OK  (phys={parts['phys']:.4f})")

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
    loss, parts = jepa_loss(model, out)
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
        loss, parts = jepa_loss(model, out)
        assert torch.isfinite(loss)
        print(f"real batch {tuple(batch['frame'].shape)} -> "
              f"pred_next_z {tuple(out['pred_next_z'].shape)}  "
              f"loss={parts['total']:.4f}  pred={parts['pred']:.4f}  sigreg={parts['sigreg']:.4f}")
    else:
        print(f"(skipping real-batch test: {DATA_PATH} not found)")

    print("All JEPA tests passed.")


if __name__ == "__main__":
    _test_jepa()

