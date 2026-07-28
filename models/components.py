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


def state_to_target(states: torch.Tensor, velocities: torch.Tensor = None) -> torch.Tensor:
    """(N,3) (x,y,theta) -> (N,4) (x,y,cos,sin). With `velocities` (N,) or (N,1), appends v -> (N,5) for the
    bicycle block [x,y,cos,sin,v]. Shared anchor-target builder."""
    x, y, th = states[:, 0], states[:, 1], states[:, 2]
    cols = [x, y, torch.cos(th), torch.sin(th)]
    if velocities is not None:
        cols.append(velocities.reshape(-1))
    return torch.stack(cols, dim=1)

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

def pose_stats(states, velocities=None):
    """(N,3) -> standardization mean/std over [x,y,cosθ,sinθ]; with velocities, over [...,v] (bicycle)."""
    T = state_to_target(states, velocities)
    return T.mean(0), T.std(0) + 1e-6

def standardize(t, mean, std):   return (t - mean) / std
def unstandardize(t, mean, std): return t * std + mean


def pearson_per_dim(Z, t):
    """Per-dim Pearson correlation of each latent dim in Z (N,D) with a 1-D target t (N,).
    Returns (D,). Shared by the ego + grounded eval hooks."""
    Zc = Z - Z.mean(0, keepdim=True)
    tc = t - t.mean()
    return (Zc * tc.unsqueeze(1)).mean(0) / (Z.std(0) * t.std() + 1e-8)

def conv_trunk(grid_size, in_channels, channels):
    layers, c = [], in_channels
    for c_out in channels:
        layers += [nn.Conv2d(c, c_out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]; c = c_out
    trunk = nn.Sequential(*layers)
    with torch.no_grad():
        flat = trunk(torch.zeros(1, in_channels, grid_size, grid_size)).flatten(1).shape[1]
    return trunk, flat

def constrain_pose(raw, eps=1e-6):
    """[x,y] into [0,1] (sigmoid), heading onto the unit circle. If raw has a 5th column (bicycle block),
    constrain velocity to >=0 (softplus) and append it."""
    out = torch.cat([torch.sigmoid(raw[:, :2]), F.normalize(raw[:, 2:4], dim=1, eps=eps)], dim=1)
    if raw.shape[1] >= 5:
        out = torch.cat([out, F.softplus(raw[:, 4:5])], dim=1)   # v >= 0
    return out


def unicycle_step(state, action, dt, a_v=1.0, a_omega=1.0, eps=1e-6):
    """Semi-implicit unicycle step on [x, y, cosθ, sinθ] (matches sim/dynamics.py): rotate
    the heading by a_omega·ω·dt, then move along the NEW heading at a_v·v. Heading stays on
    the unit circle (a rotation), so no wrap handling. a_v, a_omega are gray-box scales
    (default 1 = pure known kinematics). Shared by ego (whole state) and grounded (block)."""
    x, y = state[:, 0:1], state[:, 1:2]
    c, s = state[:, 2:3], state[:, 3:4]
    n    = torch.clamp(torch.sqrt(c * c + s * s), min=eps)   # normalize input heading (no-op when already unit)
    c, s = c / n, s / n
    v, omega = action[:, 0:1], action[:, 1:2]
    cw, sw = torch.cos(a_omega * omega * dt), torch.sin(a_omega * omega * dt)
    c_new  = c * cw - s * sw
    s_new  = s * cw + c * sw
    x_new  = x + a_v * v * c_new * dt
    y_new  = y + a_v * v * s_new * dt
    return torch.cat([x_new, y_new, c_new, s_new], dim=1)


def bicycle_step(state, action, dt, wheelbase, a_accel=1.0, v_max=None, eps=1e-6):
    """Kinematic-bicycle step on the 5-D block [x, y, cosθ, sinθ, v]; action [a, delta] (accel, steering).
    Velocity integrates the throttle (v_new = v + a_accel*a*dt, clamped >=0); heading turns by
    (v_new/L)*tan(delta)*dt (yaw scales with speed); position moves along the new heading. Semi-implicit,
    matching unicycle_step. a_accel is the gray-box throttle gain (=1 = exact known dynamics). The torch
    counterpart to sim.dynamics.bicycle_step."""
    x, y = state[:, 0:1], state[:, 1:2]
    c, s = state[:, 2:3], state[:, 3:4]
    v     = state[:, 4:5]
    n = torch.clamp(torch.sqrt(c * c + s * s), min=eps); c, s = c / n, s / n
    a, delta = action[:, 0:1], action[:, 1:2]
    v_new = torch.clamp(v + a_accel * a * dt, min=0.0)
    if v_max is not None:
        v_new = torch.clamp(v_new, max=v_max)
    dtheta = (v_new / wheelbase) * torch.tan(delta) * dt        # yaw rate scales with speed
    cw, sw = torch.cos(dtheta), torch.sin(dtheta)
    c_new  = c * cw - s * sw
    s_new  = s * cw + c * sw
    x_new  = x + v_new * c_new * dt
    y_new  = y + v_new * s_new * dt
    return torch.cat([x_new, y_new, c_new, s_new, v_new], dim=1)


def mlp(sizes, out_act=None):
    """Linear/ReLU stack. sizes=[in, h1, ..., out]; ReLU between layers, optional final
    activation module (e.g. nn.Sigmoid()). Layer indices match the old explicit Sequentials,
    so existing state_dict keys (net.0, net.2, ...) still load."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class MLPDecoder(nn.Module):
    """MLP: a low-dim vector (ego state / grounded block) -> frame in [0,1]. Merges the old
    Renderer and BlockDecoder (identical structure)."""

    def __init__(self, in_dim, grid_size, hidden=512):
        super().__init__()
        self.grid_size = grid_size
        self.net = mlp([in_dim, hidden, hidden, grid_size * grid_size], out_act=nn.Sigmoid())

    def forward(self, v):
        g = self.grid_size
        return self.net(v).view(-1, 1, g, g)


class StructuredResidual(nn.Module):
    """Body-frame physics-basis residual for the gray-box dynamics: captures higher-order effects
    we don't fully know (drag ~ v^2, slip ~ v*w, understeer ~ w^2) as a BOUNDED, SPARSE combination
    of physical monomials, ADDED to the exact-kinematics next pose [x, y, cosθ, sinθ].

    Basis Phi(v, w) = [1, v, w, v^2, v*w, w^2]  (degree 2; degree 3 adds the four cubics).
    Coefficients are global learnable scalars by default; if free_dim > 0 they are read from the free
    latent (per-context c_k(free)). The constant term Phi_0=1 covers a steady offset, so no separate
    'latents-only' residual is needed. Zero-init so the residual starts as identity. .l1() regularizes
    the coefficients for readable, sparse terms. Output is bounded by budget*tanh so it patches the
    gap without swallowing the known kinematics or running away.
    """
    def __init__(self, dt, budget=0.1, free_dim=0, degree=2):
        super().__init__()
        self.dt, self.budget, self.free_dim, self.degree = dt, budget, free_dim, degree
        self.n_basis = 6 if degree < 3 else 10
        if free_dim > 0:
            self.coef_head = nn.Linear(free_dim, 3 * self.n_basis)   # 3 body channels x n_basis
            nn.init.zeros_(self.coef_head.weight); nn.init.zeros_(self.coef_head.bias)
        else:
            self.coef = nn.Parameter(torch.zeros(3, self.n_basis))

    def _basis(self, v, w):
        terms = [torch.ones_like(v), v, w, v * v, v * w, w * w]
        if self.degree >= 3:
            terms += [v * v * v, v * v * w, v * w * w, w * w * w]
        return torch.cat(terms, dim=-1)                     # (B, n_basis)

    def forward(self, pose, action, free=None):
        v, w = action[:, 0:1], action[:, 1:2]
        Phi = self._basis(v, w)
        if self.free_dim > 0:
            c = self.coef_head(free).view(-1, 3, self.n_basis)
            raw = (c * Phi.unsqueeze(1)).sum(-1)            # (B, 3)  latent-modulated
        else:
            raw = Phi @ self.coef.t()                       # (B, 3)  global
        d = self.budget * torch.tanh(raw)                   # (B, 3): d_forward, d_lateral, d_omega
        cos, sin = pose[:, 2:3], pose[:, 3:4]
        d_fwd, d_lat, d_om = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        dx = d_fwd * cos - d_lat * sin                      # body frame -> world by heading
        dy = d_fwd * sin + d_lat * cos
        dcos = -sin * d_om * self.dt                        # 1st-order heading rotation
        dsin =  cos * d_om * self.dt
        return torch.cat([dx, dy, dcos, dsin], dim=1)       # (B, 4) correction to add to next pose

    def l1(self):
        w = self.coef_head.weight if self.free_dim > 0 else self.coef
        return w.abs().mean()

    _BASIS = ["1", "v", "w", "v^2", "v*w", "w^2", "v^3", "v^2*w", "v*w^2", "w^3"]
    _CHANNELS = ["forward", "lateral", "omega"]

    @torch.no_grad()
    def named_coeffs(self, free=None):
        """{channel: {basis_term: coefficient}}. Global residuals read self.coef directly; latent-
        modulated ones need a (B, free_dim) free latent and are averaged over the batch."""
        if self.free_dim > 0:
            if free is None:
                raise ValueError("latent-modulated residual needs a free latent to evaluate coefficients")
            c = self.coef_head(free).view(-1, 3, self.n_basis).mean(0)
        else:
            c = self.coef
        names = self._BASIS[:self.n_basis]
        return {ch: {names[k]: c[i, k].item() for k in range(self.n_basis)} for i, ch in enumerate(self._CHANNELS)}


def format_residual(residual, free=None, thresh=1e-3):
    """Human-readable dump of a StructuredResidual's learned law. Drag shows up as a NEGATIVE
    v^2 coefficient on the 'forward' channel; slip as v*w on 'lateral'; understeer as w^2 on 'omega'."""
    lines = ["-- recovered gray-box residual g (raw coeff on each basis term; drag = negative v^2 on 'forward') --"]
    for ch, terms in residual.named_coeffs(free).items():
        top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
        s = "  ".join(f"{n}={v:+.3f}" for n, v in top if abs(v) > thresh)
        lines.append(f"  {ch:8s}  {s or '(all ~0)'}")
    return "\n".join(lines)


class MLPResidual(nn.Module):
    """Unstructured neural counterpart to StructuredResidual: same role (a bounded pose correction
    added to the exact-kinematics next pose), but with NO fixed physics basis and NO named terms.
    A small MLP maps (heading cosθ,sinθ ; action v,ω ; optional free latent) -> bounded Δ[x,y,cosθ,sinθ].

    This is the control condition for the g-ablation. It sees the SAME inputs and has the SAME output
    scale (budget*tanh) as the structured residual, so the only thing that differs between them is the
    inductive bias: a fixed monomial basis with sparse, readable coefficients vs. a free-form net.
    (x, y are deliberately excluded so the correction is translation-invariant, exactly like the
    structured one.) Zero-init on the last layer -> the residual starts as identity.
    """
    def __init__(self, dt, budget=0.1, free_dim=0, hidden=64):
        super().__init__()
        self.dt, self.budget, self.free_dim = dt, budget, free_dim
        self.net = mlp([4 + free_dim, hidden, hidden, 4])          # [cos, sin, v, w] (+ free) -> Δpose
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, pose, action, free=None):
        cos, sin = pose[:, 2:3], pose[:, 3:4]
        v, w = action[:, 0:1], action[:, 1:2]
        feats = [cos, sin, v, w] + ([free] if self.free_dim > 0 else [])
        return self.budget * torch.tanh(self.net(torch.cat(feats, dim=1)))   # (B, 4) bounded correction

    def l1(self):
        return self.net[0].weight.abs().mean()   # parity with StructuredResidual.l1(): keep the correction small


def make_residual(mode, dt, budget, free_dim):
    """Factory shared by ego + grounded: 'basis' -> StructuredResidual, 'mlp' -> MLPResidual, else None."""
    b = budget or 0.1
    if mode == "basis":
        return StructuredResidual(dt, budget=b, free_dim=free_dim)
    if mode == "mlp":
        return MLPResidual(dt, budget=b, free_dim=free_dim)
    return None
