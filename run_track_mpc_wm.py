"""World-model MPC comparison on the straight-track task.

Same MPPI controller and track cost as run_track_mpc.py, but the INTERNAL model the controller
plans with is a trained world model. Methodology: **plan with model M, apply to the TRUE system.**
Each control step:
  1. render the true state to a frame (the camera; numpy sim renderer)
  2. the model observes it -> its state s0  (ego/grounded: physical dims; JEPA: a latent)
  3. MPPI rolls the MODEL forward and scores with the track cost -> best action
  4. apply the first action to the TRUE dynamics (numpy sim); repeat (receding horizon)

The cost is a function of POSE [x, y, theta]. For ego/grounded the model state IS pose (a slice);
for JEPA the state is an abstract latent, so `pose()` must run the readout PROBE to decode it -
concretely why JEPA is harder to use for control than the physical-state models.

Only the neural models need torch (the tower); `--model unicycle` (perfect known dynamics) runs in
pure numpy and verifies the whole loop on any machine.

    python run_track_mpc_wm.py --model unicycle                                   # numpy oracle
    python run_track_mpc_wm.py --model ego      --ckpt <ego.pt>   --grid-size 64
    python run_track_mpc_wm.py --model grounded --ckpt <grn.pt>   --grid-size 64
    python run_track_mpc_wm.py --model jepa     --ckpt <jepa.pt>  --grid-size 64
    python run_track_mpc_wm.py --model unicycle,ego,grounded,jepa --ckpt ... (see --ckpt-* flags)  # overlay
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from sim.render import render_frame
from sim.dynamics import step as true_step          # ground-truth unicycle (the real system)
from control.mppi import mppi_plan


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def pose_from_xycs(S):                                # (K,>=4) [x,y,cos,sin,..] -> (K,3) [x,y,theta]
    return np.stack([S[:, 0], S[:, 1], np.arctan2(S[:, 3], S[:, 2])], axis=1)


# ---------------------------------------------------------------------------
# Controllers: each returns (observe, dynamics, pose, state_dim, label).
#   observe(true_state, frame) -> s0   (model's state, numpy (d,))
#   dynamics(S (K,d), A (K,1)=omega) -> S' (K,d)
#   pose(S (K,d)) -> (K,3) [x,y,theta]
# ---------------------------------------------------------------------------
def build_unicycle(a):
    dt, v = a.dt, a.v
    def observe(true_state, frame):
        return np.asarray(true_state, float).copy()
    def dynamics(S, A):
        x, y, th = S[:, 0], S[:, 1], S[:, 2]
        th2 = th + A[:, 0] * dt
        return np.stack([x + v * np.cos(th2) * dt, y + v * np.sin(th2) * dt, th2], axis=1)
    def pose(S):
        return S[:, :3].copy()
    return observe, dynamics, pose, 3, "unicycle (oracle)"


def build_neural(name, a):
    """ego / grounded / jepa. Torch-only; runs on the tower."""
    import torch
    dev = a.device
    g = a.grid_size
    v = a.v

    if name == "ego":
        from models.state_ae import EgoWorldModel
        model = EgoWorldModel(grid_size=g, dt=a.dt, learn_coeffs=True, decoder="mlp")
        physical = True; latent_dim = 4; label = "ego (gray-box)"
    elif name == "grounded":
        from models.grounded import GroundedJEPA
        model = GroundedJEPA(grid_size=g, latent_dim=C.LATENT_DIM, block_dim=4, dt=a.dt, learn_coeffs=True)
        physical = True; latent_dim = C.LATENT_DIM; label = "grounded"
    elif name == "jepa":
        from models.jepa import JEPA
        model = JEPA(grid_size=g, latent_dim=C.LATENT_DIM, predictor_mode="residual", state_head=True)
        physical = False; latent_dim = C.LATENT_DIM; label = "residual JEPA (probe)"
    else:
        raise ValueError(f"unknown neural model {name!r}")

    sd = torch.load(a.ckpt_for(name), map_location=dev, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()

    # decode_pose un-standardizes JEPA's readout with the model's pose_mean/std buffers. Archived
    # checkpoints predate those buffers (they load as the 0/1 default), so back-fill them from the
    # dataset once, keeping decode_pose self-contained (new checkpoints carry the stats themselves).
    if not physical and torch.allclose(model.pose_std, torch.ones_like(model.pose_std)):
        import h5py
        from models.components import pose_stats
        with h5py.File(C.DATASETS_DIR / f"{a.run}.h5", "r") as f:
            states = torch.from_numpy(f["states"][:]).float()
        model.set_pose_stats(*pose_stats(states))

    def to_frame(frame):
        return torch.from_numpy(np.asarray(frame, np.float32)).view(1, 1, g, g).to(dev)

    def full_action(A):                               # (K,1)=omega -> (K,2)=[v,omega] torch
        K = A.shape[0]
        vv = torch.full((K, 1), float(v))
        om = torch.from_numpy(A.astype(np.float32))
        return torch.cat([vv, om], dim=1).to(dev)

    @torch.no_grad()
    def observe(true_state, frame):
        return model.encode(to_frame(frame)).cpu().numpy()[0]

    @torch.no_grad()
    def dynamics(S, A):
        s = torch.from_numpy(S.astype(np.float32)).to(dev)
        s2 = model.predict(s, full_action(A)) if hasattr(model, "predict") else model.step(s, full_action(A))
        return s2.cpu().numpy()

    @torch.no_grad()
    def pose(S):                                      # one path for all models: decode -> [x,y,theta]
        z = torch.from_numpy(S.astype(np.float32)).to(dev)
        return pose_from_xycs(model.decode_pose(z).cpu().numpy())

    return observe, dynamics, pose, latent_dim, label


def make_costs(pose_fn, y_c, half, a):
    def running(S, A):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2]); om = A[:, 0]
        viol = np.maximum(0.0, np.abs(lat) - half)
        return a.w_lat*lat**2 + a.w_head*head**2 + a.w_ctrl*om**2 + a.w_bound*viol**2
    def terminal(S):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2])
        viol = np.maximum(0.0, np.abs(lat) - half)
        return a.term_scale*(a.w_lat*lat**2 + a.w_head*head**2) + a.w_bound*viol**2
    return running, terminal


def run_one(name, a, y_c, half):
    observe, dynamics, pose, d, label = (build_unicycle(a) if name == "unicycle" else build_neural(name, a))
    running, terminal = make_costs(pose, y_c, half, a)
    rng = np.random.default_rng(a.seed)
    a_low = np.array([-a.omega_max]); a_high = np.array([a.omega_max])

    true = np.array([a.x0, y_c + a.y0, np.deg2rad(a.theta0_deg)], float)   # TRUE state (world [0,1])
    a_nom = np.zeros((a.horizon, 1))
    traj = [true.copy()]
    for _ in range(a.steps):
        frame = render_frame(true, grid_size=a.grid_size, marker=a.marker)
        s0 = observe(true, frame)                      # model's estimate of the current state
        a_nom, _ = mppi_plan(s0, a_nom, dynamics, running, terminal,
                             n_samples=a.samples, sigma=a.sigma, lam=a.lam,
                             a_low=a_low, a_high=a_high, rng=rng)
        omega = float(a_nom[0, 0])
        true = true_step(true, np.array([a.v, omega]), a.dt)   # apply first action to TRUTH
        traj.append(true.copy())
        a_nom = np.roll(a_nom, -1, axis=0); a_nom[-1] = 0.0

    traj = np.array(traj)
    y_err = traj[:, 1] - y_c
    metrics = {"model": label,
               "final_lat": abs(y_err[-1]),
               "final_head_deg": abs(np.rad2deg(wrap(traj[-1, 2]))),
               "max_excursion": np.abs(y_err).max(),
               "in_bounds": bool(np.abs(y_err).max() <= half + 1e-6),
               "rms_lat": float(np.sqrt((y_err**2).mean()))}
    return traj, metrics


def parse_args():
    p = argparse.ArgumentParser(description="World-model MPC comparison on the straight track.")
    p.add_argument("--model", default="unicycle", help="comma list: unicycle,ego,grounded,jepa")
    p.add_argument("--ckpt-ego", default=None); p.add_argument("--ckpt-grounded", default=None)
    p.add_argument("--ckpt-jepa", default=None); p.add_argument("--ckpt", default=None,
                   help="shortcut when running a single --model")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--marker", default="ring")
    p.add_argument("--run", default="run07_64x64_ring",
                   help="dataset whose pose stats de-standardize the JEPA readout (match its training data)")
    p.add_argument("--dt", type=float, default=C.DT)
    p.add_argument("--v", type=float, default=0.4, help="fixed forward speed (world [0,1] units)")
    p.add_argument("--track-width", type=float, default=0.4, help="track width (|y - y_c| <= W/2)")
    p.add_argument("--x0", type=float, default=0.15); p.add_argument("--y0", type=float, default=0.0,
                   help="initial lateral offset from centerline")
    p.add_argument("--theta0-deg", type=float, default=45.0)
    p.add_argument("--horizon", type=int, default=25); p.add_argument("--samples", type=int, default=800)
    p.add_argument("--sigma", type=float, default=0.8); p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=2.5); p.add_argument("--steps", type=int, default=18)
    p.add_argument("--w-lat", type=float, default=6.0); p.add_argument("--w-head", type=float, default=1.0)
    p.add_argument("--w-ctrl", type=float, default=0.05); p.add_argument("--w-bound", type=float, default=60.0)
    p.add_argument("--term-scale", type=float, default=4.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    def ckpt_for(name):
        specific = {"ego": a.ckpt_ego, "grounded": a.ckpt_grounded, "jepa": a.ckpt_jepa}[name]
        path = specific or a.ckpt
        if path is None:
            raise SystemExit(f"no checkpoint for '{name}': pass --ckpt-{name} or --ckpt")
        return path
    a.ckpt_for = ckpt_for
    return a


def main():
    a = parse_args()
    y_c = 0.5
    half = a.track_width / 2.0
    models = [m.strip() for m in a.model.split(",") if m.strip()]
    if any(m != "unicycle" for m in models):          # torch only needed for neural models
        import torch
        if a.device != "cpu" and not torch.cuda.is_available():
            a.device = "cpu"

    print(f"=== world-model MPC: straight track (start y0={a.y0:+.2f}, theta={a.theta0_deg:.0f} deg) ===")
    results = []
    for name in models:
        traj, mt = run_one(name, a, y_c, half)
        results.append((mt["model"], traj, mt))
        print(f"  {mt['model']:22s}  final_lat {mt['final_lat']:.3f}  "
              f"final_head {mt['final_head_deg']:5.1f} deg  max|y| {mt['max_excursion']:.3f}  "
              f"rms_lat {mt['rms_lat']:.3f}  {'IN' if mt['in_bounds'] else 'OUT of'} bounds")

    # overlay figure
    report = ROOT / "results" / "track_mpc"; report.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhspan(y_c - half, y_c + half, color="0.92", label="track")
    ax.axhline(y_c, color="0.6", ls="--", lw=1)
    colors = ["#c0392b", "#2c7fb8", "#31a354", "#e08214", "#6a51a3"]
    for i, (label, traj, mt) in enumerate(results):
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=colors[i % len(colors)], label=label)
    ax.plot(results[0][1][0, 0], results[0][1][0, 1], "ks", ms=7)
    ax.set_xlabel("x (down-track)"); ax.set_ylabel("y")
    ax.set_title(f"Plan-with-model, apply-to-truth (theta0={a.theta0_deg:.0f} deg)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = report / f"wm_mpc_compare_theta{int(a.theta0_deg)}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
