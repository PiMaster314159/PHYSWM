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
    def bound_cost(lat):
        if getattr(a, "hard_bound", False):     # HARD: any predicted step outside -> ~inf -> MPPI rejects it
            return np.where(np.abs(lat) > half, 1e6, 0.0)
        viol = np.maximum(0.0, np.abs(lat) - half)      # SOFT: quadratic penalty past the edge
        return a.w_bound * viol**2
    def running(S, A):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2]); om = A[:, 0]
        return a.w_lat*lat**2 + a.w_head*head**2 + a.w_ctrl*om**2 + bound_cost(lat)
    def terminal(S):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2])
        return a.term_scale*(a.w_lat*lat**2 + a.w_head*head**2) + bound_cost(lat)
    return running, terminal


def build_one(name, a):
    observe, dynamics, pose, d, label = (build_unicycle(a) if name == "unicycle" else build_neural(name, a))
    return observe, dynamics, pose, label


def roll_episode(built, a, y_c, half):
    observe, dynamics, pose, label = built
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
        true = true_step(true, np.array([a.actuator_gain * a.v, omega]), a.dt)   # actuator gain on speed only (truth)
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


def run_one(name, a, y_c, half):
    return roll_episode(build_one(name, a), a, y_c, half)


def _sweep_color(label):
    l = label.lower()
    return ("#898781" if "oracle" in l else "#2a78d6" if "ego" in l
            else "#1baf7a" if "grounded" in l else "#eb6834")


def _short_name(label):
    l = label.lower()
    return ("unicycle" if "oracle" in l or "unicycle" in l else
            "jepa" if "jepa" in l else "grounded" if "grounded" in l else "ego")


def _save_coords(traj, y_c, path):
    """Per-step rollout coords: t, x, y, heading, lateral error, heading error."""
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["t", "x", "y", "theta_rad", "lateral_err", "heading_deg"])
        for k, (x, y, th) in enumerate(traj):
            w.writerow([k, f"{x:.5f}", f"{y:.5f}", f"{th:.5f}", f"{y - y_c:.5f}", f"{np.rad2deg(wrap(th)):.3f}"])


def _overlay_figure(results, y_c, half, title, out):
    """results: list of (label, traj, mt). Plot each model's path on the track."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhspan(y_c - half, y_c + half, color="0.92", label="track")
    ax.axhline(y_c, color="0.6", ls="--", lw=1)
    for label, traj, mt in results:
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=_sweep_color(label), label=label)
    ax.plot(results[0][1][0, 0], results[0][1][0, 1], "ks", ms=7)   # start marker
    ax.set_xlabel("x (down-track)"); ax.set_ylabel("y"); ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def run_sweep(a, y_c, models):
    """Sweep one axis (start heading / track width / w_lat), all models, one figure + CSV.
    Models are built ONCE and rolled at every value."""
    vals = [float(x) for x in a.sweep_values.split(",") if x.strip()]
    axis = {"theta": "theta0_deg", "width": "track_width", "wlat": "w_lat"}[a.sweep]
    built = {name: build_one(name, a) for name in models}
    report = ROOT / "results" / "track_mpc"; report.mkdir(parents=True, exist_ok=True)
    traj_dir = report / f"sweep_{a.sweep}_traj"; traj_dir.mkdir(exist_ok=True)   # per-rollout coords + overlays
    print(f"=== MPC sweep: {a.sweep} = {vals} ===")
    rows = []
    for v in vals:
        setattr(a, axis, v)
        half = a.track_width / 2.0
        per_val = []
        for name in models:
            traj, mt = roll_episode(built[name], a, y_c, half)
            per_val.append((mt["model"], traj, mt))
            rows.append((v, mt["model"], mt["final_lat"], mt["max_excursion"], mt["in_bounds"]))
            print(f"  {a.sweep}={v:6.2f}  {mt['model']:22s}  final_lat {mt['final_lat']:.3f}  "
                  f"max|y| {mt['max_excursion']:.3f}  {'IN' if mt['in_bounds'] else 'OUT'}")
            _save_coords(traj, y_c, traj_dir / f"coords_{v:g}_{_short_name(mt['model'])}.csv")
        _overlay_figure(per_val, y_c, half, f"MPC rollouts  ({a.sweep} = {v:g})", traj_dir / f"overlay_{v:g}.png")
    print(f"per-rollout coords + overlays -> {traj_dir}")

    import csv as _csv
    with open(report / f"sweep_{a.sweep}.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow([a.sweep, "model", "final_lat", "max_excursion", "in_bounds"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", int(r[4])])

    labels = sorted({r[1] for r in rows}, key=lambda l: ("oracle" not in l.lower(), l))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl in labels:
        fl = [r[2] for r in rows if r[1] == lbl]
        mx = [r[3] for r in rows if r[1] == lbl]
        ax1.plot(vals, fl, "-o", ms=4, color=_sweep_color(lbl), label=lbl)
        ax2.plot(vals, mx, "-o", ms=4, color=_sweep_color(lbl), label=lbl)
    xlabel = {"theta": "start heading (deg)", "width": "track width", "wlat": "w_lat"}[a.sweep]
    ax1.set_xlabel(xlabel); ax1.set_ylabel("final lateral error"); ax1.set_title("Convergence (final |y|)")
    ax2.set_xlabel(xlabel); ax2.set_ylabel("peak |y| excursion"); ax2.set_title("Peak excursion / constraint")
    if a.sweep == "width":     # boundary line: max|y| above it = left the track
        ax2.plot(vals, [v / 2 for v in vals], "k--", lw=1, label="track edge (W/2)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report / f"sweep_{a.sweep}.png", dpi=140)
    print(f"wrote {report / ('sweep_' + a.sweep + '.csv')}  and  sweep_{a.sweep}.png")


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
    p.add_argument("--actuator-gain", type=float, default=1.0,
                   help="unmodeled actuator efficiency at CONTROL time: truth moves at gain*v. " \
                   "Set to the dataset's gain (e.g. 0.7) so the test matches training; the naive oracle " \
                   "assumes 1 and mis-plans")
    p.add_argument("--track-width", type=float, default=0.4, help="track width (|y - y_c| <= W/2)")
    p.add_argument("--x0", type=float, default=0.15); p.add_argument("--y0", type=float, default=0.0,
                   help="initial lateral offset from centerline")
    p.add_argument("--theta0-deg", type=float, default=45.0)
    p.add_argument("--horizon", type=int, default=25); p.add_argument("--samples", type=int, default=800)
    p.add_argument("--sigma", type=float, default=0.8); p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=2.5); p.add_argument("--steps", type=int, default=18)
    p.add_argument("--w-lat", type=float, default=6.0); p.add_argument("--w-head", type=float, default=1.0)
    p.add_argument("--w-ctrl", type=float, default=0.05); p.add_argument("--w-bound", type=float, default=60.0)
    p.add_argument("--hard-bound", action="store_true",
                   help="hard track constraint: MPPI rejects any rolled trajectory that leaves the track (vs the soft w-bound penalty)")
    p.add_argument("--term-scale", type=float, default=4.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sweep", default=None, choices=["theta", "width", "wlat"],
                   help="sweep an axis instead of a single run (start heading / track width / w-lat)")
    p.add_argument("--sweep-values", default="", help="comma list for --sweep, e.g. 30,45,60,75,90")
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

    if a.sweep:
        run_sweep(a, y_c, models)
        return

    print(f"=== world-model MPC: straight track (start y0={a.y0:+.2f}, theta={a.theta0_deg:.0f} deg) ===")
    results = []
    for name in models:
        traj, mt = run_one(name, a, y_c, half)
        results.append((mt["model"], traj, mt))
        print(f"  {mt['model']:22s}  final_lat {mt['final_lat']:.3f}  "
              f"final_head {mt['final_head_deg']:5.1f} deg  max|y| {mt['max_excursion']:.3f}  "
              f"rms_lat {mt['rms_lat']:.3f}  {'IN' if mt['in_bounds'] else 'OUT of'} bounds")
        if name != "unicycle":                      # merge control result into the run's metrics.json
            ck = Path(a.ckpt_for(name))
            if len(ck.parts) >= 4 and ck.parent.parent.parent.name == "checkpoints":   # checkpoints/<run>/<model>/<tag>.pt
                from eval.metrics import update_mpc
                rep = C.RESULTS_DIR / ck.parent.parent.name / ck.parent.name / ck.stem
                update_mpc(rep, {k: mt[k] for k in ("final_lat", "final_head_deg", "max_excursion", "rms_lat", "in_bounds")})

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

    import csv as _csv                              # cross-model control summary (one row per model)
    with open(report / "mpc_summary.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["model", "final_lat", "final_head_deg", "max_excursion", "rms_lat", "in_bounds"])
        for label, _traj, mt in results:
            w.writerow([label, f"{mt['final_lat']:.4f}", f"{mt['final_head_deg']:.4f}",
                        f"{mt['max_excursion']:.4f}", f"{mt['rms_lat']:.4f}", int(mt["in_bounds"])])
    print(f"wrote {report / 'mpc_summary.csv'}")

    for label, traj, mt in results:      # per-rollout coords: t, x, y, heading, lateral/heading error
        _save_coords(traj, y_c, report / f"coords_theta{int(a.theta0_deg)}_{_short_name(label)}.csv")
    print(f"wrote per-rollout coords -> {report}")


if __name__ == "__main__":
    main()
