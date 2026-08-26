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
from collections import deque
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
from sim.dynamics import bicycle_step as true_bicycle_step   # ground-truth bicycle+throttle
from control.mppi import mppi_plan


ORACLES = ("unicycle", "bicycle")   # numpy known-dynamics baselines: no checkpoint, no torch


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
        vv = A[:, 0] if a.plan_speed else v        # 2-D action carries v; else fixed (naive: NO drag model)
        om = A[:, -1]
        th2 = th + om * dt
        return np.stack([x + vv * np.cos(th2) * dt, y + vv * np.sin(th2) * dt, th2], axis=1)
    def pose(S):
        return S[:, :3].copy()
    return observe, dynamics, pose, 3, "unicycle (oracle)", 1


def build_bicycle(a):
    """Naive bicycle oracle: exact kinematics, but it does NOT know about drag -- the same stance the
    unicycle oracle takes. That is deliberate. Drag is the unmodeled physics under test, so an oracle
    that knew it would beat every learned model for reasons that have nothing to do with the thesis.
    Its 4-D state (x, y, theta, v) carries velocity because the throttle integrates into it."""
    dt, L = a.dt, a.wheelbase

    def observe(true_state, frame):
        return np.asarray(true_state, float).copy()

    def dynamics(S, A):
        x, y, th, v = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
        acc, delta = A[:, 0], A[:, 1]
        v2 = np.clip(v + acc * dt, a.v_min, a.v_max)           # no drag term: the naive model
        th2 = wrap(th + (v2 / L) * np.tan(delta) * dt)         # steering authority scales with speed
        return np.stack([x + v2 * np.cos(th2) * dt, y + v2 * np.sin(th2) * dt, th2, v2], axis=1)

    def pose(S):
        return S[:, :3].copy()

    return observe, dynamics, pose, 4, "bicycle (oracle)", 1


def build_neural(name, a):
    """ego / grounded / jepa. Torch-only; runs on the tower."""
    import torch
    dev = a.device
    g = a.grid_size
    v = a.v

    sd = torch.load(a.ckpt_for(name), map_location=dev, weights_only=True)
    fc = sd.get("encoder.conv.0.weight")               # infer frame-stack depth from the checkpoint's first conv
    n_frames = int(fc.shape[1]) if fc is not None else 1   # (out, IN, k, k) -> IN = frames stacked as channels

    dyn = a.dynamics                                   # "unicycle" | "bicycle"; must match the checkpoint
    bike = dyn == "bicycle"
    if name == "ego":
        from models.state_ae import EgoWorldModel
        model = EgoWorldModel(grid_size=g, dt=a.dt, learn_coeffs=True, decoder="mlp", residual_mode=a.residual,
                              in_channels=n_frames, dynamics=dyn, wheelbase=a.wheelbase)
        # ego's latent IS the state, so the planner's state width follows the dynamics: 5-D for the
        # bicycle ([x,y,cos,sin,v] -- velocity is part of the state the throttle integrates into).
        physical = True; latent_dim = 5 if bike else 4; label = "ego (gray-box)"
    elif name == "grounded":
        from models.grounded import GroundedJEPA
        model = GroundedJEPA(grid_size=g, latent_dim=C.LATENT_DIM, block_dim=5 if bike else 4, dt=a.dt,
                             learn_coeffs=True, residual_mode=a.residual, in_channels=n_frames,
                             dynamics=dyn, wheelbase=a.wheelbase)
        physical = True; latent_dim = C.LATENT_DIM; label = "grounded"
    elif name == "jepa":
        from models.jepa import JEPA
        model = JEPA(grid_size=g, latent_dim=C.LATENT_DIM, predictor_mode="residual", state_head=True,
                     in_channels=n_frames, dynamics=dyn)
        physical = False; latent_dim = C.LATENT_DIM; label = "residual JEPA (probe)"
    else:
        raise ValueError(f"unknown neural model {name!r}")

    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    if n_frames > 1:
        print(f"  [{label}] history checkpoint: stacking {n_frames} frames at control time")

    # decode_pose un-standardizes JEPA's readout with the model's pose_mean/std buffers. Archived
    # checkpoints predate those buffers (they load as the 0/1 default), so back-fill them from the
    # dataset once, keeping decode_pose self-contained (new checkpoints carry the stats themselves).
    if not physical and torch.allclose(model.pose_std, torch.ones_like(model.pose_std)):
        import h5py
        from models.components import pose_stats
        with h5py.File(C.DATASETS_DIR / f"{a.run}.h5", "r") as f:
            states = torch.from_numpy(f["states"][:]).float()
        model.set_pose_stats(*pose_stats(states))

    def to_frame(frames):                              # frames: (g,g) single or (n_frames,g,g) stack -> (1,n_frames,g,g)
        return torch.from_numpy(np.asarray(frames, np.float32)).view(1, n_frames, g, g).to(dev)

    def full_action(A):                               # -> (K,2) torch: [v,omega] unicycle | [accel,delta] bicycle
        if bike or a.plan_speed:                      # already 2-D and in the model's own action space
            return torch.from_numpy(A.astype(np.float32)).to(dev)
        K = A.shape[0]                                # A is (K,1)=omega; prepend the fixed speed
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

    return observe, dynamics, pose, latent_dim, label, n_frames


def make_costs(pose_fn, y_c, half, a):
    def bound_cost(lat):
        if getattr(a, "hard_bound", False):     # HARD: any predicted step outside -> ~inf -> MPPI rejects it
            return np.where(np.abs(lat) > half, 1e6, 0.0)
        viol = np.maximum(0.0, np.abs(lat) - half)      # SOFT: quadratic penalty past the edge
        return a.w_bound * viol**2
    def running(S, A, S_next=None):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2]); om = A[:, -1]
        cost = a.w_lat*lat**2 + a.w_head*head**2 + a.w_ctrl*om**2 + bound_cost(lat)
        # Bicycle always prices speed: the action IS throttle, so speed is under control by construction
        # (there is no "fixed v" mode to fall back on).
        if (a.plan_speed or a.dynamics == "bicycle") and S_next is not None:
            # track a reference GROUND speed. ds is the model's PREDICTED world displacement, so a
            # drag-aware model (ego gray-box) knows command v -> slower ground speed and picks the
            # right v; a naive model (no drag) commands v=v_ref and the true car then falls short.
            ds = np.sqrt(((S_next[:, :2] - S[:, :2]) ** 2).sum(1))
            cost = cost + a.w_speed * (ds / a.dt - a.v_ref) ** 2
        return cost
    def terminal(S):
        P = pose_fn(S); lat = P[:, 1] - y_c; head = wrap(P[:, 2])
        return a.term_scale*(a.w_lat*lat**2 + a.w_head*head**2) + bound_cost(lat)
    return running, terminal


def build_one(name, a):
    if name == "unicycle":
        built = build_unicycle(a)
    elif name == "bicycle":
        built = build_bicycle(a)
    else:
        built = build_neural(name, a)
    observe, dynamics, pose, d, label, n_frames = built
    return observe, dynamics, pose, label, n_frames


def roll_episode(built, a, y_c, half):
    observe, dynamics, pose, label, n_frames = built
    running, terminal = make_costs(pose, y_c, half, a)
    rng = np.random.default_rng(a.seed)
    bike = a.dynamics == "bicycle"
    if bike:                                           # 2-D action [accel, delta]: throttle + steering
        a_low = np.array([-a.a_max, -a.delta_max]); a_high = np.array([a.a_max, a.delta_max])
        sigma = np.array([a.sigma_a, a.sigma_delta]); m = 2
    elif a.plan_speed:                                 # 2-D action [v, omega]: MPPI also plans speed
        a_low = np.array([a.v_min, -a.omega_max]); a_high = np.array([a.v_max, a.omega_max])
        sigma = np.array([a.sigma_v, a.sigma]); m = 2
    else:                                              # 1-D action [omega]: speed fixed at a.v
        a_low = np.array([-a.omega_max]); a_high = np.array([a.omega_max])
        sigma = a.sigma; m = 1

    # TRUE state (world [0,1]); the bicycle carries velocity as a 4th dim (the throttle integrates into it)
    true = np.array([a.x0, y_c + a.y0, np.deg2rad(a.theta0_deg)] + ([a.v0] if bike else []), float)
    a_nom = np.zeros((a.horizon, m))
    if a.plan_speed and not bike:
        a_nom[:, 0] = a.v_ref                          # warm-start the speed channel at the reference (not 0)
    traj = [true.copy()]
    pred_peak = 0.0                                     # largest |lat| the MODEL ever predicted for a committed plan
    frame_hist = deque(maxlen=n_frames)                # last n_frames rendered frames (for history checkpoints)
    for _ in range(a.steps):
        frame = render_frame(true[:3], grid_size=a.grid_size, marker=a.marker)   # camera sees POSE only -- the
                                                                                 # bicycle 4th dim (v) is hidden

        if not frame_hist:                             # cold start: pad the buffer with the first frame
            frame_hist.extend([frame] * n_frames)
        else:
            frame_hist.append(frame)
        s0 = observe(true, np.stack(frame_hist))       # model's estimate of the current state (oldest..newest)
        a_nom, _ = mppi_plan(s0, a_nom, dynamics, running, terminal,
                             n_samples=a.samples, sigma=sigma, lam=a.lam,
                             a_low=a_low, a_high=a_high, rng=rng)
        if a.log_pred_excursion:                       # the model's estimate of its OWN current |lat| (localization,
            model_lat = abs(float(pose(np.asarray(s0, float)[None])[0, 1]) - y_c)   # what the constraint rollout starts from)
            pred_peak = max(pred_peak, model_lat)
        if bike:
            # truth applies the SAME drag the dataset was collected with; the models must have learned it
            true = true_bicycle_step(true, np.array([float(a_nom[0, 0]), float(a_nom[0, 1])]), a.dt,
                                     wheelbase=a.wheelbase, drag_c=a.drag_c, gain=a.actuator_gain,
                                     v_min=a.v_min, v_max=a.v_max)
        else:
            v_cmd = float(a_nom[0, 0]) if a.plan_speed else a.v
            omega = float(a_nom[0, -1])
            v_true = max(0.0, a.actuator_gain * v_cmd - a.drag_c * v_cmd ** 2)    # non-ideal speed at CONTROL time
            true = true_step(true, np.array([v_true, omega]), a.dt)               # actuator gain + v^2 drag (truth only)
        traj.append(true.copy())
        a_nom = np.roll(a_nom, -1, axis=0); a_nom[-1] = 0.0
        if a.plan_speed and not bike:
            a_nom[-1, 0] = a.v_ref                     # keep cruising; don't let planned speed decay to 0

    traj = np.array(traj)
    y_err = traj[:, 1] - y_c
    metrics = {"model": label,
               "final_lat": abs(y_err[-1]),
               "final_head_deg": abs(np.rad2deg(wrap(traj[-1, 2]))),
               "max_excursion": np.abs(y_err).max(),
               "in_bounds": bool(np.abs(y_err).max() <= half + 1e-6),
               "rms_lat": float(np.sqrt((y_err**2).mean()))}
    if a.plan_speed or bike:                           # achieved GROUND speed vs reference (the drag-relevant metric)
        step_d = np.sqrt((np.diff(traj[:, :2], axis=0) ** 2).sum(1))
        mean_speed = float(step_d.mean() / a.dt)
        metrics["mean_speed"] = mean_speed
        metrics["speed_err"] = float(abs(mean_speed - a.v_ref))
    if a.log_pred_excursion:                           # peak |lat| the MODEL believed (from its own estimates) vs the truth
        metrics["pred_max_excursion"] = float(pred_peak)
        metrics["excursion_gap"] = float(metrics["max_excursion"] - pred_peak)   # +ve = under-localized (thought it was safer)
    return traj, metrics


def run_one(name, a, y_c, half):
    return roll_episode(build_one(name, a), a, y_c, half)


def _sweep_color(label):
    l = label.lower()
    return ("#898781" if "oracle" in l else "#2a78d6" if "ego" in l
            else "#1baf7a" if "grounded" in l else "#eb6834")


def _short_name(label):
    l = label.lower()
    return ("bicycle" if "bicycle" in l else
            "unicycle" if "oracle" in l or "unicycle" in l else
            "jepa" if "jepa" in l else "grounded" if "grounded" in l else "ego")


def _save_coords(traj, y_c, path):
    """Per-step rollout coords: t, x, y, heading, lateral error, heading error."""
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["t", "x", "y", "theta_rad", "lateral_err", "heading_deg", "v"])
        for k, row in enumerate(traj):                 # bicycle rows carry a 4th dim (velocity); unicycle do not
            x, y, th = row[0], row[1], row[2]
            v = f"{row[3]:.5f}" if len(row) > 3 else ""
            w.writerow([k, f"{x:.5f}", f"{y:.5f}", f"{th:.5f}", f"{y - y_c:.5f}",
                        f"{np.rad2deg(wrap(th)):.3f}", v])


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
    if a.sweep == "vref" and not a.plan_speed:
        raise SystemExit("--sweep vref requires --plan-speed (v_ref only affects the speed-planning objective)")
    vals = [float(x) for x in a.sweep_values.split(",") if x.strip()]
    axis = {"theta": "theta0_deg", "width": "track_width", "wlat": "w_lat", "vref": "v_ref", "y0": "y0"}[a.sweep]
    built = {name: build_one(name, a) for name in models}
    report = ROOT / "results" / "track_mpc"
    if a.name:
        report = report / a.name
    report.mkdir(parents=True, exist_ok=True)
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
            se = mt.get("speed_err", float("nan"))
            rows.append((v, mt["model"], mt["final_lat"], mt["max_excursion"], mt["in_bounds"], se,
                         mt.get("pred_max_excursion", float("nan")), mt.get("excursion_gap", float("nan"))))
            extra = f"  speed_err {se:.3f}" if a.plan_speed else ""
            if a.log_pred_excursion:
                extra += f"  pred|y| {mt['pred_max_excursion']:.3f}  gap {mt['excursion_gap']:+.3f}"
            print(f"  {a.sweep}={v:6.3f}  {mt['model']:22s}  final_lat {mt['final_lat']:.3f}  "
                  f"max|y| {mt['max_excursion']:.3f}  {'IN' if mt['in_bounds'] else 'OUT'}{extra}")
            _save_coords(traj, y_c, traj_dir / f"coords_{v:g}_{_short_name(mt['model'])}.csv")
        _overlay_figure(per_val, y_c, half, f"MPC rollouts  ({a.sweep} = {v:g})", traj_dir / f"overlay_{v:g}.png")
    print(f"per-rollout coords + overlays -> {traj_dir}")

    import csv as _csv
    with open(report / f"sweep_{a.sweep}.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow([a.sweep, "model", "final_lat", "max_excursion", "in_bounds", "speed_err",
                    "pred_max_excursion", "excursion_gap"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", int(r[4]), f"{r[5]:.4f}",
                        f"{r[6]:.4f}", f"{r[7]:.4f}"])

    labels = sorted({r[1] for r in rows}, key=lambda l: ("oracle" not in l.lower(), l))
    show_speed = a.sweep == "vref"     # only the v_ref sweep wants the speed-error panel; others want excursion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl in labels:
        fl = [r[2] for r in rows if r[1] == lbl]
        p2 = [(r[5] if show_speed else r[3]) for r in rows if r[1] == lbl]
        ax1.plot(vals, fl, "-o", ms=4, color=_sweep_color(lbl), label=lbl)
        ax2.plot(vals, p2, "-o", ms=4, color=_sweep_color(lbl), label=lbl)
    xlabel = {"theta": "start heading (deg)", "width": "track width", "wlat": "w_lat",
              "vref": "target ground speed  v_ref", "y0": "start offset  y0"}[a.sweep]
    ax1.set_xlabel(xlabel); ax1.set_ylabel("final lateral error"); ax1.set_title("Convergence (final |y|)")
    ax2.set_xlabel(xlabel)
    if show_speed:
        ax2.set_ylabel("speed error  |mean_v − v_ref|"); ax2.set_title("Speed tracking")
    else:
        ax2.set_ylabel("peak |y| excursion"); ax2.set_title("Peak excursion / constraint")
    if a.sweep == "width" and not show_speed:   # boundary line: max|y| above it = left the track
        ax2.plot(vals, [v / 2 for v in vals], "k--", lw=1, label="track edge (W/2)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report / f"sweep_{a.sweep}.png", dpi=140)
    print(f"wrote {report / ('sweep_' + a.sweep + '.csv')}  and  sweep_{a.sweep}.png")


def parse_args():
    p = argparse.ArgumentParser(description="World-model MPC comparison on the straight track.")
    p.add_argument("--model", default="unicycle", help="comma list: unicycle,bicycle,ego,grounded,jepa "
                                                       "('bicycle' is the naive kinematic oracle, no drag)")
    p.add_argument("--dynamics", default="unicycle", choices=["unicycle", "bicycle"],
                   help="TRUE system + model action space. bicycle: 4-D truth (x,y,theta,v), action "
                        "(accel, delta), velocity is a hidden state -> use history checkpoints. MUST match "
                        "how the checkpoints were trained")
    p.add_argument("--a-max", type=float, default=1.5, help="throttle limit |accel| (--dynamics bicycle)")
    p.add_argument("--delta-max", type=float, default=0.6, help="steering-angle limit rad (--dynamics bicycle)")
    p.add_argument("--wheelbase", type=float, default=C.WHEELBASE, help="bicycle wheelbase L")
    p.add_argument("--sigma-a", type=float, default=0.6, help="MPPI exploration std on throttle")
    p.add_argument("--sigma-delta", type=float, default=0.3, help="MPPI exploration std on steering")
    p.add_argument("--v0", type=float, default=0.2, help="initial true speed (--dynamics bicycle)")
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
    p.add_argument("--drag-c", type=float, default=0.0,
                   help="unmodeled v^2 drag at CONTROL time: truth moves at gain*v - drag_c*v^2. " \
                   "Set to the dataset's drag_c so the test matches training")
    p.add_argument("--plan-speed", action="store_true",
                   help="2-D action: MPPI plans forward speed v too (not just omega), tracking a reference "
                        "GROUND speed --v-ref. This is what makes v^2 drag control-relevant: a drag-aware model "
                        "commands the right v, a naive one commands v=v_ref and the true car falls short. At fixed "
                        "v (default) drag is just a constant offset and re-planning absorbs it.")
    p.add_argument("--v-ref", type=float, default=0.20,
                   help="target ground speed for --plan-speed (must be below the drag curve's peak "
                        "gain^2/(4*drag_c); e.g. 0.25 at gain 1, drag_c 1)")
    p.add_argument("--v-min", type=float, default=0.05, help="min planned speed (--plan-speed)")
    p.add_argument("--v-max", type=float, default=0.6, help="max planned speed (--plan-speed)")
    p.add_argument("--sigma-v", type=float, default=0.15, help="MPPI exploration std on the speed action")
    p.add_argument("--w-speed", type=float, default=8.0, help="weight on ground-speed tracking (--plan-speed)")
    p.add_argument("--residual", default="none", choices=["none", "basis", "mlp"],
                   help="gray-box residual mode for ego/grounded; MUST match how the checkpoint was trained")
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
    p.add_argument("--log-pred-excursion", action="store_true",
                   help="per rollout, log the model's PREDICTED peak |y| vs the TRUE peak (localization gap: +ve = model "
                        "thought it was safer than it was). Diagnoses hard-constraint failures.")
    p.add_argument("--term-scale", type=float, default=4.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sweep", default=None, choices=["theta", "width", "wlat", "vref", "y0"],
                   help="sweep an axis instead of a single run (start heading / track width / w-lat / "
                        "target ground speed v_ref [needs --plan-speed] / start lateral offset y0)")
    p.add_argument("--sweep-values", default="", help="comma list for --sweep, e.g. 30,45,60,75,90")
    p.add_argument("--name", default=None,
                   help="label for this run: outputs go to results/track_mpc/<name>/ instead of overwriting the shared files")
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
    if any(m not in ORACLES for m in models):         # torch only needed for neural models
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
        speed_str = (f"  mean_v {mt['mean_speed']:.3f} (ref {a.v_ref:.2f}, err {mt['speed_err']:.3f})"
                     if "mean_speed" in mt else "")
        print(f"  {mt['model']:22s}  final_lat {mt['final_lat']:.3f}  "
              f"final_head {mt['final_head_deg']:5.1f} deg  max|y| {mt['max_excursion']:.3f}  "
              f"rms_lat {mt['rms_lat']:.3f}  {'IN' if mt['in_bounds'] else 'OUT of'} bounds{speed_str}")
        if name not in ORACLES:                     # merge control result into the run's metrics.json
            ck = Path(a.ckpt_for(name))
            if len(ck.parts) >= 4 and ck.parent.parent.parent.name == "checkpoints":   # checkpoints/<run>/<model>/<tag>.pt
                from eval.metrics import update_mpc
                rep = C.RESULTS_DIR / ck.parent.parent.name / ck.parent.name / ck.stem
                keys = ["final_lat", "final_head_deg", "max_excursion", "rms_lat", "in_bounds"]
                if "mean_speed" in mt: keys += ["mean_speed", "speed_err"]
                update_mpc(rep, {k: mt[k] for k in keys})

    # overlay figure -> results/track_mpc/<name>/ when --name is given (else the shared, overwritten files)
    report = ROOT / "results" / "track_mpc"
    if a.name:
        report = report / a.name
    report.mkdir(parents=True, exist_ok=True)
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
    speed_cols = ["mean_speed", "speed_err"] if a.plan_speed else []
    with open(report / "mpc_summary.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model", "final_lat", "final_head_deg", "max_excursion", "rms_lat", "in_bounds"] + speed_cols)
        for label, _traj, mt in results:
            row = [label, f"{mt['final_lat']:.4f}", f"{mt['final_head_deg']:.4f}",
                   f"{mt['max_excursion']:.4f}", f"{mt['rms_lat']:.4f}", int(mt["in_bounds"])]
            w.writerow(row + [f"{mt[c]:.4f}" for c in speed_cols])
    print(f"wrote {report / 'mpc_summary.csv'}")

    for label, traj, mt in results:      # per-rollout coords: t, x, y, heading, lateral/heading error
        _save_coords(traj, y_c, report / f"coords_theta{int(a.theta0_deg)}_{_short_name(label)}.csv")
    print(f"wrote per-rollout coords -> {report}")


if __name__ == "__main__":
    main()
