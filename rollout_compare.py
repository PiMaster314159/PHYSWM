"""Open-loop ROLLOUT comparison across the world models (ego, residual JEPA, grounded).

Pose recovery only tests perception (encode one frame). This tests PREDICTION: encode the
first frame of a test episode ONCE, then roll each model's own predictor forward over the
action sequence in its latent space (no re-encoding), and decode every rolled latent to pose
with a fixed linear probe. We then plot pose error vs horizon. A model whose dynamics is
wrong (e.g. a frozen a_v on actuator data, or grounded's locked kinematics) drifts as the
horizon grows; a correct gray-box tracks. This is the real world-model test for control.

Run on the tower (needs torch + the dataset + the four s4 checkpoints):
    python rollout_compare.py --run run08_64x64_actuator --pred-step 4 --epochs 40
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import csv
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from models.jepa import JEPA
from models.grounded import GroundedJEPA
from models.state_ae import EgoWorldModel
from eval.probe import make_linear_probe, train_probe, state_to_target

GRID = 64
CK   = C.CHECKPOINTS_DIR


def build_models(run, step, epochs):
    """(label, model, checkpoint path). Tags mirror the run_*.py naming for these runs."""
    dt = step * C.DT
    jepa_res_anc = JEPA(grid_size=GRID, latent_dim=128, predictor_mode="residual", state_head=False)
    jepa_res_rd  = JEPA(grid_size=GRID, latent_dim=128, predictor_mode="residual", state_head=True)
    jepa_res_anc.predictor.dt = dt; jepa_res_rd.predictor.dt = dt
    return [
        ("ego (learnable a_v)",
         EgoWorldModel(grid_size=GRID, dt=dt, residual_budget=0.0, learn_coeffs=True, decoder="mlp"),
         CK / f"{run}_ego_s{step}_e{epochs}_fg5_anc1_ancp1_learn_mlpdec.pt"),
        ("residual JEPA + anchor", jepa_res_anc,
         CK / f"{run}_residual_s{step}_e{epochs}_anc1.pt"),
        ("residual JEPA + readout", jepa_res_rd,
         CK / f"{run}_residual_s{step}_e{epochs}_rd1.pt"),
        ("grounded + block anchor",
         GroundedJEPA(grid_size=GRID, latent_dim=128, block_dim=4, dt=dt,
                      lock_block=True, block_budget=0.0, use_decoder=False),
         CK / f"{run}_grounded_s{step}_e{epochs}_anc1.pt"),
    ]


def roll_fn(model):
    """Each model's one-transition predictor in its own latent space."""
    return model.step if isinstance(model, EgoWorldModel) else model.predict


def load_val_episodes(data_path, seed, n_eps, val_frac=0.1):
    """Episode-contiguous (frames, actions, states) for the held-out val episodes."""
    with h5py.File(data_path, "r") as f:
        frames = f["frames"][:]; actions = f["actions"][:]; states = f["states"][:]
        starts = f["episode_starts"][:]; lengths = f["episode_lengths"][:]
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(starts))
    val  = perm[:int(round(val_frac * len(starts)))]
    eps = []
    for e in val[:n_eps]:
        s, L = int(starts[e]), int(lengths[e])
        eps.append((frames[s:s + L], actions[s:s + L], states[s:s + L]))
    return eps


def fit_probe(model, episodes, device, max_frames=4000):
    """Fit a fixed linear probe z -> pose on the ENCODED frames (the decoder for rollout).

    Only the encoding is under no_grad; the probe itself must train with grad enabled.
    """
    F = np.concatenate([ep[0] for ep in episodes])
    S = np.concatenate([ep[2] for ep in episodes])
    if len(F) > max_frames:
        sel = np.random.default_rng(0).choice(len(F), max_frames, replace=False)
        F, S = F[sel], S[sel]
    Ft = torch.from_numpy(F).float().unsqueeze(1).to(device)
    with torch.no_grad():
        Z = torch.cat([model.encode(Ft[i:i + 512]).cpu() for i in range(0, len(Ft), 512)])
    T = state_to_target(torch.from_numpy(S).float())
    probe = train_probe(make_linear_probe(Z.shape[1]), Z, T, epochs=120, device=device)
    return probe.to(device).eval()


@torch.no_grad()
def rollout(model, probe, episodes, pred_step, H, device):
    """Mean pose error vs horizon h (in transitions). Encode frame 0, roll the predictor h
    times, decode with the probe, compare to true pose at sim-step h*pred_step."""
    roll = roll_fn(model)
    pos = {h: [] for h in range(H + 1)}
    head = {h: [] for h in range(H + 1)}
    for frames, actions, states in episodes:
        L = len(frames)
        Hmax = min(H, (L - 1) // pred_step)
        z = model.encode(torch.from_numpy(frames[0]).float().view(1, 1, GRID, GRID).to(device))
        for h in range(Hmax + 1):
            idx = h * pred_step
            p = probe(z)[0].cpu().numpy()                     # decoded (x, y, cos, sin)
            tp = states[idx]                                  # true (x, y, theta)
            pos[h].append(float(np.hypot(p[0] - tp[0], p[1] - tp[1])))
            d = np.arctan2(p[3], p[2]) - tp[2]
            head[h].append(float(abs(np.arctan2(np.sin(d), np.cos(d))) * 180.0 / np.pi))
            if h < Hmax:
                a = torch.from_numpy(actions[idx]).float().view(1, -1).to(device)
                z = roll(z, a)
    horizons = [h for h in range(H + 1) if pos[h]]
    return (horizons,
            [float(np.mean(pos[h])) for h in horizons],
            [float(np.mean(head[h])) for h in horizons])


def main():
    ap = argparse.ArgumentParser(description="Open-loop rollout drift comparison.")
    ap.add_argument("--run", default="run08_64x64_actuator")
    ap.add_argument("--pred-step", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=12, help="max rollout length in transitions")
    ap.add_argument("--n-episodes", type=int, default=80)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = C.DATASETS_DIR / f"{a.run}.h5"
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path}")
    episodes = load_val_episodes(data_path, C.SEED, a.n_episodes)
    print(f"rolling out {len(episodes)} val episodes from {data_path.name}  (pred_step={a.pred_step})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rows = []
    for label, model, ckpt in build_models(a.run, a.pred_step, a.epochs):
        if not ckpt.exists():
            print(f"!! missing checkpoint, skipping: {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()
        probe = fit_probe(model, episodes, device)
        hz, pe, he = rollout(model, probe, episodes, a.pred_step, a.horizon, device)
        steps = [h * a.pred_step for h in hz]                  # x-axis in sim-steps (comparable across pred_step)
        axes[0].plot(steps, pe, marker="o", ms=3, label=label)
        axes[1].plot(steps, he, marker="o", ms=3, label=label)
        for h, s, p, q in zip(hz, steps, pe, he):
            rows.append([label, h, s, f"{p:.4f}", f"{q:.4f}"])
        print(f"  {label:26s}  pos@end={pe[-1]:.3f}  theta@end={he[-1]:.1f}deg  (horizon {hz[-1]} transitions)")

    axes[0].set_xlabel("rollout horizon (sim-steps)"); axes[0].set_ylabel("position error"); axes[0].set_title("position drift")
    axes[1].set_xlabel("rollout horizon (sim-steps)"); axes[1].set_ylabel("heading error (deg)"); axes[1].set_title("heading drift")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"open-loop rollout drift  |  {a.run}  s{a.pred_step}", y=1.02)
    fig.tight_layout()
    out = C.RESULTS_DIR / "_compare"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"rollout_{a.run}_s{a.pred_step}.png", dpi=120, bbox_inches="tight")
    with open(out / f"rollout_{a.run}_s{a.pred_step}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "horizon_transitions", "horizon_simsteps", "pos_err", "theta_mae_deg"])
        w.writerows(rows)
    print(f"wrote {out / f'rollout_{a.run}_s{a.pred_step}.png'}")


if __name__ == "__main__":
    main()
