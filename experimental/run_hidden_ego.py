"""Train + evaluate the HIDDEN-STATE ego model (models/hidden_ego.py) on a varying-gain dataset.

    python run_hidden_ego.py --run run09_64x64_varactuator --grid-size 64 --pred-step 4 --epochs 40 --lam-anchor 1.0 --lam-anchor-pred 1.0
    # add --lam-gain 1.0 to SUPERVISE the gain (anchor a_v=g(h) to truth); omit it for the
    # unsupervised "discover the gain" run.

Reads a STACK of frames (the gain is invisible in one frame), grounds the 4 pose dims with the
anchor, lets a_v = exp(head(h)) be set by the inferred hidden state, and reports both pose recovery
AND how well the predicted gain g(h) matches the true per-episode actuator gain.
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

import torch
import torch.nn.functional as F
torch.set_num_threads(4)

import config as C
from models.hidden_ego import HiddenEgoWorldModel, hidden_ego_loss, hidden_ego_rollout_loss
from models.dataset import make_history_dataloaders, make_history_rollout_dataloaders


def state_to_target(states):
    x, y, th = states[:, 0], states[:, 1], states[:, 2]
    return torch.stack([x, y, torch.cos(th), torch.sin(th)], dim=1)


@torch.no_grad()
def extract(model, dl, device):
    """Encoded states, predicted gains, true (x,y,theta), true gains over a loader."""
    model.eval()
    S, G, T, GT = [], [], [], []
    for b in dl:
        s = model.encode(b["frame"].to(device), b["action_hist"].to(device))
        S.append(s.cpu()); G.append(model.gain(s).cpu())
        T.append(b["state"] if "state" in b else b["poses"][:, 0])   # rollout loader carries poses, not state
        if "gain" in b:
            GT.append(b["gain"])
    S, G, T = torch.cat(S), torch.cat(G), torch.cat(T)
    GT = torch.cat(GT) if GT else None
    return S, G, T, GT


def pearson(a, b):
    a, b = a.flatten(), b.flatten()
    return ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8)


@torch.no_grad()
def val_total(model, dl, device, a):
    model.eval(); tot = []
    for i, b in enumerate(dl):
        if i >= 50:
            break
        frame, nxt = b["frame"].to(device), b["next_frame"].to(device)
        out = model(frame, b["action_hist"].to(device), b["action"].to(device),
                    nxt, b["next_action_hist"].to(device))
        _, parts = hidden_ego_loss(out, frame, nxt,
                                   state_to_target(b["state"]).to(device),
                                   state_to_target(b["next_state"]).to(device),
                                   b["gain"].to(device) if "gain" in b else None,
                                   a.lam_recon, a.lam_dyn, a.lam_pred, a.recon_fg_weight,
                                   a.lam_anchor, a.lam_anchor_pred, a.lam_gain)
        tot.append(parts["total"])
    model.train()
    return sum(tot) / max(len(tot), 1)


@torch.no_grad()
def val_total_rollout(model, dl, device, a):
    model.eval(); tot = []
    for i, b in enumerate(dl):
        if i >= 50:
            break
        _, parts = hidden_ego_rollout_loss(model, b["frame"].to(device), b["action_hist"].to(device),
                                           b["roll_actions"].to(device), b["poses"].to(device),
                                           b["gain"].to(device) if "gain" in b else None,
                                           a.lam_recon, a.recon_fg_weight, a.lam_anchor, a.lam_rollout, a.lam_gain)
        tot.append(parts["total"])
    model.train()
    return sum(tot) / max(len(tot), 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate the hidden-state ego model.")
    p.add_argument("--run", default=C.RUN)
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    p.add_argument("--stack", type=int, default=4, help="frames per history stack (at stride pred_step)")
    p.add_argument("--hidden-dim", type=int, default=1, help="hidden latent dims for the inferred gain")
    p.add_argument("--lam-recon", type=float, default=1.0)
    p.add_argument("--lam-dyn", type=float, default=1.0)
    p.add_argument("--lam-pred", type=float, default=1.0)
    p.add_argument("--recon-fg-weight", type=float, default=5.0)
    p.add_argument("--lam-anchor", type=float, default=1.0, help="anchor the 4 pose dims to true pose")
    p.add_argument("--lam-anchor-pred", type=float, default=1.0, help="anchor the PREDICTED next pose")
    p.add_argument("--rollout-k", type=int, default=1,
                   help=">1 switches to K-step OPEN-LOOP rollout training (anchor the accumulated pose); "
                        "1 = the default single-step training")
    p.add_argument("--lam-rollout", type=float, default=1.0, help="weight on the K-step rollout pose anchor")
    p.add_argument("--lam-gain", type=float, default=0.0,
                   help="SUPERVISE a_v=g(h) against the true per-episode gain. 0 = unsupervised (probe h afterward)")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--note", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    tag = f"hidego_s{a.pred_step}_e{a.epochs}_st{a.stack}_h{a.hidden_dim}"
    tag += "_gsup" if a.lam_gain > 0 else "_gunsup"
    if a.rollout_k > 1:
        tag += f"_roll{a.rollout_k}"
    experiment = f"{a.run}_{tag}" + (f"_{a.note}" if a.note else "")
    data_path  = C.DATASETS_DIR / f"{a.run}.h5"
    ckpt_path  = C.CHECKPOINTS_DIR / f"{experiment}.pt"
    report_dir = C.RESULTS_DIR / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path}")

    print(f"=== {experiment} ===")
    print(f"device={a.device}  data={data_path.name}  stack={a.stack}  hidden_dim={a.hidden_dim}  "
          f"gain={'SUPERVISED' if a.lam_gain > 0 else 'unsupervised'}")

    torch.manual_seed(C.SEED)
    rollout = a.rollout_k > 1
    if rollout:
        train_dl, val_dl = make_history_rollout_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                                            stack=a.stack, K=a.rollout_k, step=a.pred_step)
    else:
        train_dl, val_dl = make_history_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                                    stack=a.stack, step=a.pred_step)
    print(f"{len(train_dl.dataset)} train windows, {len(val_dl.dataset)} val"
          + (f"  (K={a.rollout_k}-step rollout training)" if rollout else ""))
    model = HiddenEgoWorldModel(grid_size=a.grid_size, dt=a.pred_step * C.DT,
                                stack=a.stack, hidden_dim=a.hidden_dim).to(a.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best, step = float("inf"), 0
    train_hist, val_hist = [], []
    for epoch in range(a.epochs):
        for b in train_dl:
            if rollout:
                loss, parts = hidden_ego_rollout_loss(model, b["frame"].to(a.device), b["action_hist"].to(a.device),
                                                      b["roll_actions"].to(a.device), b["poses"].to(a.device),
                                                      b["gain"].to(a.device) if "gain" in b else None,
                                                      a.lam_recon, a.recon_fg_weight, a.lam_anchor,
                                                      a.lam_rollout, a.lam_gain)
            else:
                frame, nxt = b["frame"].to(a.device), b["next_frame"].to(a.device)
                out = model(frame, b["action_hist"].to(a.device), b["action"].to(a.device),
                            nxt, b["next_action_hist"].to(a.device))
                loss, parts = hidden_ego_loss(out, frame, nxt,
                                              state_to_target(b["state"]).to(a.device),
                                              state_to_target(b["next_state"]).to(a.device),
                                              b["gain"].to(a.device) if "gain" in b else None,
                                              a.lam_recon, a.lam_dyn, a.lam_pred, a.recon_fg_weight,
                                              a.lam_anchor, a.lam_anchor_pred, a.lam_gain)
                parts["a_v"] = out["gain"].mean().item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); step += 1
            if step % 50 == 0:
                train_hist.append({"step": step, "epoch": epoch + 1, **parts})
                if rollout:
                    print(f"  step {step:5d}  total {parts['total']:.4f}  recon {parts['recon']:.4f}  "
                          f"anchor {parts['anchor']:.4f}  rollout {parts['rollout']:.4f}  "
                          f"gain {parts['gain']:.4f}  a_v {parts['a_v']:.3f}")
                else:
                    print(f"  step {step:5d}  total {parts['total']:.4f}  recon {parts['recon']:.4f}  "
                          f"dyn {parts['dyn']:.4f}  anchor {parts['anchor']:.4f}  "
                          f"anchor_pred {parts['anchor_pred']:.4f}  gain {parts['gain']:.4f}  a_v {parts['a_v']:.3f}")
        v = val_total_rollout(model, val_dl, a.device, a) if rollout else val_total(model, val_dl, a.device, a)
        val_hist.append({"epoch": epoch + 1, "step": step, "val_total": v})
        print(f"epoch {epoch+1}/{a.epochs}  val total {v:.4f}")
        if v < best:
            best = v; torch.save(model.state_dict(), ckpt_path); print(f"  new best {best:.4f} -> {ckpt_path}")

    def _write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
    train_cols = (["step", "epoch", "total", "recon", "anchor", "rollout", "gain", "a_v"] if rollout
                  else ["step", "epoch", "total", "recon", "dyn", "pred_recon", "anchor", "anchor_pred", "gain", "a_v"])
    _write_csv(report_dir / "train_history.csv", train_hist, train_cols)
    _write_csv(report_dir / "val_history.csv", val_hist, ["epoch", "step", "val_total"])
    print(f"wrote train_history.csv ({len(train_hist)} rows) + val_history.csv")

    model.load_state_dict(torch.load(ckpt_path, map_location=a.device, weights_only=True))

    # ---- evaluation (pose + hidden gain readout, on the same val windows) ----
    S, G, T, GT = extract(model, val_dl, a.device)
    Ttarg = state_to_target(T)
    deg = 180.0 / torch.pi
    th_pred = torch.atan2(S[:, 3], S[:, 2]); d = th_pred - T[:, 2]
    ang = torch.abs(torch.atan2(torch.sin(d), torch.cos(d)))
    pose = {"theta_flip_pct": (ang > torch.pi/2).float().mean().item()*100,
            "theta_mae_deg":  (ang.mean()*deg).item(),
            "x_rmse": (S[:,0]-T[:,0]).pow(2).mean().sqrt().item(),
            "y_rmse": (S[:,1]-T[:,1]).pow(2).mean().sqrt().item()}
    print("\n-- pose read DIRECTLY from the latent --")
    print(f"  theta_flip {pose['theta_flip_pct']:.2f}  theta_mae {pose['theta_mae_deg']:.2f}  "
          f"x_rmse {pose['x_rmse']:.4f}  y_rmse {pose['y_rmse']:.4f}")
    with open(report_dir / "state_direct.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(pose.keys())); w.writerow([f"{v:.4f}" for v in pose.values()])

    print("\n-- HIDDEN GAIN: predicted a_v=g(h) --")
    print(f"  pred[min {G.min().item():.3f}  mean {G.mean().item():.3f}  max {G.max().item():.3f}]")
    if GT is not None:
        gain_mae  = (G - GT).abs().mean().item()
        gain_corr = pearson(G, GT).item()
        print(f"  vs TRUE per-episode gain: corr {gain_corr:+.3f}  mae {gain_mae:.4f}  "
              f"true[min {GT.min().item():.3f}  max {GT.max().item():.3f}]")
        with open(report_dir / "gain_recovery.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["corr", "mae", "pred_mean", "true_min", "true_max"])
            w.writerow([f"{gain_corr:.4f}", f"{gain_mae:.4f}", f"{G.mean().item():.4f}",
                        f"{GT.min().item():.4f}", f"{GT.max().item():.4f}"])
    else:
        print("  (no per-step 'gains' array in this dataset, e.g. run08 fixed gain; with a constant"
              " gain g(h) should sit ~flat near the dataset's actuator_gain)")

    print("\n-- per-dim correlation of the 4 pose dims with pose --")
    for j, nm in enumerate(["x", "y", "cos_th", "sin_th"]):
        corr = torch.stack([pearson(S[:, d], Ttarg[:, j]) for d in range(4)])
        print(f"  {nm:7s}  " + "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(4)))

    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
