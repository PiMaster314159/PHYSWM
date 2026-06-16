"""Train + evaluate the ego world model (models/state_ae.py) on an existing dataset.

    python run_ego.py --run spin_64x64_bignose --grid-size 64 --pred-step 1 --epochs 10
    python run_ego.py --run run04_64x64_nose   --grid-size 64 --pred-step 4 --epochs 20

Self-contained: no dependency on the old JEPA/probe code. Evaluation reads the 4-dim
state out of the encoder and (a) probes pose from it (linear + MLP, gauge-invariant)
and (b) reports the per-dim correlation, which should show d0~x, d1~y, d2~cos, d3~sin
if the state grounded. Also prints the learned gray-box coefficients a_v, a_omega.
"""

import os 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import csv
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

import config as C
from models.state_ae import EgoWorldModel, ego_loss, STATE_DIM
from models.dataset import make_dataloaders


# ---- small, self-contained evaluation helpers ----

def state_to_target(states: torch.Tensor) -> torch.Tensor: 
    """(N,3) (x,y,theta) -> (N,4) (x,y,cos,sin)."""
    x, y, th = states[:, 0], states[:, 1], states[:, 2]
    return torch.stack([x, y, torch.cos(th), torch.sin(th)], dim=1)


@torch.no_grad()
def extract_states(model, dl, device):
    """Encoded 4-dim states and true (x,y,theta) over a loader."""
    model.eval()
    enc, true = [], []
    for b in dl:
        enc.append(model.encode(b["frame"].to(device)).cpu())
        true.append(b["state"])
    return torch.cat(enc), torch.cat(true)


def pearson_per_dim(Z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(0, keepdim=True)
    tc = t - t.mean()
    return (Zc * tc.unsqueeze(1)).mean(0) / (Z.std(0) * t.std() + 1e-8)


def probe_pose(Z_tr, T_tr, Z_va, S_va, hidden, device, epochs=120, bs=4096):
    """Fit a probe Z->(x,y,cos,sin); report heading flip %, mae, x/y R^2.

    Minibatched so the probe actually converges. (Full-batch GD for a few dozen steps
    underfits badly, giving bogus negative R^2 that understates the grounding.)
    """
    probe = (nn.Linear(Z_tr.shape[1], 4) if hidden == 0 else
             nn.Sequential(nn.Linear(Z_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 4)))
    probe.to(device); Z_tr, T_tr = Z_tr.to(device), T_tr.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n = Z_tr.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad(); F.mse_loss(probe(Z_tr[b]), T_tr[b]).backward(); opt.step()
    with torch.no_grad():
        p = probe(Z_va.to(device)).cpu()
    th_pred = torch.atan2(p[:, 3], p[:, 2])
    d = th_pred - S_va[:, 2]
    ang = torch.abs(torch.atan2(torch.sin(d), torch.cos(d)))
    deg = 180.0 / torch.pi

    def r2(pr, tt):
        return (1 - (pr - tt).pow(2).sum() / (tt - tt.mean()).pow(2).sum()).item()
    return {
        "theta_flip_pct": (ang > torch.pi / 2).float().mean().item() * 100.0,
        "theta_mae_deg":  (ang.mean() * deg).item(),
        "x_r2": r2(p[:, 0], S_va[:, 0]), "y_r2": r2(p[:, 1], S_va[:, 1]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate the ego world model.")
    p.add_argument("--run", default=C.RUN, help="dataset name (.h5 stem); must already exist")
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    p.add_argument("--lam-dyn",  type=float, default=1.0, help="state-space dynamics-consistency weight")
    p.add_argument("--lam-pred", type=float, default=1.0, help="predicted-state -> next-frame reconstruction weight")
    p.add_argument("--residual-budget", type=float, default=0.0, help="gray-box: bounded learned dynamics correction (0 = pure kinematics)")
    p.add_argument("--learn-coeffs", action="store_true", help="learn a_v/a_omega (gray-box); default frozen at the known values (=1) so they can't collapse")
    p.add_argument("--lam-var",   type=float, default=1.0, help="variance-floor weight (anti-collapse: forbids dead state dims). 0 = off")
    p.add_argument("--var-gamma", type=float, default=0.1, help="per-dim std floor for the variance term")
    p.add_argument("--decoder", default="mlp", choices=["broadcast", "mlp"], help="renderer: MLP (fast) or spatial-broadcast (slower, better at placing objects)")
    p.add_argument("--recon-fg-weight", type=float, default=5.0, help="foreground weighting of recon (bg_mean + w*fg_mean); stops the ~99%% black background from drowning the object. 0 = plain MSE")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--note", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def val_total(model, dl, device, a, max_batches=50):
    model.eval()
    tot = []
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        out = model(b["frame"].to(device), b["action"].to(device), b["next_frame"].to(device))
        _, parts = ego_loss(out, b["frame"].to(device), b["next_frame"].to(device),
                            a.lam_dyn, a.lam_pred, a.lam_var, a.var_gamma, a.recon_fg_weight)
        tot.append(parts["total"])
    model.train()
    return sum(tot) / max(len(tot), 1)


def main():
    a = parse_args()
    tag = f"ego_s{a.pred_step}_e{a.epochs}"
    if a.lam_dyn != 1.0:       tag += f"_ld{a.lam_dyn:g}"      # so different weights -> different folders
    if a.lam_pred != 1.0:      tag += f"_lp{a.lam_pred:g}"
    if a.lam_var > 0:          tag += f"_v{a.lam_var:g}"     # mark the variance-floor runs
    if a.recon_fg_weight > 0:  tag += f"_fg{a.recon_fg_weight:g}"
    if a.residual_budget > 0:  tag += f"_gray{a.residual_budget:g}"
    if a.learn_coeffs:         tag += "_learn"
    tag += "_bc" if a.decoder == "broadcast" else "_mlpdec"   # decoder in the name -> own folder
    experiment = f"{a.run}_{tag}" + (f"_{a.note}" if a.note else "")
    data_path  = C.DATASETS_DIR / f"{a.run}.h5"
    ckpt_path  = C.CHECKPOINTS_DIR / f"{experiment}.pt"
    report_dir = C.RESULTS_DIR / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path}")

    print(f"=== {experiment} ===")
    print(f"device={a.device}  data={data_path.name}  state_dim={STATE_DIM}  "
          f"lam_dyn={a.lam_dyn}  lam_pred={a.lam_pred}  lam_var={a.lam_var}(gamma={a.var_gamma})  "
          f"residual_budget={a.residual_budget}")

    torch.manual_seed(C.SEED)
    train_dl, val_dl = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED, step=a.pred_step)
    model = EgoWorldModel(grid_size=a.grid_size, dt=a.pred_step * C.DT,
                          residual_budget=a.residual_budget, learn_coeffs=a.learn_coeffs,
                          decoder=a.decoder).to(a.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best, step = float("inf"), 0
    train_hist, val_hist = [], []
    for epoch in range(a.epochs):
        for b in train_dl:
            frame, nxt = b["frame"].to(a.device), b["next_frame"].to(a.device)
            out = model(frame, b["action"].to(a.device), nxt)
            loss, parts = ego_loss(out, frame, nxt, a.lam_dyn, a.lam_pred,
                                   a.lam_var, a.var_gamma, a.recon_fg_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); step += 1
            if step % 50 == 0:
                per_dim = out["s"].std(0).tolist()                 # [std d0, d1, d2, d3]
                st_std = sum(per_dim) / len(per_dim)
                row = {"step": step, "epoch": epoch + 1, **parts,
                       "a_v": model.log_a_v.exp().item(), "a_omega": model.log_a_omega.exp().item(),
                       "state_std": st_std,
                       "std0": per_dim[0], "std1": per_dim[1], "std2": per_dim[2], "std3": per_dim[3]}
                train_hist.append(row)
                print(f"  step {step:5d}  total {parts['total']:.4f}  recon {parts['recon']:.4f}  "
                      f"dyn {parts['dyn']:.4f}  pred_recon {parts['pred_recon']:.4f}  var {parts['var']:.4f}  "
                      f"std[{per_dim[0]:.2f} {per_dim[1]:.2f} {per_dim[2]:.2f} {per_dim[3]:.2f}]")
        v = val_total(model, val_dl, a.device, a)
        val_hist.append({"epoch": epoch + 1, "step": step, "val_total": v})
        print(f"epoch {epoch+1}/{a.epochs}  val total {v:.4f}")
        if v < best:
            best = v
            torch.save(model.state_dict(), ckpt_path)
            print(f"  new best {best:.4f} -> {ckpt_path}")

    # history CSVs (so the long logs are readable as data)
    def _write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
    _write_csv(report_dir / "train_history.csv", train_hist,
               ["step", "epoch", "total", "recon", "dyn", "pred_recon", "var",
                "a_v", "a_omega", "state_std", "std0", "std1", "std2", "std3"])
    _write_csv(report_dir / "val_history.csv", val_hist, ["epoch", "step", "val_total"])
    print(f"wrote train_history.csv ({len(train_hist)} rows) + val_history.csv")

    model.load_state_dict(torch.load(ckpt_path, map_location=a.device, weights_only=True))

    # ---- evaluation ----
    tr_dl, va_dl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    Z_tr, S_tr = extract_states(model, tr_dl, a.device)
    Z_va, S_va = extract_states(model, va_dl, a.device)
    T_tr = state_to_target(S_tr)

    print(f"\nlearned gray-box coefficients: a_v={model.log_a_v.exp().item():.4f}  "
          f"a_omega={model.log_a_omega.exp().item():.4f}  (toy target ~1.0)")

    print("\n-- pose probed from the 4-dim ego state (gauge-invariant) --")
    rows = []
    for name, hidden in (("linear", 0), ("mlp", 128)):
        torch.manual_seed(C.SEED)
        m = probe_pose(Z_tr, T_tr, Z_va, S_va, hidden, a.device)
        rows.append((name, m))
        print(f"  {name:7s}  theta_flip {m['theta_flip_pct']:6.2f}  theta_mae {m['theta_mae_deg']:6.2f}  "
              f"x_r2 {m['x_r2']:+.3f}  y_r2 {m['y_r2']:+.3f}")
    with open(report_dir / "state_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "theta_flip_pct", "theta_mae_deg", "x_r2", "y_r2"])
        for name, m in rows:
            w.writerow([name, f"{m['theta_flip_pct']:.4f}", f"{m['theta_mae_deg']:.4f}",
                        f"{m['x_r2']:.4f}", f"{m['y_r2']:.4f}"])

    print("\n-- per-dim correlation of the 4 state dims with pose (expect d0~x d1~y d2~cos d3~sin) --")
    T_va = state_to_target(S_va)
    for j, nm in enumerate(["x", "y", "cos_th", "sin_th"]):
        corr = pearson_per_dim(Z_va, T_va[:, j])
        print(f"  {nm:7s}  " + "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(STATE_DIM)))

    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
