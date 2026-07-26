"""Ego world-model evaluation (moved from run_ego.py). Reads the 4-dim state out of the
encoder and reports: learned gray-box coefficients + actuator recovery, direct pose readout
(anchored), a gauge-invariant probe (linear + MLP), per-dim correlations, and 1-step
prediction error. Called by train.py's per-model eval dispatch."""
import csv
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from models.components import state_to_target, pearson_per_dim, format_residual
from models.dataset import make_dataloaders
from eval import metrics


@torch.no_grad()
def extract_states(model, dl, device):
    """Encoded 4-dim states and true (x,y,theta) over a loader."""
    model.eval()
    enc, true = [], []
    for b in dl:
        enc.append(model.encode(b["frame"].to(device)).cpu())
        true.append(b["state"])
    return torch.cat(enc), torch.cat(true)


def probe_pose(Z_tr, T_tr, Z_va, S_va, hidden, device, epochs=120, bs=4096):
    """Fit a probe Z->(x,y,cos,sin); report heading flip %, mae, x/y R^2. Minibatched so the
    probe converges (full-batch GD underfits and gives bogus negative R^2)."""
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


def direct_readout(Z, S):
    """Read the 4-dim latent DIRECTLY as pose (no probe). Meaningful only with the anchor."""
    th_pred = torch.atan2(Z[:, 3], Z[:, 2])
    d   = th_pred - S[:, 2]
    ang = torch.abs(torch.atan2(torch.sin(d), torch.cos(d)))
    deg = 180.0 / torch.pi
    return {
        "theta_flip_pct": (ang > torch.pi / 2).float().mean().item() * 100.0,
        "theta_mae_deg":  (ang.mean() * deg).item(),
        "x_rmse": (Z[:, 0] - S[:, 0]).pow(2).mean().sqrt().item(),
        "y_rmse": (Z[:, 1] - S[:, 1]).pow(2).mean().sqrt().item(),
    }


@torch.no_grad()
def predict_eval(model, dl, device):
    """One-step prediction error: predict(encode(frame_t), action_t) vs TRUE next pose."""
    model.eval()
    pe_sum, he_sum, n = 0.0, 0.0, 0
    for b in dl:
        s      = model.encode(b["frame"].to(device))
        s_pred = model.predict(s, b["action"].to(device))
        tgt    = state_to_target(b["next_state"]).to(device)
        pe = (s_pred[:, :2] - tgt[:, :2]).pow(2).sum(1).sqrt()
        d  = torch.atan2(s_pred[:, 3], s_pred[:, 2]) - torch.atan2(tgt[:, 3], tgt[:, 2])
        he = torch.abs(torch.atan2(torch.sin(d), torch.cos(d)))
        pe_sum += pe.sum().item(); he_sum += he.sum().item(); n += pe.numel()
    return pe_sum / n, (he_sum / n) * 180.0 / torch.pi


def evaluate(model, a, data_path, report_dir):
    """Full ego evaluation, writing CSVs into report_dir."""
    device = a.device
    STATE_DIM = 4
    need_state = a.lam_anchor > 0 or a.lam_anchor_pred > 0 or a.rollout_k > 1

    tr_dl, va_dl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    Z_tr, S_tr = extract_states(model, tr_dl, device)
    Z_va, S_va = extract_states(model, va_dl, device)
    T_tr = state_to_target(S_tr)

    print(f"\nlearned gray-box coefficients: a_v={model.log_a_v.exp().item():.4f}  "
          f"a_omega={model.log_a_omega.exp().item():.4f}  (toy target ~1.0)")

    with h5py.File(data_path, "r") as _f:
        true_gain = float(_f.attrs.get("actuator_gain", 1.0))
    learned_av = model.log_a_v.exp().item()
    print(f"actuator recovery: true gain={true_gain:.3f}  ->  learned a_v={learned_av:.4f}  "
          f"(abs error {abs(learned_av - true_gain):.4f})")
    with open(report_dir / "actuator_recovery.csv", "w", newline="") as _fc:
        _w = csv.writer(_fc); _w.writerow(["true_gain", "learned_a_v", "a_omega"])
        _w.writerow([f"{true_gain:.4f}", f"{learned_av:.4f}", f"{model.log_a_omega.exp().item():.4f}"])

    if getattr(model, "residual", None) is not None and hasattr(model.residual, "named_coeffs"):  # structured only
        print("\n" + format_residual(model.residual))

    if need_state:   # anchored: the latent IS pose, so read it straight off (camera-only, no probe)
        dm = direct_readout(Z_va, S_va)
        print("\n-- pose read DIRECTLY from the latent (no probe; camera-only inference) --")
        print(f"  direct   theta_flip {dm['theta_flip_pct']:6.2f}  theta_mae {dm['theta_mae_deg']:6.2f}  "
              f"x_rmse {dm['x_rmse']:.4f}  y_rmse {dm['y_rmse']:.4f}")
        with open(report_dir / "state_direct.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["theta_flip_pct", "theta_mae_deg", "x_rmse", "y_rmse"])
            w.writerow([f"{dm['theta_flip_pct']:.4f}", f"{dm['theta_mae_deg']:.4f}",
                        f"{dm['x_rmse']:.4f}", f"{dm['y_rmse']:.4f}"])

    print("\n-- pose probed from the 4-dim ego state (gauge-invariant) --")
    rows = []
    for name, hidden in (("linear", 0), ("mlp", 128)):
        torch.manual_seed(C.SEED)
        m = probe_pose(Z_tr, T_tr, Z_va, S_va, hidden, device)
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

    pred_pos_err, pred_theta_mae = predict_eval(model, va_dl, device)
    print("\n-- 1-step prediction (dynamics: predict(encode(frame), action) vs true next pose) --")
    print(f"  predict  pos_err {pred_pos_err:.4f}  theta_mae {pred_theta_mae:.2f}")
    with open(report_dir / "predict_eval.csv", "w", newline="") as _fp:
        _w = csv.writer(_fp); _w.writerow(["pred_pos_err", "pred_theta_mae_deg"])
        _w.writerow([f"{pred_pos_err:.4f}", f"{pred_theta_mae:.4f}"])

    metrics.finalize(model, a, data_path, report_dir,
                     coeffs={"a_v": model.log_a_v.exp().item(), "a_omega": model.log_a_omega.exp().item()},
                     dynamics={"pred_pos_err": pred_pos_err, "pred_theta_mae_deg": pred_theta_mae})
