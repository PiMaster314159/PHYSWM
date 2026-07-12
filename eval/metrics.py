"""Unified metrics: one metrics.json per run with a common schema, so cross-model
comparison is a table read, not CSV-wrangling.

The headline pose numbers are computed through `model.decode_pose` — i.e. exactly what the
MPC controller reads — so ego/grounded/jepa are measured the same way and the number means
"control-relevant pose accuracy". Each eval hook calls `finalize(...)`; the MPC harness calls
`update_mpc(...)` to merge its control result into the same file.
"""
import csv
import json

import torch

import config as C
from models.components import state_to_target, pearson_per_dim
from models.dataset import make_dataloaders

POSE_NAMES = ["x", "y", "cos_th", "sin_th"]


@torch.no_grad()
def pose_accuracy(model, dl, device):
    """decode_pose(encode(frame)) vs true pose over a loader. Same readout the MPC uses."""
    model.eval()
    P, T = [], []
    for b in dl:
        P.append(model.decode_pose(model.encode(b["frame"].to(device))).cpu())
        T.append(state_to_target(b["state"]))
    P, T = torch.cat(P), torch.cat(T)                       # (N,4) [x,y,cosθ,sinθ]
    th_p, th_t = torch.atan2(P[:, 3], P[:, 2]), torch.atan2(T[:, 3], T[:, 2])
    d = torch.atan2(torch.sin(th_p - th_t), torch.cos(th_p - th_t)).abs()    # wrapped angular error
    deg = 180.0 / torch.pi

    def r2(p, t):
        return (1 - (p - t).pow(2).sum() / (t - t.mean()).pow(2).sum()).item()
    return {
        "theta_mae_deg":  (d.mean() * deg).item(),
        "theta_flip_pct": (d > torch.pi / 2).float().mean().item() * 100.0,
        "x_r2": r2(P[:, 0], T[:, 0]), "y_r2": r2(P[:, 1], T[:, 1]),
        "pos_rmse": (P[:, :2] - T[:, :2]).pow(2).sum(1).mean().sqrt().item(),
    }


@torch.no_grad()
def pose_dim_corr(model, dl, device, ndim=4):
    """Signed correlation of latent dims 0..ndim-1 with each pose component. (4 pose, ndim).
    Diagonal for ego/grounded (pose IS the first dims); distributed for jepa."""
    model.eval()
    Z, S = [], []
    for b in dl:
        Z.append(model.encode(b["frame"].to(device)).cpu()); S.append(b["state"])
    Z, T = torch.cat(Z)[:, :ndim], state_to_target(torch.cat(S))
    return torch.stack([pearson_per_dim(Z, T[:, j]) for j in range(4)], dim=0)   # (4, ndim)


def write_metrics(report_dir, **sections):
    """Merge sections into report_dir/metrics.json (so training and the MPC run can each add)."""
    path = report_dir / "metrics.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    rec.update(sections)
    path.write_text(json.dumps(rec, indent=2))
    return path


def finalize(model, a, data_path, report_dir, **sections):
    """Called at the end of each eval hook: compute decode_pose accuracy + the dim-correlation
    grid, merge with model-specific `sections` (coeffs, dynamics), write metrics.json + a
    pose_dim_corr.csv (for the grounding heatmap)."""
    _, va = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    pose = pose_accuracy(model, va, a.device)
    corr = pose_dim_corr(model, va, a.device)
    write_metrics(report_dir, model=a.model, run=a.run, tag=report_dir.name, pose=pose, **sections)
    with open(report_dir / "pose_dim_corr.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pose"] + [f"d{d}" for d in range(corr.shape[1])])
        for j, nm in enumerate(POSE_NAMES):
            w.writerow([nm] + [f"{corr[j, d].item():.4f}" for d in range(corr.shape[1])])
    print(f"\n[metrics] decode_pose accuracy: theta_mae {pose['theta_mae_deg']:.2f} deg  "
          f"flip {pose['theta_flip_pct']:.2f}%  x_r2 {pose['x_r2']:+.3f}  y_r2 {pose['y_r2']:+.3f}  "
          f"pos_rmse {pose['pos_rmse']:.4f}  -> {report_dir / 'metrics.json'}")


def update_mpc(report_dir, mpc):
    """Merge a control result into an existing run's metrics.json (called by the MPC harness)."""
    if report_dir.exists():
        write_metrics(report_dir, mpc=mpc)


def collect(results_root):
    """Every metrics.json under results/<run>/<model>/<tag>/ as flat dict rows."""
    rows = []
    for mj in sorted(results_root.glob("*/*/*/metrics.json")):
        rec = json.loads(mj.read_text())
        flat = {"run": rec.get("run"), "model": rec.get("model"), "tag": rec.get("tag")}
        for section in ("pose", "dynamics", "coeffs", "mpc"):
            for k, v in (rec.get(section) or {}).items():
                flat[f"{section}_{k}" if section != "pose" else k] = v
        rows.append(flat)
    return rows
