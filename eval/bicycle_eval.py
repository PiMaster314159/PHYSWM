"""Lean evaluation for the BICYCLE / throttle models. The whole point of the bicycle task is that
VELOCITY IS HIDDEN (it never appears in a single frame), so the headline metric is velocity recovery:
does the model read speed off the frame STACK? Model-agnostic via encode / decode_pose / predict, so
ego / grounded / jepa all run through it. Reports pose accuracy, velocity recovery, and 1-step prediction.
"""
import json
import math
import torch

import config as C
from models.dataset import make_dataloaders
from models.components import state_to_target


@torch.no_grad()
def _decoded_vs_true(model, dl, device):
    """Decoded pose [x,y,cos,sin,v] and the true 5-D target over a loader."""
    model.eval()
    P, T = [], []
    for b in dl:
        z = model.encode(b["frame"].to(device))
        P.append(model.decode_pose(z).cpu())
        T.append(state_to_target(b["state"], b.get("velocity")))
    return torch.cat(P), torch.cat(T)


@torch.no_grad()
def _predict_err(model, dl, device):
    """One-step prediction error: predict(encode(stack), action) vs the true next 5-D state."""
    model.eval()
    pe = he = ve = 0.0; n = 0
    for b in dl:
        z = model.encode(b["frame"].to(device))
        s = model.decode_pose(model.predict(z, b["action"].to(device))).cpu()
        t = state_to_target(b["next_state"], b.get("next_velocity"))
        pe += (s[:, :2] - t[:, :2]).pow(2).sum(1).sqrt().sum().item()
        d   = torch.atan2(s[:, 3], s[:, 2]) - torch.atan2(t[:, 3], t[:, 2])
        he += torch.abs(torch.atan2(torch.sin(d), torch.cos(d))).sum().item()
        ve += (s[:, 4] - t[:, 4]).abs().sum().item()
        n  += s.shape[0]
    return pe / n, he / n * 180.0 / math.pi, ve / n


def evaluate(model, a, data_path, report_dir):
    device = a.device
    _, va = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    P, T = _decoded_vs_true(model, va, device)

    pos_rmse = (P[:, :2] - T[:, :2]).pow(2).sum(1).mean().sqrt().item()
    d = torch.atan2(P[:, 3], P[:, 2]) - torch.atan2(T[:, 3], T[:, 2])
    theta_mae = torch.abs(torch.atan2(torch.sin(d), torch.cos(d))).mean().item() * 180.0 / math.pi
    v_rmse = (P[:, 4] - T[:, 4]).pow(2).mean().sqrt().item()
    v_corr = torch.corrcoef(torch.stack([P[:, 4], T[:, 4]]))[0, 1].item()
    pe, hmae, ve = _predict_err(model, va, device)

    print("\n-- bicycle eval (pose read from the frame stack; velocity is HIDDEN) --")
    print(f"  pose      pos_rmse {pos_rmse:.4f}  theta_mae {theta_mae:6.2f} deg")
    print(f"  VELOCITY  v_rmse {v_rmse:.4f}  v_corr {v_corr:+.3f}   <- did it recover hidden speed from history?")
    print(f"  1-step    pos_err {pe:.4f}  theta_mae {hmae:6.2f} deg  v_err {ve:.4f}")

    metrics = {"model": a.model, "run": a.run, "dynamics": "bicycle", "n_frames": a.n_frames,
               "pose": {"pos_rmse": pos_rmse, "theta_mae_deg": theta_mae},
               "velocity": {"v_rmse": v_rmse, "v_corr": v_corr},
               "predict": {"pos_err": pe, "theta_mae_deg": hmae, "v_err": ve}}
    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {report_dir / 'metrics.json'}")
