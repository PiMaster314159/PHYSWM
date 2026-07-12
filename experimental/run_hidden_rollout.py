"""Open-loop rollout drift for the HIDDEN-STATE ego model: does the recovered per-episode gain a_v
actually make multi-step prediction accurate?

For each held-out window: encode the frame stack at t=0 (giving pose_0 AND the episode's a_v=h),
then chain the gray-box dynamics K steps forward on the TRUE actions (no re-encoding). We do this
three ways, all sharing the same encoded pose_0 and only swapping the speed gain:
  - model  : a_v = g(h), the gain the encoder inferred for this episode
  - one    : a_v = 1, the "no gain modeling" baseline (locked kinematics)
  - oracle : a_v = the true per-episode gain (upper bound if the gain were perfect)
and report position-drift RMSE vs horizon for each, plus the correlation between each window's
final drift and |a_v - true_gain| (does the gain error drive the drift?).

The gap (one -> model) is how much the recovered gain helps; the gap (model -> oracle) is what is
left on the table by imperfect gain recovery. 1-step anchors are blind to this; it only shows in
the accumulated rollout.

    python run_hidden_rollout.py --run run09_64x64_varactuator --grid-size 64 --pred-step 4 --epochs 40 --rollout-k 8            # unsupervised ckpt
    python run_hidden_rollout.py --run run09_64x64_varactuator --grid-size 64 --pred-step 4 --epochs 40 --rollout-k 8 --gsup     # supervised ckpt
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

import numpy as np
import torch
torch.set_num_threads(4)

import config as C
from models.hidden_ego import HiddenEgoWorldModel
from models.dataset import make_history_rollout_dataloaders


def pearson(a, b):
    a, b = a.flatten(), b.flatten()
    return ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8)


@torch.no_grad()
def rollout_drift(model, dl, device, K):
    """Per-horizon summed squared position error for the three gain modes, plus per-window final
    drift and a_v / true-gain for the correlation."""
    model.eval()
    modes = ["model", "one", "oracle"]
    sq = {m: np.zeros(K + 1) for m in modes}     # summed squared euclidean position error per horizon
    n = 0
    final_drift, av_all, gain_all = [], [], []
    for b in dl:
        frames = b["frame"].to(device)              # (B, stack, H, W)
        ah     = b["action_hist"].to(device)        # (B, stack, ACTION_DIM)
        racts  = b["roll_actions"].to(device)       # (B, K, ACTION_DIM)
        true_xy = b["poses"][:, :, :2].to(device)   # (B, K+1, 2)
        s0 = model.encode(frames, ah)               # (B, 5) = [pose | hidden]
        av_model = model.gain(s0)                    # (B, 1)
        has_gain = "gain" in b
        true_gain = b["gain"].to(device) if has_gain else av_model

        def with_gain(av):
            s = s0.clone(); s[:, 4:5] = av; return s
        starts = {"model": s0.clone(), "one": with_gain(torch.ones_like(av_model)),
                  "oracle": with_gain(true_gain)}

        for m, s in starts.items():
            cur = s
            sq[m][0] += (cur[:, :2] - true_xy[:, 0]).pow(2).sum(1).sum().item()   # horizon 0 (perception)
            for k in range(K):
                cur = model.step(cur, racts[:, k])
                sq[m][k + 1] += (cur[:, :2] - true_xy[:, k + 1]).pow(2).sum(1).sum().item()
            if m == "model":
                final_drift.append((cur[:, :2] - true_xy[:, K]).pow(2).sum(1).sqrt().cpu())
        n += s0.shape[0]
        if has_gain:
            av_all.append(av_model.cpu()); gain_all.append(b["gain"])

    rmse = {m: np.sqrt(sq[m] / max(n, 1)) for m in modes}      # position-drift RMSE per horizon
    final_drift = torch.cat(final_drift)
    av_all = torch.cat(av_all) if av_all else None
    gain_all = torch.cat(gain_all) if gain_all else None
    return rmse, n, final_drift, av_all, gain_all


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop rollout drift for the hidden-state ego model.")
    p.add_argument("--run", default=C.RUN)
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40, help="epoch count in the checkpoint tag (e.g. 40 -> _e40_)")
    p.add_argument("--rollout-k", type=int, default=8, help="open-loop horizon (steps at stride pred_step)")
    p.add_argument("--gsup", action="store_true", help="load the SUPERVISED checkpoint (default: unsupervised)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    suffix = "gsup" if a.gsup else "gunsup"
    experiment = f"{a.run}_hidego_s{a.pred_step}_e{a.epochs}_st{a.stack}_h{a.hidden_dim}_{suffix}"
    data_path = C.DATASETS_DIR / f"{a.run}.h5"
    ckpt_path = C.CHECKPOINTS_DIR / f"{experiment}.pt"
    report_dir = C.RESULTS_DIR / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path}")

    print(f"=== ROLLOUT {experiment}  (K={a.rollout_k}) ===")
    torch.manual_seed(C.SEED)
    _, val_dl = make_history_rollout_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                                 stack=a.stack, K=a.rollout_k, step=a.pred_step)
    print(f"{len(val_dl.dataset)} val rollout windows")
    model = HiddenEgoWorldModel(grid_size=a.grid_size, dt=a.pred_step * C.DT,
                                stack=a.stack, hidden_dim=a.hidden_dim).to(a.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=a.device, weights_only=True))

    rmse, n, final_drift, av_all, gain_all = rollout_drift(model, val_dl, a.device, a.rollout_k)

    print("\n-- position-drift RMSE vs horizon (open-loop) --")
    print(f"  {'k':>3}   {'a_v=1':>8}  {'model a_v':>10}  {'oracle a_v':>11}")
    for k in range(a.rollout_k + 1):
        print(f"  {k:>3}   {rmse['one'][k]:8.4f}  {rmse['model'][k]:10.4f}  {rmse['oracle'][k]:11.4f}")

    if av_all is not None:
        gerr = (av_all - gain_all).abs().flatten()
        corr = pearson(final_drift, gerr).item()
        print(f"\n-- gain error drives drift? --")
        print(f"  corr( final drift , |a_v - true_gain| ) = {corr:+.3f}")
        print(f"  mean |a_v - true_gain| = {gerr.mean().item():.4f}   mean final drift (model) = {final_drift.mean().item():.4f}")

    with open(report_dir / f"rollout_drift_k{a.rollout_k}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["horizon", "drift_one", "drift_model", "drift_oracle"])
        for k in range(a.rollout_k + 1):
            w.writerow([k, f"{rmse['one'][k]:.5f}", f"{rmse['model'][k]:.5f}", f"{rmse['oracle'][k]:.5f}"])
    print(f"\nwrote rollout_drift_k{a.rollout_k}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = np.arange(a.rollout_k + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, rmse["one"],    "o-", label="a_v = 1 (no gain model)", color="#c0392b")
        ax.plot(ks, rmse["model"],  "o-", label="model a_v = g(h)",        color="#2c7fb8")
        ax.plot(ks, rmse["oracle"], "o--", label="oracle a_v (true gain)", color="#31a354")
        ax.set_xlabel("rollout horizon (steps)"); ax.set_ylabel("position-drift RMSE")
        ax.set_title(f"open-loop rollout drift  ({suffix})")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        out = report_dir / f"rollout_drift_k{a.rollout_k}.png"
        fig.savefig(out, dpi=130); print(f"wrote {out.name}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
