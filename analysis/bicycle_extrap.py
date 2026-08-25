"""Does the learned residual EXTRAPOLATE? Train inside a narrow low-speed band, test outside it.

Drag makes this test unusually clean. The unmodeled term is -drag_c*v^2, so it is nearly invisible at
low speed and dominates at high speed. Train the models where drag barely matters, then measure
one-step error as a function of the true speed at test time:

  * inside the training band every residual looks the same -- a flexible MLP interpolates fine, which
    is exactly what the in-distribution fit table already showed;
  * outside it the curves separate. A structured basis residual has learned a v^2 LAW and keeps
    working; an MLP has learned the band it was shown and degrades.

That separation is the whole claim: capacity buys you fit, structure buys you extrapolation. Unlike
the unicycle version (which swept v_ref through the MPC harness and so mixed model error with
controller behaviour) this is pure open-loop prediction -- no controller involved.

    python analysis/bicycle_extrap.py --train-run 64x64_bicycle_drag_narrow --test-run 64x64_bicycle_drag
"""
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
import train as T
from models.dataset import make_dataloaders, set_default_n_frames
from models.components import state_to_target

STYLE = {"basis": "-", "mlp": "--", "none": ":"}
COLOR = {"ego": "tab:blue", "grounded": "tab:red"}


def _args_for(run, model, residual, n_frames, epochs):
    """Rebuild the exact argparse namespace a training run used, so build_tag lands on the same
    checkpoint. Mirrors main()'s post-processing (per-model defaults, then grid/dynamics inferred
    from the dataset) -- those run AFTER parse_args and before build_tag, so skipping them would
    silently produce a different tag and a 'checkpoint not found'."""
    argv = ["train.py", "--model", model, "--run", run, "--n-frames", str(n_frames),
            "--residual", residual, "--epochs", str(epochs),
            "--lam-anchor", "1", "--lam-anchor-pred", "1"]
    old, sys.argv = sys.argv, argv
    try:
        a = T.parse_args()
    finally:
        sys.argv = old
    if a.lam_recon is None:
        a.lam_recon = 1.0 if a.model == "ego" else C.LAM_RECON
    if a.recon_fg_weight is None:
        a.recon_fg_weight = 5.0 if a.model == "ego" else C.RECON_FG_WEIGHT
    with h5py.File(C.DATASETS_DIR / f"{run}.h5", "r") as f:
        a.grid_size = int(f.attrs["grid_size"])
        a.dynamics = str(f.attrs.get("dynamics", "unicycle"))
    return a


def _load(a, device):
    tag = T.build_tag(a)
    ckpt, _ = C.experiment_paths(a.run, a.model, tag)
    if not ckpt.exists():
        return None, tag
    m = T.build_model(a).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.eval(), tag


@torch.no_grad()
def error_vs_speed(model, dl, device, edges):
    """Mean one-step velocity / position error, bucketed by the TRUE speed at the start of the step."""
    n = len(edges) - 1
    sv, sp, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for b in dl:
        z = model.encode(b["frame"].to(device))
        s = model.decode_pose(model.predict(z, b["action"].to(device))).cpu()
        t = state_to_target(b["next_state"], b.get("next_velocity"))
        v_now = b["velocity"].squeeze(-1).numpy()
        ve = (s[:, 4] - t[:, 4]).abs().numpy()
        pe = (s[:, :2] - t[:, :2]).pow(2).sum(1).sqrt().numpy()
        idx = np.clip(np.digitize(v_now, edges) - 1, 0, n - 1)
        for i in range(n):
            m = idx == i
            if m.any():
                sv[i] += ve[m].sum()
                sp[i] += pe[m].sum()
                cnt[i] += m.sum()
    safe = np.maximum(cnt, 1)
    return (np.where(cnt > 0, sv / safe, np.nan),
            np.where(cnt > 0, sp / safe, np.nan), cnt)


def train_band(run, lo=1.0, hi=99.0):
    """The speed range the training set actually covered -- everything above it is extrapolation."""
    with h5py.File(C.DATASETS_DIR / f"{run}.h5", "r") as f:
        v = f["velocities"][:]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    p = argparse.ArgumentParser(description="Speed-extrapolation curves for bicycle residuals.")
    p.add_argument("--train-run", required=True, help="narrow-band run the models were TRAINED on")
    p.add_argument("--test-run", required=True, help="wide run to EVALUATE on (covers higher speeds)")
    p.add_argument("--models", nargs="+", default=["ego", "grounded"])
    p.add_argument("--residuals", nargs="+", default=["none", "basis", "mlp"])
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", default=None, help="output folder under results/bicycle_extrap")
    a = p.parse_args()

    out_dir = ROOT / "results" / "bicycle_extrap" / (a.name or f"{a.train_run}__on__{a.test_run}")
    out_dir.mkdir(parents=True, exist_ok=True)

    v_lo, v_hi = train_band(a.train_run)
    with h5py.File(C.DATASETS_DIR / f"{a.test_run}.h5", "r") as f:
        v_test = f["velocities"][:]
    edges = np.linspace(float(v_test.min()), float(np.percentile(v_test, 99.5)), a.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"training band (1-99 pct): {v_lo:.3f} .. {v_hi:.3f}")
    print(f"test speeds binned over : {edges[0]:.3f} .. {edges[-1]:.3f}  ({a.bins} bins)")

    set_default_n_frames(a.n_frames)
    _, va = make_dataloaders(C.DATASETS_DIR / f"{a.test_run}.h5", batch_size=a.batch_size,
                             seed=C.SEED, return_state=True)

    rows, curves = [], {}
    for model in a.models:
        for residual in a.residuals:
            args = _args_for(a.train_run, model, residual, a.n_frames, a.epochs)
            m, tag = _load(args, a.device)
            if m is None:
                print(f"  skip {model:9s} {residual:5s}  (no checkpoint for tag {tag})")
                continue
            ve, pe, cnt = error_vs_speed(m, va, a.device, edges)
            curves[(model, residual)] = (ve, pe)
            inb = centers <= v_hi
            print(f"  {model:9s} {residual:5s}  in-band v_err {np.nanmean(ve[inb]):.4f}   "
                  f"out-of-band v_err {np.nanmean(ve[~inb]):.4f}")
            for c, e_v, e_p, n in zip(centers, ve, pe, cnt):
                rows.append({"model": model, "residual": residual, "v_bin": f"{c:.4f}",
                             "v_err": f"{e_v:.6f}", "pos_err": f"{e_p:.6f}", "n": int(n)})

    if not rows:
        raise SystemExit("no checkpoints found -- train the grid on --train-run first")

    csv_path = out_dir / "extrap.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "residual", "v_bin", "v_err", "pos_err", "n"])
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in ((axes[0], 0, "1-step velocity error"), (axes[1], 1, "1-step position error")):
        ax.axvspan(v_lo, v_hi, color="0.85", zorder=0)
        for (model, residual), vals in curves.items():
            ax.plot(centers, vals[key], STYLE.get(residual, "-"), color=COLOR.get(model, "k"),
                    marker="o", ms=3, label=f"{model} / {residual}")
        ax.set_xlabel("true speed v at test time")
        ax.set_ylabel(lab)
        ax.set_title(lab + "  (shaded = training band)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig_path = out_dir / "extrap.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()
