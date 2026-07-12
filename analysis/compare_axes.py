"""Side-by-side latent-probe-axes comparison across the four state-supervised models.

Run on the tower (needs torch + the run06 dataset + the four checkpoints):

    python compare_axes.py

Produces results/_compare/axes_4way.png (4 rows, one per model) and also drops a
per-model latent_probe_axes.png into each model's results folder (so the grounded and
ego runs get the figure they did not generate at train time).

The panel is gauge-invariant: it fits a linear probe over the WHOLE latent and plots the
probe's readout, so it is directly comparable across the JEPA, grounded, and ego latents
even though they have different latent dimensions (128 vs 128 vs 4).
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
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from models.dataset import make_dataloaders
from models.jepa import JEPA
from models.grounded import GroundedJEPA
from models.state_ae import EgoWorldModel
from eval.probe import extract_latents, make_linear_probe, train_probe, state_to_target

GRID = 64
CK   = C.CHECKPOINTS_DIR
RUN  = "run06_64x64_bignose"   # overridden in main() from --run
DATA = C.DATASETS_DIR / f"{RUN}.h5"


def build_models():
    """(label, model, checkpoint path, results-folder name) for each run, in order."""
    return [
        ("residual JEPA + readout head",
         JEPA(grid_size=GRID, latent_dim=128, predictor_mode="residual", state_head=True),
         CK / f"{RUN}_residual_s1_e20_rd1.pt", f"{RUN}_residual_s1_e20_rd1"),
        ("residual JEPA + anchor",
         JEPA(grid_size=GRID, latent_dim=128, predictor_mode="residual", state_head=False),
         CK / f"{RUN}_residual_s1_e20_anc1.pt", f"{RUN}_residual_s1_e20_anc1"),
        ("grounded + block anchor",
         GroundedJEPA(grid_size=GRID, latent_dim=128, block_dim=4, dt=C.DT,
                      lock_block=True, block_budget=0.0, use_decoder=False),
         CK / f"{RUN}_grounded_s1_e20_anc1.pt", f"{RUN}_grounded_s1_e20_anc1"),
        ("ego + anchor + anchor_pred",
         EgoWorldModel(grid_size=GRID, dt=C.DT, residual_budget=0.0, learn_coeffs=False, decoder="mlp"),
         CK / f"{RUN}_ego_s1_e20_fg5_anc1_ancp1_mlpdec.pt", f"{RUN}_ego_s1_e20_fg5_anc1_ancp1_mlpdec"),
    ]


def draw_row(axes_row, Z, S, fig, seed=0, epochs=80):
    """Fit a linear probe over Z and draw the 3 probe-decode panels into axes_row."""
    torch.manual_seed(seed)
    probe = make_linear_probe(Z.shape[1])
    probe = train_probe(probe, Z, state_to_target(S), epochs=epochs, device="cpu")
    with torch.no_grad():
        pred = probe(Z.cpu()).numpy()
    s = S.cpu().numpy()

    rng = np.random.default_rng(seed)
    n   = min(5000, Z.shape[0])
    idx = rng.choice(Z.shape[0], size=n, replace=False)
    pred, s = pred[idx], s[idx]
    px, py, pcos, psin = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx, ty, tth = s[:, 0], s[:, 1], s[:, 2]

    a0, a1, a2 = axes_row
    s0 = a0.scatter(px, py, c=tx, cmap="viridis", s=5, alpha=0.5)
    a0.set_xlabel("predicted x"); fig.colorbar(s0, ax=a0)
    s1 = a1.scatter(px, py, c=ty, cmap="viridis", s=5, alpha=0.5)
    a1.set_xlabel("predicted x"); fig.colorbar(s1, ax=a1)

    pth = np.arctan2(psin, pcos)
    d   = pth - tth
    ang = np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))
    s2 = a2.scatter(np.degrees(tth), np.degrees(pth), c=ang, cmap="viridis_r", s=5, alpha=0.5)
    a2.plot([-180, 180], [-180, 180], "k--", lw=1, alpha=0.5)
    a2.set_xlabel("true theta (deg)"); a2.set_ylabel("predicted theta (deg)")
    fig.colorbar(s2, ax=a2, label="ang err (deg)")
    return float(ang.mean())


def main():
    global RUN, DATA
    ap = argparse.ArgumentParser(description="4-way latent-probe-axes comparison.")
    ap.add_argument("--run", default=RUN, help="dataset/checkpoint stem (e.g. run07_64x64_ring)")
    a = ap.parse_args()
    RUN  = a.run
    DATA = C.DATASETS_DIR / f"{RUN}.h5"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}")
    models = build_models()

    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.3 * len(models)))
    col_titles = ["predicted position (color = true x)",
                  "predicted position (color = true y)",
                  "predicted vs true heading"]

    for r, (name, model, ckpt, folder) in enumerate(models):
        if not ckpt.exists():
            print(f"!! missing checkpoint, skipping row: {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        _, vdl = make_dataloaders(DATA, batch_size=256, seed=C.SEED, return_state=True, step=1)
        Z, S = extract_latents(model, vdl, device)
        mae = draw_row(axes[r], Z, S, fig)
        axes[r][0].set_ylabel(f"{name}\n(theta_mae {mae:.1f} deg)", fontsize=9, fontweight="bold")
        if r == 0:
            for c in range(3):
                axes[0][c].set_title(col_titles[c], fontsize=10)

        # also save this model's own latent_probe_axes.png into its results folder
        rep = C.RESULTS_DIR / folder
        if rep.exists():
            sub, sax = plt.subplots(1, 3, figsize=(15, 4.5))
            draw_row(sax, Z, S, sub)
            for c in range(3):
                sax[c].set_title(col_titles[c], fontsize=10)
            sub.suptitle(f"latent viewed along probe-decode axes  |  {name}", y=1.02)
            sub.tight_layout()
            sub.savefig(rep / "latent_probe_axes.png", dpi=110, bbox_inches="tight")
            plt.close(sub)
            print(f"wrote {rep / 'latent_probe_axes.png'}")
        print(f"row {r}: {name}  (latent dim {Z.shape[1]}, theta_mae {mae:.2f})")

    fig.suptitle("latent viewed along probe-decode axes  |  run06 big-nose, s1", y=1.003, fontsize=13)
    fig.tight_layout()
    out = C.RESULTS_DIR / "_compare"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "axes_4way.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out / 'axes_4way.png'}")


if __name__ == "__main__":
    main()
