"""JEPA evaluation (moved from run.py). Runs the shared probe, the per-dim correlation of
dims 0..3 when anchored, latent + reconstruction figures, and the physics-mode scales."""
import csv

import config as C
from models.dataset import make_dataloaders
from eval.probe import (
    run_probe, extract_latents, state_to_target, plot_latent_pca, plot_latent_probe_axes,
)
from eval.reconstruct import train_decoder, plot_reconstructions, plot_pose_reconstructions
from eval import metrics


def evaluate(model, a, data_path, report_dir):
    device = a.device
    need_state = a.lam_anchor > 0 or a.lam_anchor_pred > 0

    run_probe(model=model, data_path=data_path, latent_dim=a.latent_dim,
              probe_epochs=a.probe_epochs, device=device, seed=C.SEED, save_dir=report_dir)

    if need_state:   # per-dim correlation of the first 4 dims with pose
        _, cdl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
        Zc, Sc = extract_latents(model, cdl, device)
        Tc = state_to_target(Sc)
        Zc4 = Zc[:, :4] - Zc[:, :4].mean(0, keepdim=True)
        print("\n-- per-dim correlation of latent dims 0..3 with pose (anchored) --")
        with open(report_dir / "block_correlations.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pose", "d0", "d1", "d2", "d3"])
            for j, nm in enumerate(["x", "y", "cos_th", "sin_th"]):
                tc = Tc[:, j] - Tc[:, j].mean()
                corr = (Zc4 * tc.unsqueeze(1)).mean(0) / (Zc[:, :4].std(0) * Tc[:, j].std() + 1e-8)
                print(f"  {nm:7s}  " + "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(4)))
                w.writerow([nm] + [f"{corr[d]:.4f}" for d in range(4)])

    if not a.no_figures:
        _, vdl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
        Z, S = extract_latents(model, vdl, device)
        plot_latent_pca(Z, S, save_to=report_dir / "latent_pca.png")
        plot_latent_probe_axes(Z, S, save_to=report_dir / "latent_probe_axes.png")
        tdl, _ = make_dataloaders(data_path, seed=C.SEED, return_state=True, step=a.pred_step)
        decoder = train_decoder(model, tdl, device=device)
        plot_reconstructions(model, decoder, vdl, device=device, save_to=report_dir / "reconstructions.png")
        plot_pose_reconstructions(model, vdl, device=device, save_to=report_dir / "pose_reconstructions.png")

    if a.predictor_mode == "physics":
        print(f"learned scales: a_pos={model.predictor.log_a_pos.exp().item():.3f}  "
              f"a_theta={model.predictor.log_a_theta.exp().item():.3f}")

    metrics.finalize(model, a, data_path, report_dir)
