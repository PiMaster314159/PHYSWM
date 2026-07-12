"""Grounded-JEPA evaluation (moved from run_grounded.py). Runs the shared probe, then the
block-specific diagnostics: per-dim correlations (is pose in the block?), a BLOCK-ONLY probe
(the direct grounding test), and actuator recovery when the block coeffs are learned."""
import csv
import h5py
import torch

import config as C
from models.dataset import make_dataloaders
from eval.probe import (
    run_probe, extract_latents, state_to_target, chance_baseline,
    make_linear_probe, make_mlp_probe, train_probe, evaluate_probe, save_probe_table,
)
from diagnose import pearson_per_dim


def evaluate(model, a, data_path, report_dir):
    device = a.device

    # shared probe (model-agnostic via .encode) -> metrics table
    run_probe(model=model, data_path=data_path, latent_dim=a.latent_dim,
              probe_epochs=a.probe_epochs, device=device, seed=C.SEED, save_dir=report_dir)

    tr_dl, va_dl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    Ztr, Str = extract_latents(model, tr_dl, device)
    Zva, Sva = extract_latents(model, va_dl, device)
    K = a.block_dim
    names = ["x", "y", "cos_th", "sin_th"]

    # (a) per-dim signed correlations (val), ALL dims -> CSV; block dims flagged
    corrs = torch.stack([pearson_per_dim(Zva, state_to_target(Sva)[:, j]) for j in range(4)], dim=1)  # (D,4)
    corr_path = report_dir / "block_correlations.csv"
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "in_block", "corr_x", "corr_y", "corr_cos", "corr_sin"])
        for d in range(Zva.shape[1]):
            w.writerow([d, int(d < K)] + [f"{corrs[d, j].item():.4f}" for j in range(4)])
    print(f"wrote {corr_path}")
    print(f"\n-- per-dim |correlation| with pose (block = dims 0..{K - 1}) --")
    for j, nm in enumerate(names):
        corr = corrs[:, j]
        blk  = "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(K))
        top  = torch.topk(corr.abs(), 5)
        tops = "  ".join(f"d{int(i)}={corr[int(i)]:+.2f}" for i in top.indices)
        print(f"  {nm:8s}  block[{blk}]   top5  {tops}")

    # (b) BLOCK-ONLY probe: read pose from JUST the block dims (direct grounding test)
    res = {"chance": chance_baseline(Str, Sva)}
    for nm, fac in [("linear", make_linear_probe), ("mlp", make_mlp_probe)]:
        torch.manual_seed(C.SEED)
        p = train_probe(fac(K), Ztr[:, :K], state_to_target(Str), epochs=a.probe_epochs, device=device)
        res[nm] = evaluate_probe(p, Zva[:, :K], Sva, device=device)
    save_probe_table(res, report_dir / "block_probe.csv")
    print(f"\n-- BLOCK-ONLY probe (pose from the {K} block dims; high heading flip = heading NOT grounded) --")
    for nm in ("chance", "linear", "mlp"):
        m = res[nm]
        print(f"  {nm:7s}  theta_flip {m['theta_flip_pct']:6.2f}  theta_mae {m['theta_mae_deg']:6.2f}  "
              f"x_r2 {m['x_r2']:+.3f}  y_r2 {m['y_r2']:+.3f}")

    if a.learn_coeffs:
        with h5py.File(data_path, "r") as _f:
            true_gain = float(_f.attrs.get("actuator_gain", 1.0))
        learned_av = model.predictor.log_a_v.exp().item()
        print(f"\nactuator recovery: true gain={true_gain:.3f}  ->  learned block a_v={learned_av:.4f}  "
              f"a_omega={model.predictor.log_a_omega.exp().item():.4f}  (abs error {abs(learned_av - true_gain):.4f})")
        with open(report_dir / "actuator_recovery.csv", "w", newline="") as _fc:
            _w = csv.writer(_fc); _w.writerow(["true_gain", "learned_a_v", "a_omega"])
            _w.writerow([f"{true_gain:.4f}", f"{learned_av:.4f}", f"{model.predictor.log_a_omega.exp().item():.4f}"])
