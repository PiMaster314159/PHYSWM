"""Standing diagnostic for a trained JEPA encoder (Phase 0b) + a JEPA-training
sanity check (Phase 0a).

Two modes:

  python diagnose.py --gradcheck --data data/datasets/run04_64x64_nose.h5
      Phase 0a. Builds a fresh JEPA, runs one jepa_loss backward pass, and
      confirms the ENCODER receives gradient (i.e. the encoder is actually
      trained by the JEPA objective, not frozen / bypassed). Reports per-block
      grad norms and the fraction of encoder params with nonzero grad.

  python diagnose.py <ckpt> --data <dataset.h5>
      Phase 0b. Freezes the checkpoint's encoder, fits an MLP probe, and prints:
        - heading metrics (flip %, MAE, median) + position/heading R^2
        - the top-5 latent dims correlated with x, y, cos th, sin th
        - the correlations at the physics pose dims 0,1,2 specifically
      Use this after every experiment. The marker/grid of <dataset> must match
      what the checkpoint was trained on.

Run from the PHYSWM root.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
torch.set_num_threads(4)

from config import DATA_PATH, SEED, LATENT_DIM
from models.jepa import JEPA, jepa_loss
from models.dataset import make_dataloaders
from eval.probe import (
    extract_latents, make_mlp_probe, train_probe, evaluate_probe,
    state_to_target, chance_baseline,
)


def infer_predictor_mode(state_dict: dict) -> str:
    """Read the predictor mode off a checkpoint so load_state_dict matches.

    Physics checkpoints carry predictor.log_a_pos; mlp/residual share the same
    parameter shapes (they differ only in forward logic), and the encoder we
    probe is identical across all three, so "residual" loads either cleanly.
    """
    if any("log_a_pos" in k for k in state_dict):
        return "physics"
    return "residual"


def pearson_per_dim(Z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-dim Pearson correlation between each latent dim and a target. (D,)."""
    Zc = Z - Z.mean(0, keepdim=True)
    tc = t - t.mean()
    num = (Zc * tc.unsqueeze(1)).mean(0)
    den = Z.std(0) * t.std() + 1e-8
    return num / den


def gradcheck(data_path: Path, device: str) -> None:
    """Phase 0a: confirm the encoder is in the JEPA optimization (gets gradient)."""
    train_dl, _ = make_dataloaders(data_path, batch_size=64, seed=SEED)
    grid = train_dl.dataset.grid_size
    model = JEPA(grid_size=grid, latent_dim=LATENT_DIM).to(device).train()

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    n_total = sum(p.numel() for p in model.parameters())
    n_enc   = sum(p.numel() for p in enc_params)
    print(f"params: {n_total} total, {n_enc} in encoder "
          f"({100.0 * n_enc / n_total:.1f}% of the model, all updated by Adam)")

    batch = next(iter(train_dl))
    out = model(batch["frame"].to(device), batch["action"].to(device), batch["next_frame"].to(device))
    loss, parts = jepa_loss(out, lam=0.005)
    model.zero_grad()
    loss.backward()

    def block_norm(prefix: str) -> float:
        g = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        return float(torch.sqrt(sum((gi ** 2).sum() for gi in g))) if g else 0.0

    enc_with_grad = sum(1 for p in enc_params if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"loss {loss.item():.4f}  (pred {parts['pred']:.4f}  sigreg {parts['sigreg']:.4f})")
    print(f"grad norm  encoder.conv={block_norm('encoder.conv'):.4e}  "
          f"encoder.head={block_norm('encoder.head'):.4e}  "
          f"predictor={block_norm('predictor'):.4e}")
    print(f"encoder param tensors with nonzero grad: {enc_with_grad}/{len(enc_params)}")
    ok = enc_with_grad == len(enc_params) and block_norm("encoder.conv") > 0
    print("PASS: encoder is trained by JEPA (gradient flows into every encoder block)"
          if ok else "FAIL: some encoder params get no gradient")


def diagnose(ckpt: Path, data_path: Path, device: str, probe_epochs: int = 40) -> None:
    """Phase 0b: heading metrics + per-dim pose correlations for one checkpoint."""
    train_dl, val_dl = make_dataloaders(data_path, batch_size=256, seed=SEED, return_state=True)
    grid = train_dl.dataset.grid_size

    sd = torch.load(ckpt, map_location=device)
    mode = infer_predictor_mode(sd)
    model = JEPA(grid_size=grid, latent_dim=LATENT_DIM, predictor_mode=mode)
    model.load_state_dict(sd)
    model.to(device)
    print(f"loaded {ckpt.name}  (predictor_mode={mode}, grid={grid}, data={data_path.name})")

    Z_tr, S_tr = extract_latents(model, train_dl, device)
    Z_va, S_va = extract_latents(model, val_dl, device)

    torch.manual_seed(SEED)
    probe = train_probe(make_mlp_probe(Z_tr.shape[1]), Z_tr, state_to_target(S_tr),
                        epochs=probe_epochs, device=device)
    m = evaluate_probe(probe, Z_va, S_va, device=device)
    ch = chance_baseline(S_tr, S_va)

    print("\n-- heading (MLP probe on frozen latent) --")
    print(f"  theta_flip_pct   {m['theta_flip_pct']:6.2f}   (chance {ch['theta_flip_pct']:.0f}, solved ~0)")
    print(f"  theta_median_deg {m['theta_median_deg']:6.2f}")
    print(f"  theta_mae_deg    {m['theta_mae_deg']:6.2f}")
    print(f"  pos x_r2/y_r2    {m['x_r2']:.3f} / {m['y_r2']:.3f}")

    # per-dim correlations on the full set (more samples = steadier estimate)
    Z_all = torch.cat([Z_tr, Z_va]).to(device)
    S_all = torch.cat([S_tr, S_va]).to(device)
    T_all = state_to_target(S_all)        # x, y, cos th, sin th
    names = ["x", "y", "cos_th", "sin_th"]

    print("\n-- per-dim |correlation| with pose --")
    print(f"  {'target':8s}  {'pose dims 0,1,2':28s}   top-5 dims (|corr|)")
    for j, nm in enumerate(names):
        corr = pearson_per_dim(Z_all, T_all[:, j])
        ac = corr.abs()
        top = torch.topk(ac, 5)
        pose = "  ".join(f"d{d}={corr[d]:+.2f}" for d in (0, 1, 2))
        tops = "  ".join(f"d{int(i)}={corr[int(i)]:+.2f}" for i in top.indices)
        print(f"  {nm:8s}  [{pose}]   {tops}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", help="checkpoint .pt (omit with --gradcheck)")
    ap.add_argument("--data", default=str(DATA_PATH), help="dataset .h5 (marker/grid must match the ckpt)")
    ap.add_argument("--gradcheck", action="store_true", help="Phase 0a: prove the encoder gets gradient")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data)

    if args.gradcheck:
        gradcheck(data_path, device)
    else:
        if not args.ckpt:
            ap.error("pass a checkpoint, or use --gradcheck")
        diagnose(Path(args.ckpt), data_path, device)
