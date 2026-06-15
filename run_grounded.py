"""Train + probe a grounded-physics-block JEPA on an existing dataset.

    python run_grounded.py --run run04_64x64_nose --grid-size 64 --pred-step 4 --epochs 10
    python run_grounded.py --run run04_64x64_nose --grid-size 64 --no-lock-block --block-budget 0.1   # gray-box
    python run_grounded.py --run spin_64x64_nose  --grid-size 64 --use-decoder --lam-recon 1.0         # add the decoder

Label-free grounding: no state supervision. State is loaded ONLY to score the
probe and the per-dim correlations at the end. Mirrors run.py but for GroundedJEPA.
The per-dim block (expect dim0~x, dim1~y, dim2~cos, dim3~sin) is the headline:
if grounding worked, pose now lives in the block, not smeared across the latent.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
torch.set_num_threads(4)

import config as C
from models.grounded import GroundedJEPA, grounded_loss
from models.dataset import make_dataloaders
from eval.probe import run_probe, extract_latents, state_to_target
from diagnose import pearson_per_dim


def parse_args():
    p = argparse.ArgumentParser(description="Train + probe a grounded-physics-block JEPA.")
    p.add_argument("--run", default=C.RUN, help="dataset name (.h5 stem); must already exist")
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--block-dim", type=int, default=C.PHYSICS_BLOCK_DIM)
    p.add_argument("--latent-dim", type=int, default=C.LATENT_DIM)
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    # grounding knobs
    p.add_argument("--no-lock-block", dest="lock_block", action="store_false",
                   default=C.GROUNDED_LOCK_BLOCK, help="gray-box: allow a bounded learned block correction")
    p.add_argument("--block-budget", type=float, default=C.GROUNDED_BLOCK_BUDGET,
                   help="max scale of the learned block correction (gray-box only)")
    p.add_argument("--use-decoder", action="store_true", help="reconstruct frame from the block (grounding booster)")
    p.add_argument("--lam-recon", type=float, default=C.LAM_RECON, help="decoder reconstruction weight")
    # training
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--lam", type=float, default=C.LAM, help="SIGReg weight (free dims only)")
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--note", default="", help="optional label appended to the experiment name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def val_pred(model, dl, device, max_batches=50):
    """Mean prediction MSE over the val loader (the metric we checkpoint on)."""
    model.eval()
    losses = []
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        out = model(b["frame"].to(device), b["action"].to(device), b["next_frame"].to(device))
        losses.append(F.mse_loss(out["pred_next_z"], out["target_next_z"]).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    a = parse_args()

    tag = f"grounded_s{a.pred_step}_e{a.epochs}"
    if not a.lock_block:   tag += f"_gray{a.block_budget:g}"
    if a.lam_recon > 0:    tag += f"_rec{a.lam_recon:g}"
    experiment = f"{a.run}_{tag}" + (f"_{a.note}" if a.note else "")
    data_path  = C.DATASETS_DIR / f"{a.run}.h5"
    ckpt_path  = C.CHECKPOINTS_DIR / f"{experiment}.pt"
    report_dir = C.RESULTS_DIR / experiment
    report_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path} (this runner does not collect; build it first)")

    print(f"=== {experiment} ===")
    print(f"device={a.device}  data={data_path.name}  block_dim={a.block_dim}  "
          f"lock_block={a.lock_block}  budget={a.block_budget}  lam_recon={a.lam_recon}")

    torch.manual_seed(C.SEED)
    train_dl, val_dl = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED, step=a.pred_step)
    model = GroundedJEPA(
        grid_size=a.grid_size, latent_dim=a.latent_dim, block_dim=a.block_dim,
        dt=a.pred_step * C.DT, lock_block=a.lock_block, block_budget=a.block_budget,
        use_decoder=(a.use_decoder or a.lam_recon > 0),   # decoder grounds the block; auto-on with lam_recon
    ).to(a.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best = float("inf")
    step = 0
    for epoch in range(a.epochs):
        for b in train_dl:
            frame = b["frame"].to(a.device)
            out   = model(frame, b["action"].to(a.device), b["next_frame"].to(a.device))
            loss, parts = grounded_loss(out, frame, block_dim=a.block_dim, lam=a.lam, lam_recon=a.lam_recon)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # governor on the block feedback loop
            opt.step()
            step += 1
            if step % 50 == 0:
                z = out["z"]
                blk_std  = z[:, :a.block_dim].std(0).mean().item()
                free_std = z[:, a.block_dim:].std(0).mean().item()
                rec = f" recon {parts['recon']:.4f}" if "recon" in parts else ""
                print(f"  step {step:5d}  total {parts['total']:.4f}  pred {parts['pred']:.4f}  "
                      f"sigreg {parts['sigreg']:.4f}{rec}  block_std {blk_std:.3f}  free_std {free_std:.3f}")

        v = val_pred(model, val_dl, a.device)
        print(f"epoch {epoch+1}/{a.epochs}  val pred {v:.4f}")
        if v < best:
            best = v
            torch.save(model.state_dict(), ckpt_path)
            print(f"  new best val pred {best:.4f} -> {ckpt_path}")

    # reload best
    model.load_state_dict(torch.load(ckpt_path, map_location=a.device, weights_only=True))

    # probe (reuses eval.probe; model-agnostic via .encode) and save the metrics table
    run_probe(model=model, data_path=data_path, latent_dim=a.latent_dim,
              probe_epochs=a.probe_epochs, device=a.device, seed=C.SEED, save_dir=report_dir)

    # per-dim correlations: did pose move into the block?
    _, sdl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    Z, S = extract_latents(model, sdl, a.device)
    T = state_to_target(S)
    names = ["x", "y", "cos_th", "sin_th"]
    print("\n-- per-dim |correlation| with pose (block = dims 0..{}) --".format(a.block_dim - 1))
    for j, nm in enumerate(names):
        corr = pearson_per_dim(Z, T[:, j])
        blk  = "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(a.block_dim))
        top  = torch.topk(corr.abs(), 5)
        tops = "  ".join(f"d{int(i)}={corr[int(i)]:+.2f}" for i in top.indices)
        print(f"  {nm:8s}  block[{blk}]   top5  {tops}")

    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
