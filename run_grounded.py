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
from eval.probe import (
    run_probe, extract_latents, state_to_target, chance_baseline,
    make_linear_probe, make_mlp_probe, train_probe, evaluate_probe, save_probe_table,
)
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
    p.add_argument("--recon-fg-weight", type=float, default=C.RECON_FG_WEIGHT,
                   help="foreground weighting of recon (bg_mean + w*fg_mean); decoder lever (grounds position)")
    p.add_argument("--lam-anchor", type=float, default=0.0,
                   help="privileged state anchor on the SIGReg-free block (raw pose); tests if the exempt dims encode pose properly under direct supervision")
    p.add_argument("--lam-anchor-pred", type=float, default=0.0,
                   help="anchor the PREDICTED next block to the true next pose (like the ego model); grounds heading + a_v through the dynamics at any pred_step (the s1 heading fix)")
    p.add_argument("--pred-block-weight", type=float, default=C.PRED_BLOCK_WEIGHT,
                   help="weight on block dims in the prediction loss; physics-consistency strength (the heading lever)")
    p.add_argument("--learn-coeffs", action="store_true",
                   help="learn block a_v/a_omega (gray-box scales); lets the block absorb an unmodeled actuator gain instead of locking it at 1")
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
    if a.recon_fg_weight > 0: tag += f"_fg{a.recon_fg_weight:g}"
    if a.pred_block_weight != 1.0: tag += f"_pb{a.pred_block_weight:g}"
    if a.lam_anchor > 0:   tag += f"_anc{a.lam_anchor:g}"
    if a.lam_anchor_pred > 0: tag += f"_ancp{a.lam_anchor_pred:g}"
    if a.learn_coeffs:     tag += "_learn"
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
    need_state = a.lam_anchor > 0 or a.lam_anchor_pred > 0   # anchors need true pose during training
    train_dl, val_dl = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                        return_state=need_state, step=a.pred_step)
    model = GroundedJEPA(
        grid_size=a.grid_size, latent_dim=a.latent_dim, block_dim=a.block_dim,
        dt=a.pred_step * C.DT, lock_block=a.lock_block, block_budget=a.block_budget,
        use_decoder=(a.use_decoder or a.lam_recon > 0),   # decoder grounds the block; auto-on with lam_recon
        learn_coeffs=a.learn_coeffs,
    ).to(a.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best = float("inf")
    step = 0
    for epoch in range(a.epochs):
        for b in train_dl:
            frame = b["frame"].to(a.device)
            out   = model(frame, b["action"].to(a.device), b["next_frame"].to(a.device))
            s_tgt      = state_to_target(b["state"]).to(a.device) if need_state else None
            s_next_tgt = state_to_target(b["next_state"]).to(a.device) if need_state else None
            loss, parts = grounded_loss(out, frame, block_dim=a.block_dim, lam=a.lam,
                                        lam_recon=a.lam_recon, recon_fg_weight=a.recon_fg_weight,
                                        pred_block_weight=a.pred_block_weight,
                                        s_target=s_tgt, lam_anchor=a.lam_anchor,
                                        s_next_target=s_next_tgt, lam_anchor_pred=a.lam_anchor_pred)
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
                anc = f" anchor {parts['anchor']:.4f}" if "anchor" in parts else ""
                ancp = f" ancp {parts['anchor_pred']:.4f}" if "anchor_pred" in parts else ""
                print(f"  step {step:5d}  total {parts['total']:.4f}  pred {parts['pred']:.4f}  "
                      f"pred_blk {parts['pred_block']:.4f}  sigreg {parts['sigreg']:.4f}{rec}{anc}{ancp}  "
                      f"block_std {blk_std:.3f}  free_std {free_std:.3f}")

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

    # ---- diagnostics: did pose move into the block? ----
    import csv
    tr_dl, va_dl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
    Ztr, Str = extract_latents(model, tr_dl, a.device)
    Zva, Sva = extract_latents(model, va_dl, a.device)
    K = a.block_dim
    names = ["x", "y", "cos_th", "sin_th"]

    # (a) per-dim signed correlations (on val), ALL dims -> CSV. block dims flagged.
    corrs = torch.stack([pearson_per_dim(Zva, state_to_target(Sva)[:, j]) for j in range(4)], dim=1)  # (D,4)
    corr_path = report_dir / "block_correlations.csv"
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "in_block", "corr_x", "corr_y", "corr_cos", "corr_sin"])
        for d in range(Zva.shape[1]):
            w.writerow([d, int(d < K)] + [f"{corrs[d, j].item():.4f}" for j in range(4)])
    print(f"wrote {corr_path}")
    print("\n-- per-dim |correlation| with pose (block = dims 0..{}) --".format(K - 1))
    for j, nm in enumerate(names):
        corr = corrs[:, j]
        blk  = "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(K))
        top  = torch.topk(corr.abs(), 5)
        tops = "  ".join(f"d{int(i)}={corr[int(i)]:+.2f}" for i in top.indices)
        print(f"  {nm:8s}  block[{blk}]   top5  {tops}")

    # (b) BLOCK-ONLY probe: read pose from JUST the 4 block dims. This is the direct,
    # gauge-invariant grounding test (a linear map undoes any constant heading-frame
    # rotation). High heading flip here = heading is NOT in the block.
    res = {"chance": chance_baseline(Str, Sva)}
    for nm, fac in [("linear", make_linear_probe), ("mlp", make_mlp_probe)]:
        torch.manual_seed(C.SEED)
        p = train_probe(fac(K), Ztr[:, :K], state_to_target(Str), epochs=a.probe_epochs, device=a.device)
        res[nm] = evaluate_probe(p, Zva[:, :K], Sva, device=a.device)
    save_probe_table(res, report_dir / "block_probe.csv")
    print(f"\n-- BLOCK-ONLY probe (pose from the {K} block dims; high heading flip = heading NOT grounded) --")
    for nm in ("chance", "linear", "mlp"):
        m = res[nm]
        print(f"  {nm:7s}  theta_flip {m['theta_flip_pct']:6.2f}  theta_mae {m['theta_mae_deg']:6.2f}  "
              f"x_r2 {m['x_r2']:+.3f}  y_r2 {m['y_r2']:+.3f}")

    # actuator-recovery readout (only meaningful with --learn-coeffs): does the learnable
    # block a_v recover the dataset's true actuator gain, like the ego model does?
    if a.learn_coeffs:
        import h5py
        with h5py.File(data_path, "r") as _f:
            true_gain = float(_f.attrs.get("actuator_gain", 1.0))
        learned_av = model.predictor.log_a_v.exp().item()
        print(f"\nactuator recovery: true gain={true_gain:.3f}  ->  learned block a_v={learned_av:.4f}  "
              f"a_omega={model.predictor.log_a_omega.exp().item():.4f}  (abs error {abs(learned_av - true_gain):.4f})")
        with open(report_dir / "actuator_recovery.csv", "w", newline="") as _fc:
            _w = csv.writer(_fc); _w.writerow(["true_gain", "learned_a_v", "a_omega"])
            _w.writerow([f"{true_gain:.4f}", f"{learned_av:.4f}", f"{model.predictor.log_a_omega.exp().item():.4f}"])

    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
