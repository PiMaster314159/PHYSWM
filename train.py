"""Train + evaluate any world model on an existing dataset — one entry point for all three.

    python train.py --model jepa     --run run07_64x64_ring --grid-size 64 --lam-anchor 1 --lam-anchor-pred 1
    python train.py --model ego      --run run07_64x64_ring --grid-size 64 --learn-coeffs --lam-anchor 1 --lam-anchor-pred 1
    python train.py --model grounded --run run07_64x64_ring --grid-size 64 --learn-coeffs --lam-anchor 1 --lam-anchor-pred 1

Training is uniform via model.loss (representation_loss + pose_supervision). --lam-anchor /
--lam-anchor-pred are the pose-supervision weights for EVERY model (jepa's old readout is the
same term). Checkpoints + results land in checkpoints/<run>/<model>/<tag> and results/<...>.
Per-model diagnostics live in eval/<model>_eval.py. Ego's --rollout-k switches to multi-step
rollout training. This runner does NOT collect data; build the dataset first.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import csv
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import torch
import torch.nn.functional as F
torch.set_num_threads(4)

import config as C
from models.components import state_to_target, pose_stats
from models.jepa import JEPA
from models.state_ae import EgoWorldModel, ego_rollout_loss
from models.grounded import GroundedJEPA
from models.dataset import make_dataloaders, make_rollout_dataloaders, set_default_n_frames
from eval import ego_eval, grounded_eval, jepa_eval

EVAL = {"ego": ego_eval.evaluate, "grounded": grounded_eval.evaluate, "jepa": jepa_eval.evaluate}


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate a world model (ego / grounded / jepa).")
    p.add_argument("--model", required=True, choices=["ego", "grounded", "jepa"])
    # shared
    p.add_argument("--run", default=C.RUN, help="dataset name (.h5 stem); must already exist")
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--n-frames", type=int, default=C.N_FRAMES,
                   help="frames stacked as encoder input channels (>1 = history; lets the encoder read hidden "
                        "velocity from motion -- prerequisite for throttle/bicycle dynamics)")
    p.add_argument("--latent-dim", type=int, default=C.LATENT_DIM)
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="cap batches per epoch (0 = full epoch). Shuffled, so each epoch sees a fresh random "
                        "subset -- use on big datasets to get frequent val/checkpoints without ~4k steps/epoch")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training: load the existing checkpoint for this tag and just run eval "
                        "(recover metrics after a crash). Pass the SAME flags you trained with so the tag matches")
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--note", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # pose supervision (the shared pose_supervision term; every model)
    p.add_argument("--lam-anchor", type=float, default=0.0, help="pose-supervision weight on the ENCODED latent (ego/grounded slice, jepa readout)")
    p.add_argument("--lam-anchor-pred", type=float, default=0.0, help="pose-supervision weight on the PREDICTED latent (the MPC-relevant term)")
    # anti-collapse / reconstruction (jepa+grounded use --lam; ego+grounded use recon)
    p.add_argument("--lam", type=float, default=C.LAM, help="SIGReg weight (jepa: all dims; grounded: free dims)")
    # per-model defaults (resolved in main): ego wants recon grounding on (1.0, fg 5); grounded off by default
    p.add_argument("--lam-recon", type=float, default=None, help="frame-reconstruction weight (ego default 1.0; grounded default C.LAM_RECON)")
    p.add_argument("--recon-fg-weight", type=float, default=None, help="foreground weighting of recon (ego default 5.0; grounded default C.RECON_FG_WEIGHT)")
    # jepa
    p.add_argument("--predictor-mode", default=C.PREDICTOR_MODE, choices=["mlp", "residual", "physics"])
    p.add_argument("--lam-phys", type=float, default=C.LAM_PHYS, help="physics-consistency weight (jepa physics mode)")
    p.add_argument("--lock-pose", action="store_true", default=C.PHYSICS_LOCK_POSE, help="jepa physics: mask the MLP off dims 0,1,2")
    p.add_argument("--no-figures", action="store_true", help="jepa: skip latent + reconstruction figures")
    # ego
    p.add_argument("--lam-dyn", type=float, default=1.0, help="ego state-space dynamics-consistency weight")
    p.add_argument("--lam-pred", type=float, default=1.0, help="ego predicted-state -> next-frame reconstruction weight")
    p.add_argument("--lam-var", type=float, default=1.0, help="ego variance-floor weight (anti-collapse). 0 = off")
    p.add_argument("--var-gamma", type=float, default=0.1, help="ego per-dim std floor")
    p.add_argument("--decoder", default="mlp", choices=["broadcast", "mlp"], help="ego renderer")
    p.add_argument("--residual-budget", type=float, default=0.0, help="ego gray-box: bounded learned dynamics correction")
    p.add_argument("--rollout-k", type=int, default=1, help="ego: >1 switches to K-step rollout training")
    p.add_argument("--lam-rollout", type=float, default=1.0, help="ego rollout-term weight (rollout mode)")
    # grounded
    p.add_argument("--block-dim", type=int, default=C.PHYSICS_BLOCK_DIM)
    p.add_argument("--no-lock-block", dest="lock_block", action="store_false", default=C.GROUNDED_LOCK_BLOCK, help="grounded gray-box: allow a bounded learned block correction")
    p.add_argument("--block-budget", type=float, default=C.GROUNDED_BLOCK_BUDGET, help="grounded max learned block correction (gray-box)")
    p.add_argument("--use-decoder", action="store_true", help="grounded: reconstruct frame from the block (grounding booster)")
    p.add_argument("--pred-block-weight", type=float, default=C.PRED_BLOCK_WEIGHT, help="grounded: weight on block dims in the prediction loss")
    # ego + grounded
    p.add_argument("--learn-coeffs", action="store_true", help="learn gray-box a_v/a_omega (ego + grounded); default frozen at 1")
    p.add_argument("--residual", default="none", choices=["none", "basis", "mlp"],
                   help="gray-box higher-order residual (ego + grounded): 'basis' = structured physics-basis g "
                        "(drag ~v^2, readable terms); 'mlp' = unstructured free-form net (ablation control); 'none' = off")
    p.add_argument("--lam-l1", type=float, default=0.001, help="L1 weight keeping the residual small/sparse (basis: readable coeffs; mlp: input weights)")
    return p.parse_args()


def build_model(a):
    dt = a.pred_step * C.DT
    dyn = getattr(a, "dynamics", "unicycle")             # inferred from the dataset in main()
    if a.model == "jepa":
        m = JEPA(grid_size=a.grid_size, latent_dim=a.latent_dim, predictor_mode=a.predictor_mode,
                 predictor_lock_pose=a.lock_pose, state_head=(a.lam_anchor > 0 or a.lam_anchor_pred > 0),
                 in_channels=a.n_frames, dynamics=dyn)
        m.predictor.dt = dt      # physics integrates over the full horizon; no-op for mlp/residual
        return m
    if a.model == "ego":
        return EgoWorldModel(grid_size=a.grid_size, dt=dt, residual_budget=a.residual_budget,
                             residual_mode=a.residual, learn_coeffs=a.learn_coeffs, decoder=a.decoder,
                             in_channels=a.n_frames, dynamics=dyn)
    return GroundedJEPA(grid_size=a.grid_size, latent_dim=a.latent_dim, block_dim=a.block_dim, dt=dt, dynamics=dyn,
                        lock_block=a.lock_block, block_budget=a.block_budget, residual_mode=a.residual,
                        in_channels=a.n_frames,
                        use_decoder=(a.use_decoder or a.lam_recon > 0), learn_coeffs=a.learn_coeffs)


def build_weights(a):
    w = {"anchor": a.lam_anchor, "anchor_pred": a.lam_anchor_pred}
    if a.model == "jepa":
        w.update(sigreg=a.lam, phys=a.lam_phys)
    elif a.model == "ego":
        w.update(dyn=a.lam_dyn, pred=a.lam_pred, var=a.lam_var, var_gamma=a.var_gamma,
                 fg_weight=a.recon_fg_weight, recon=a.lam_recon)
    else:  # grounded
        w.update(sigreg=a.lam, recon=a.lam_recon, fg_weight=a.recon_fg_weight, pred_block_weight=a.pred_block_weight)
    if a.residual != "none":
        w["l1"] = a.lam_l1                 # keep the residual small/sparse (basis coeffs or mlp weights)
    return w


def build_tag(a):
    if a.model == "jepa":
        tag = f"{a.predictor_mode}_s{a.pred_step}_e{a.epochs}"
        if a.predictor_mode == "physics":
            if a.lam_phys > 0:  tag += f"_lp{a.lam_phys:g}"
            if a.lock_pose:     tag += "_lock"
    else:
        tag = f"s{a.pred_step}_e{a.epochs}"
        if a.model == "ego":
            if a.lam_dyn != 1.0:      tag += f"_ld{a.lam_dyn:g}"
            if a.lam_pred != 1.0:     tag += f"_lpr{a.lam_pred:g}"
            if a.lam_var > 0:         tag += f"_v{a.lam_var:g}"
            if a.recon_fg_weight > 0: tag += f"_fg{a.recon_fg_weight:g}"
            if a.lam_recon != 1.0:    tag += f"_rec{a.lam_recon:g}"
            if a.residual_budget > 0: tag += f"_gray{a.residual_budget:g}"
            if a.rollout_k > 1:       tag += f"_roll{a.rollout_k}"
        else:  # grounded
            if not a.lock_block:           tag += f"_gray{a.block_budget:g}"
            if a.lam_recon > 0:            tag += f"_rec{a.lam_recon:g}"
            if a.recon_fg_weight > 0:      tag += f"_fg{a.recon_fg_weight:g}"
            if a.pred_block_weight != 1.0: tag += f"_pb{a.pred_block_weight:g}"
    if a.lam_anchor > 0:      tag += f"_anc{a.lam_anchor:g}"
    if a.lam_anchor_pred > 0: tag += f"_ancp{a.lam_anchor_pred:g}"
    if a.model in ("ego", "grounded") and a.learn_coeffs: tag += "_learn"
    if a.model in ("ego", "grounded") and a.residual != "none": tag += f"_g{a.residual}"   # gbasis / gmlp (distinct ablation ckpts)
    if a.rollout_k > 1 and a.model != "ego": tag += f"_roll{a.rollout_k}"   # ego adds this in its own branch
    if a.n_frames > 1: tag += f"_hist{a.n_frames}"   # frame-stack history depth
    if getattr(a, "dynamics", "unicycle") == "bicycle": tag += "_bike"   # bicycle/throttle (5-D state, a,delta action)
    if a.model == "ego": tag += "_bc" if a.decoder == "broadcast" else "_mlpdec"   # decoder token last
    if a.note: tag += f"_{a.note}"
    return tag


def loss_batch(b, need_state, device):
    batch = {"frame": b["frame"].to(device), "next_frame": b["next_frame"].to(device)}
    if need_state and "state" in b:
        vel, nvel = b.get("velocity"), b.get("next_velocity")   # bicycle carries v -> 5-D [x,y,cos,sin,v]
        batch["s_target"]      = state_to_target(b["state"], vel).to(device)
        batch["s_next_target"] = state_to_target(b["next_state"], nvel).to(device)
    return batch


@torch.no_grad()
def val_pred(model, dl, device, max_batches=50):
    """Mean next-latent prediction MSE — the checkpoint metric (available for every model)."""
    model.eval(); losses = []
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        out = model(b["frame"].to(device), b["action"].to(device), b["next_frame"].to(device))
        losses.append(F.mse_loss(out["pred_next_z"], out["target_next_z"]).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_standard(model, a, data_path, ckpt_path, report_dir):
    device, weights = a.device, build_weights(a)
    need_state = a.lam_anchor > 0 or a.lam_anchor_pred > 0
    # bake pose stats for jepa (decode_pose readout) AND for any bicycle model (so the pose loss STANDARDIZES:
    # otherwise raw MSE lets the O(1) heading dims drown out the O(0.1) velocity dim -> v is barely learned)
    if (a.model == "jepa" or getattr(a, "dynamics", "unicycle") == "bicycle") and need_state:
        stats_dl, _ = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED, return_state=True, step=a.pred_step)
        _vel = getattr(stats_dl.dataset, 'velocities', None)   # bicycle: standardize v too
        _velt = torch.from_numpy(_vel).float() if _vel is not None else None
        mean, std = pose_stats(torch.from_numpy(stats_dl.dataset.states).float(), _velt)
        model.set_pose_stats(mean.to(device), std.to(device))

    train_dl, val_dl = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                        return_state=need_state, step=a.pred_step)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, step, hist = float("inf"), 0, []
    for epoch in range(a.epochs):
        for i, b in enumerate(train_dl):
            if a.steps_per_epoch and i >= a.steps_per_epoch:   # cap epoch length on big datasets
                break
            out = model(b["frame"].to(device), b["action"].to(device), b["next_frame"].to(device))
            loss, parts = model.loss(out, loss_batch(b, need_state, device), weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); step += 1
            if step % 50 == 0:
                parts["latent_std"] = out["z"].std(0).mean().item()
                hist.append({"step": step, "epoch": epoch + 1, **parts})
                extras = "  ".join(f"{k} {parts[k]:.4f}" for k in parts if k not in ("step", "epoch", "total"))
                print(f"  step {step:5d}  total {parts['total']:.4f}  {extras}")
        v = val_pred(model, val_dl, device)
        print(f"epoch {epoch+1}/{a.epochs}  val pred {v:.4f}")
        if v < best:
            best = v; torch.save(model.state_dict(), ckpt_path); print(f"  new best {best:.4f} -> {ckpt_path}")
    _write_history(report_dir, hist)


def train_rollout(model, a, data_path, ckpt_path, report_dir):
    """Model-agnostic multi-step rollout training: encode frame_0, roll K steps, supervise decoded pose
    at every horizon (via WorldModel.rollout_loss). A wrong dynamics parameter compounds across the roll,
    so the optimizer gets real multi-step gradient the single-step loss can't provide -- the term meant to
    sharpen the predictor and, we hope, help JEPA."""
    device = a.device
    if (a.model == "jepa" or getattr(a, "dynamics", "unicycle") == "bicycle") and (a.lam_anchor > 0 or a.lam_anchor_pred > 0):   # standardize pose loss (see train_standard)
        stats_dl, _ = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED, return_state=True, step=a.pred_step)
        _vel = getattr(stats_dl.dataset, 'velocities', None)   # bicycle: standardize v too
        _velt = torch.from_numpy(_vel).float() if _vel is not None else None
        mean, std = pose_stats(torch.from_numpy(stats_dl.dataset.states).float(), _velt)
        model.set_pose_stats(mean.to(device), std.to(device))
    train_dl, val_dl = make_rollout_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                                K=a.rollout_k, step=a.pred_step)
    weights = build_weights(a); weights["rollout"] = a.lam_rollout
    print(f"rollout training: K={a.rollout_k} transitions  ({len(train_dl.dataset)} windows)")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, step, hist = float("inf"), 0, []
    to_dev = lambda b: {k: v.to(device) for k, v in b.items()}

    @torch.no_grad()
    def rollout_val(dl, max_batches=50):
        model.eval(); tot = []
        for i, b in enumerate(dl):
            if i >= max_batches:
                break
            _, parts = model.rollout_loss(to_dev(b), weights); tot.append(parts["total"])
        model.train()
        return sum(tot) / max(len(tot), 1)

    for epoch in range(a.epochs):
        for i, b in enumerate(train_dl):
            if a.steps_per_epoch and i >= a.steps_per_epoch:   # cap epoch length on big datasets
                break
            loss, parts = model.rollout_loss(to_dev(b), weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); step += 1
            if step % 50 == 0:
                row = {"step": step, "epoch": epoch + 1, **parts}
                if hasattr(model, "log_a_v"): row["a_v"] = model.log_a_v.exp().item()
                hist.append(row)
                extras = "  ".join(f"{k} {parts[k]:.4f}" for k in parts if k not in ("step", "epoch", "total"))
                print(f"  step {step:5d}  total {parts['total']:.4f}  {extras}")
        v = rollout_val(val_dl)
        print(f"epoch {epoch+1}/{a.epochs}  val total {v:.4f}")
        if v < best:
            best = v; torch.save(model.state_dict(), ckpt_path); print(f"  new best {best:.4f} -> {ckpt_path}")
    _write_history(report_dir, hist)


def _write_history(report_dir, hist):
    if not hist:
        return
    cols = list(dict.fromkeys(k for row in hist for k in row))   # union, order-preserving
    with open(report_dir / "train_history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for row in hist:
            w.writerow([row.get(c, "") for c in cols])
    print(f"wrote train_history.csv ({len(hist)} rows)")


def main():
    a = parse_args()
    if a.lam_recon is None:       # per-model defaults (ego grounds via reconstruction; grounded does not by default)
        a.lam_recon = 1.0 if a.model == "ego" else C.LAM_RECON
    if a.recon_fg_weight is None:
        a.recon_fg_weight = 5.0 if a.model == "ego" else C.RECON_FG_WEIGHT
    data_path = C.DATASETS_DIR / f"{a.run}.h5"
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path} (this runner does not collect; build it first)")

    with h5py.File(data_path, "r") as _f:              # grid + dynamics are properties of the DATASET, not CLI choices:
        ds_grid = int(_f.attrs["grid_size"])           # infer them so the model matches what produced the data
        a.dynamics = str(_f.attrs.get("dynamics", "unicycle"))
    if ds_grid != a.grid_size:
        print(f"grid_size {a.grid_size} -> {ds_grid}  (inferred from {data_path.name})")
        a.grid_size = ds_grid
    if a.dynamics == "bicycle":
        print(f"dynamics = bicycle  (5-D state [x,y,cos,sin,v], action (a,delta); inferred from {data_path.name})")

    tag = build_tag(a)                                 # after dynamics is set, so the tag carries _bike
    ckpt_path, report_dir = C.experiment_paths(a.run, a.model, tag)
    experiment = f"{a.run}/{a.model}/{tag}"
    set_default_n_frames(a.n_frames)                   # so eval/probe make_dataloaders() also return K-frame stacks

    print(f"=== {experiment} ===")
    print(f"device={a.device}  data={data_path.name}  model={a.model}  "
          f"anchor={a.lam_anchor} anchor_pred={a.lam_anchor_pred}")

    torch.manual_seed(C.SEED)
    model = build_model(a).to(a.device).train()
    if a.eval_only:                                   # recover metrics from an existing checkpoint (no training)
        if not ckpt_path.exists():
            raise SystemExit(f"--eval-only: checkpoint not found: {ckpt_path}\n(pass the SAME flags you trained with so the tag matches)")
        print(f"eval-only: loading {ckpt_path.name}")
        report_dir.mkdir(parents=True, exist_ok=True)
    elif a.rollout_k > 1:                              # multi-step rollout training (any model)
        train_rollout(model, a, data_path, ckpt_path, report_dir)
    else:
        train_standard(model, a, data_path, ckpt_path, report_dir)

    model.load_state_dict(torch.load(ckpt_path, map_location=a.device, weights_only=True))
    if getattr(a, "dynamics", "unicycle") == "bicycle":   # lean, model-agnostic eval (velocity recovery)
        from eval import bicycle_eval
        bicycle_eval.evaluate(model, a, data_path, report_dir)
    else:
        EVAL[a.model](model, a, data_path, report_dir)
    print(f"\ndone -> {report_dir}")


if __name__ == "__main__":
    main()
