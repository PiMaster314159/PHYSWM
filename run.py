"""Run one full experiment end to end from the command line.

    python run.py                                   # all defaults from config.py
    python run.py --predictor-mode physics --pred-step 4
    python run.py --predictor-mode physics --lam-phys 1.0      # soft physics prior
    python run.py --predictor-mode physics --lock-pose         # hard architectural prior
    python run.py --run run02_64x64 --grid-size 64 --epochs 10

Each invocation: collects the dataset if missing, trains, probes (saves the
metrics table), and writes latent + reconstruction figures into
results/<EXPERIMENT>/. The experiment name is auto-built from the knobs, exactly
like config.py, so folders never collide and are self-describing.

Sweep from PowerShell:
    foreach ($m in "residual","physics") {
      foreach ($s in 1,4) {
        python run.py --predictor-mode $m --pred-step $s --epochs 10
      }
    }

Each run is a fresh process, so there is no stale-config state between runs.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import config as C
from data.collect import collect_dataset
from models.jepa import JEPA, state_to_target
from models.dataset import make_dataloaders
from models.train import train_jepa, plot_history
from eval.probe import run_probe, extract_latents, plot_latent_pca, plot_latent_probe_axes
from eval.reconstruct import train_decoder, plot_reconstructions, plot_pose_reconstructions


def parse_args():
    p = argparse.ArgumentParser(description="Run one JEPA experiment end to end.")
    # dataset
    p.add_argument("--run", default=C.RUN, help="dataset name (.h5 stem)")
    p.add_argument("--grid-size", type=int, default=C.GRID_SIZE)
    p.add_argument("--nose-marker", default=C.RENDER_MARKER, choices=["none", "dot", "ring"],
                   help="heading cue baked into a freshly collected dataset (grayscale if set)")
    # model + horizon (the usual sweep axes)
    p.add_argument("--predictor-mode", default=C.PREDICTOR_MODE,
                   choices=["mlp", "residual", "physics"])
    p.add_argument("--pred-step", type=int, default=C.PRED_STEP)
    p.add_argument("--latent-dim", type=int, default=C.LATENT_DIM)
    # physics prior (physics mode only)
    p.add_argument("--lam-phys", type=float, default=C.LAM_PHYS,
                   help="physics-consistency loss weight (the soft prior)")
    p.add_argument("--lock-pose", action="store_true", default=C.PHYSICS_LOCK_POSE,
                   help="hard architectural prior: MLP cannot touch dims 0,1,2")
    # privileged state anchor (any predictor mode): pull dims 0..3 toward standardized pose
    p.add_argument("--lam-anchor", type=float, default=0.0,
                   help="privileged state anchor on the first 4 latent dims (standardized pose). 0 = off (pure SIGReg JEPA)")
    p.add_argument("--lam-readout", type=float, default=0.0,
                   help="privileged state via a LINEAR readout head over the whole latent (does not fight SIGReg; code stays distributed). 0 = off")
    # training
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--lam", type=float, default=C.LAM)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--probe-epochs", type=int, default=40)
    # bookkeeping
    p.add_argument("--note", default=C.NOTE, help="optional label appended to the experiment name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-figures", action="store_true", help="skip latent + reconstruction figures")
    return p.parse_args()


def build_model(a) -> JEPA:
    """A JEPA in the requested mode, with the physics dt matched to the horizon."""
    m = JEPA(grid_size=a.grid_size, latent_dim=a.latent_dim,
             predictor_mode=a.predictor_mode, predictor_lock_pose=a.lock_pose,
             state_head=a.lam_readout > 0)
    m.predictor.dt = a.pred_step * C.DT   # physics integrates over the full horizon; no-op for mlp/residual
    return m


def main():
    a = parse_args()

    # derive experiment identity from the chosen knobs (mirrors config.py)
    tag = f"{a.predictor_mode}_s{a.pred_step}_e{a.epochs}"
    if a.predictor_mode == "physics":
        if a.lam_phys > 0:  tag += f"_lp{a.lam_phys:g}"
        if a.lock_pose:     tag += "_lock"
    if a.lam_anchor > 0:    tag += f"_anc{a.lam_anchor:g}"
    if a.lam_readout > 0:   tag += f"_rd{a.lam_readout:g}"
    experiment = f"{a.run}_{tag}" + (f"_{a.note}" if a.note else "")
    data_path  = C.DATASETS_DIR / f"{a.run}.h5"
    ckpt_path  = C.CHECKPOINTS_DIR / f"{experiment}.pt"
    report_dir = C.RESULTS_DIR / experiment
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {experiment} ===")
    phys_str = f"  lam_phys={a.lam_phys:g}  lock_pose={a.lock_pose}" if a.predictor_mode == "physics" else ""
    print(f"device={a.device}  data={data_path.name}  step={a.pred_step}  mode={a.predictor_mode}{phys_str}")

    # collect the dataset if it does not exist yet
    if not data_path.exists():
        collect_dataset(data_path, grid_size=a.grid_size, marker=a.nose_marker)

    # train
    torch.manual_seed(C.SEED)
    need_state = a.lam_anchor > 0 or a.lam_readout > 0   # privileged state loaded/standardized when supervising
    train_dl, val_dl = make_dataloaders(data_path, batch_size=a.batch_size, seed=C.SEED,
                                        return_state=need_state, step=a.pred_step)
    anchor_mean = anchor_std = None
    if need_state:                    # fixed train-set stats so the standardized target is stable
        T = state_to_target(torch.from_numpy(train_dl.dataset.states).float())
        anchor_mean, anchor_std = T.mean(0), T.std(0) + 1e-6
    model = build_model(a)
    history = train_jepa(model, train_dl, val_dl, epochs=a.epochs, lr=a.lr, lam=a.lam,
                         lam_phys=a.lam_phys, lam_anchor=a.lam_anchor, lam_readout=a.lam_readout,
                         anchor_mean=anchor_mean, anchor_std=anchor_std,
                         device=a.device, save_best_to=ckpt_path)
    plot_history(history, save_to=report_dir / "training_curve.png")

    # reload the best-val checkpoint into a fresh model of the SAME mode (so the
    # state_dict matches, including physics scale params)
    eval_model = build_model(a)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=a.device))

    # probe (pass the model so it uses the right mode) and save the metrics table
    run_probe(model=eval_model, data_path=data_path, latent_dim=a.latent_dim,
              probe_epochs=a.probe_epochs, device=a.device, seed=C.SEED, save_dir=report_dir)

    # anchored runs: per-dim correlation of the FIRST 4 dims with pose, so we can see
    # whether the anchor actually pinned interpretable axes (d0~x d1~y d2~cos d3~sin) or
    # whether SIGReg kept the code distributed despite it. Mirrors the ego per-dim block.
    if need_state:
        _, cdl = make_dataloaders(data_path, batch_size=256, seed=C.SEED, return_state=True, step=a.pred_step)
        Zc, Sc = extract_latents(eval_model, cdl, a.device)
        Tc = state_to_target(Sc)
        Zc4 = Zc[:, :4] - Zc[:, :4].mean(0, keepdim=True)
        print("\n-- per-dim correlation of latent dims 0..3 with pose (anchored) --")
        with open(report_dir / "block_correlations.csv", "w", newline="") as f:
            import csv as _csv
            w = _csv.writer(f); w.writerow(["pose", "d0", "d1", "d2", "d3"])
            for j, nm in enumerate(["x", "y", "cos_th", "sin_th"]):
                tc = Tc[:, j] - Tc[:, j].mean()
                corr = (Zc4 * tc.unsqueeze(1)).mean(0) / (Zc[:, :4].std(0) * Tc[:, j].std() + 1e-8)
                print(f"  {nm:7s}  " + "  ".join(f"d{d}={corr[d]:+.2f}" for d in range(4)))
                w.writerow([nm] + [f"{corr[d]:.4f}" for d in range(4)])

    # figures
    if not a.no_figures:
        _, vdl = make_dataloaders(data_path, batch_size=256, seed=C.SEED,
                                  return_state=True, step=a.pred_step)
        Z, S = extract_latents(eval_model, vdl, a.device)
        plot_latent_pca(Z, S, save_to=report_dir / "latent_pca.png")
        plot_latent_probe_axes(Z, S, save_to=report_dir / "latent_probe_axes.png")

        tdl, _ = make_dataloaders(data_path, seed=C.SEED, return_state=True, step=a.pred_step)
        decoder = train_decoder(eval_model, tdl, device=a.device)
        plot_reconstructions(eval_model, decoder, vdl, device=a.device,
                             save_to=report_dir / "reconstructions.png")
        plot_pose_reconstructions(eval_model, vdl, device=a.device,
                                  save_to=report_dir / "pose_reconstructions.png")

    # for physics runs, the learned unit-bridging scales are an interesting readout
    if a.predictor_mode == "physics":
        print(f"learned scales: a_pos={eval_model.predictor.log_a_pos.exp().item():.3f}  "
              f"a_theta={eval_model.predictor.log_a_theta.exp().item():.3f}")

    print(f"done -> {report_dir}")


if __name__ == "__main__":
    main()
