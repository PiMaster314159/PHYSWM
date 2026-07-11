#!/usr/bin/env python
# coding: utf-8

# # Training loop
# 
# Where the JEPA actually learns. Runs the standard PyTorch update loop, logs loss components and the collapse monitor (mean per-dim latent std), checkpoints the best-val model, and plots training curves.
# 
# > To generate `models/train.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python models/train.ipynb
# > ```

# ## Path setup

# In[ ]:


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH, CKPT_PATH, LR, LAM, LAM_PHYS, EPOCHS


# ## Imports

# In[ ]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Union

from models.components import state_to_target, sigreg_loss
try:
    from models.jepa import JEPA, jepa_loss, sigreg_loss, state_to_target
    from models.dataset import make_dataloaders
except ImportError:
    from jepa import JEPA, jepa_loss, sigreg_loss, state_to_target
    from dataset import make_dataloaders


# ## Evaluate
# 
# Runs a forward-only pass over a loader and returns average pred loss, SIGReg, and the collapse monitor (mean per-dim latent std). Called at the end of each epoch on the val set.

# In[ ]:


@torch.no_grad()
def evaluate(
    model: JEPA,
    dl,
    device: str,
    lam: float = LAM,
    max_batches: Optional[int] = None,
) -> dict:
    """Mean pred/sigreg loss and latent std over a loader, no gradients.

    Parameters
    ----------
    model : JEPA
    dl : DataLoader
    device : str
    lam : float
        Used to compute the total for the returned dict. Default from config.
    max_batches : int, optional
        Cap on batches evaluated. Useful for quick val passes.

    Returns
    -------
    dict with keys pred, sigreg, latent_std, total (all floats).
    """
    model.eval()
    preds, sigs, stds = [], [], []
    for i, batch in enumerate(dl):
        if max_batches is not None and i >= max_batches:
            break
        out = model(
            batch["frame"].to(device),
            batch["action"].to(device),
            batch["next_frame"].to(device),
        )
        preds.append(F.mse_loss(out["pred_next_z"], out["target_next_z"].detach()).item())
        sigs.append(sigreg_loss(out["z"]).item())
        stds.append(out["z"].std(dim=0).mean().item())
    model.train()
    n         = max(len(preds), 1)
    pred_mean = sum(preds) / n
    sig_mean  = sum(sigs)  / n
    return {
        "pred":       pred_mean,
        "sigreg":     sig_mean,
        "latent_std": sum(stds) / n,
        "total":      pred_mean + lam * sig_mean,
    }


# ## Training loop
# 
# Standard PyTorch update loop. Logs loss components and the collapse monitor every `log_every` steps, evaluates on val at the end of each epoch, and saves the best-val-pred checkpoint.
# 
# Best-checkpointing matters here: prediction quality tends to peak after 1-2 epochs and then degrade as SIGReg keeps reshaping the latent space. `save_best_to` keeps the peak, `save_to` keeps the final epoch.

# In[ ]:


def train_jepa(
    model: JEPA,
    train_dl,
    val_dl=None,
    epochs: int = EPOCHS,
    lr: float = LR,
    lam: float = LAM,
    lam_phys: float = LAM_PHYS,
    lam_anchor: float = 0.0,
    lam_readout: float = 0.0,
    lam_readout_pred: float = 0.0,
    anchor_mean: Optional[torch.Tensor] = None,
    anchor_std: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
    log_every: int = 50,
    max_batches: Optional[int] = None,
    save_to: Optional[Union[str, Path]] = None,
    save_best_to: Optional[Union[str, Path]] = None,
) -> dict:
    """Train a JEPA model.

    Parameters
    ----------
    model : JEPA
    train_dl, val_dl : DataLoader
    epochs : int
    lr : float
        Adam learning rate.
    lam : float
        SIGReg weight. With B-scaled SIGReg, lam=0.01 keeps pred and sigreg
        on comparable scales. Tune after inspecting the first training curves.
    lam_phys : float
        Physics-consistency weight (physics mode only). Pulls the encoded next
        pose toward the kinematic prediction. 0 disables it. Default from config.
    device : str, optional
        Defaults to cuda if available, else cpu.
    log_every : int
        Print and record a training entry every this many steps.
    max_batches : int, optional
        Cap batches per epoch. Useful for smoke tests.
    save_to : str or Path, optional
        Save the final checkpoint here.
    save_best_to : str or Path, optional
        Save the lowest-val-pred checkpoint here. Usually the one you want,
        since this task tends to peak early and then degrade.

    epochs, lr, lam, lam_phys default from config.py.

    Returns
    -------
    dict with keys train and val, each a list of per-log-step dicts.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history  = {"train": [], "val": []}
    best_val = float("inf")
    step     = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_dl):
            if max_batches is not None and i >= max_batches:
                break

            out = model(
                batch["frame"].to(device),
                batch["action"].to(device),
                batch["next_frame"].to(device),
            )
            s_target = s_next_target = None
            if (lam_anchor > 0 or lam_readout > 0 or lam_readout_pred > 0) and "state" in batch:
                t = state_to_target(batch["state"]).to(device)
                if anchor_mean is not None:                  # standardize to match BN/SIGReg scale
                    t = (t - anchor_mean.to(device)) / anchor_std.to(device)
                s_target = t
                if lam_readout_pred > 0 and "next_state" in batch:   # same standardization, next pose
                    tn = state_to_target(batch["next_state"]).to(device)
                    if anchor_mean is not None:
                        tn = (tn - anchor_mean.to(device)) / anchor_std.to(device)
                    s_next_target = tn
            loss, parts = jepa_loss(out, lam=lam, lam_phys=lam_phys, s_target=s_target,
                                    lam_anchor=lam_anchor, lam_readout=lam_readout,
                                    s_next_target=s_next_target, lam_readout_pred=lam_readout_pred)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if step % log_every == 0:
                parts["step"]       = step
                parts["latent_std"] = out["z"].std(dim=0).mean().item()
                history["train"].append(parts)
                phys_str = f"  phys {parts['phys']:.4f}" if "phys" in parts else ""
                anc_str  = f"  anchor {parts['anchor']:.4f}" if "anchor" in parts else ""
                rd_str   = f"  readout {parts['readout']:.4f}" if "readout" in parts else ""
                rdp_str  = f"  readout_pred {parts['readout_pred']:.4f}" if "readout_pred" in parts else ""
                print(f"  step {step:5d}  total {parts['total']:.4f}  "
                      f"pred {parts['pred']:.4f}  sigreg {parts['sigreg']:.4f}{phys_str}{anc_str}{rd_str}{rdp_str}  "
                      f"latent_std {parts['latent_std']:.3f}")

        if val_dl is not None:
            v         = evaluate(model, val_dl, device, lam=lam, max_batches=50)
            v["step"] = step
            history["val"].append(v)
            print(f"epoch {epoch+1}/{epochs}  val pred {v['pred']:.4f}  "
                  f"val sigreg {v['sigreg']:.4f}  latent_std {v['latent_std']:.3f}")

            if save_best_to is not None and v["pred"] < best_val:
                best_val = v["pred"]
                Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_best_to)
                print(f"  new best val pred {best_val:.4f} -> {save_best_to}")

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_to)
        print(f"saved final checkpoint -> {save_to}")

    return history


# ## Plot history
# 
# Two panels: loss components over steps (with val pred overlaid) and the collapse monitor. `latent_std` should stay near 1 throughout training. A drop toward 0 means collapse is winning.

# In[ ]:


def plot_history(
    history: dict,
    save_to: Optional[Union[str, Path]] = None,
):
    """Plot training loss curves and the collapse monitor side by side.

    Parameters
    ----------
    history : dict
        Returned by train_jepa. Keys train and val.
    save_to : str or Path, optional
        Save the figure here if provided.
    """
    tr    = history["train"]
    steps = [h["step"] for h in tr]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, [h["pred"]   for h in tr], label="pred")
    axes[0].plot(steps, [h["sigreg"] for h in tr], label="sigreg")
    axes[0].plot(steps, [h["total"]  for h in tr], label="total", lw=2, color="k")
    if history["val"]:
        val_steps = [h["step"] for h in history["val"]]
        axes[0].scatter(val_steps, [h["pred"] for h in history["val"]],
                        color="red", zorder=5, label="val pred")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("training losses")
    axes[0].legend()

    axes[1].plot(steps, [h["latent_std"] for h in tr], color="green")
    axes[1].axhline(1.0, ls="--", color="gray", alpha=0.6, label="target ~1")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean per-dim latent std")
    axes[1].set_title("collapse monitor")
    axes[1].legend()

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110)
        print(f"wrote {save_to}")
    return fig


# ## Tests
# 
# Smoke test on real data: 2 epochs capped at 30 batches. Verifies that training records are produced, loss stays finite, and the latent does not collapse.

# In[6]:


def _test_train():
    assert DATA_PATH.exists(), f"dataset not found: {DATA_PATH}"

    train_dl, val_dl = make_dataloaders(DATA_PATH, batch_size=32)
    g     = train_dl.dataset.grid_size
    model = JEPA(grid_size=g, latent_dim=128)

    history = train_jepa(
        model, train_dl, val_dl,
        epochs=2, max_batches=30, lam=0.01, log_every=10,
    )

    assert len(history["train"]) > 0, "no training records logged"
    last = history["train"][-1]
    assert torch.isfinite(torch.tensor(last["total"])), "loss is non-finite"
    assert last["latent_std"] > 0.1, f"latent collapsed: std={last['latent_std']:.3f}"
    assert len(history["val"]) == 2, "expected one val record per epoch"

    fig = plot_history(history)
    assert fig is not None
    plt.close(fig)

    print("All training tests passed.")


# In[7]:


if __name__ == "__main__":
    _test_train()

